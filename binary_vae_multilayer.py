from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

""" Helper Functions """
def safe_log_prob(x, eps=1e-8):
    return tf.log(tf.clip_by_value(x, eps, 1.0))


def safe_clip(x, eps=1e-8):
    return tf.clip_by_value(x, eps, 1.0)


def gs(x):
    return x.get_shape().as_list()


def softplus(x):
    '''
    Let m = max(0, x), then,

    sofplus(x) = log(1 + e(x)) = log(e(0) + e(x)) = log(e(m)(e(-m) + e(x-m)))
                         = m + log(e(-m) + e(x - m))

    The term inside of the log is guaranteed to be between 1 and 2.
    '''
    m = tf.maximum(tf.zeros_like(x), x)
    return m + tf.log(tf.exp(-m) + tf.exp(x - m))


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-softplus(-log_alpha)) + (1 - b) * (-log_alpha - softplus(-log_alpha))


def bernoulli_loglikelihood_derivitive(b, log_alpha):
    assert gs(b) == gs(log_alpha)
    sna = tf.sigmoid(-log_alpha)
    return b * sna - (1-b) * (1 - sna)


def v_from_u(u, log_alpha):
    u_prime = tf.nn.sigmoid(-log_alpha)
    v_1 = (u - u_prime) / safe_clip(1 - u_prime)
    v_1 = tf.clip_by_value(v_1, 0, 1)
    v_1 = tf.stop_gradient(v_1)
    v_1 = v_1 * (1 - u_prime) + u_prime
    v_0 = u / safe_clip(u_prime)
    v_0 = tf.clip_by_value(v_0, 0, 1)
    v_0 = tf.stop_gradient(v_0)
    v_0 = v_0 * u_prime

    v = tf.where(u > u_prime, v_1, v_0)
    v = tf.check_numerics(v, 'v sampling is not numerically stable.')
    v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors
    return v


def reparameterize(log_alpha, noise):
    return log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)


def concrete_relaxation(z, temp, log_alpha):
    return tf.sigmoid(z / temp + log_alpha)


def assert_same_shapes(*args):
    shapes = [gs(arg) for arg in args]
    s0, sr = shapes[0], shapes[1:]
    assert all([s == s0 for s in sr])


def neg_elbo(x, samples, log_alphas_inf, log_alphas_gen):
    assert len(samples) == len(log_alphas_inf) == len(log_alphas_gen)
    # compute log[q(b1|x)q(b2|b1)...q(bN|bN-1)]
    log_q_bs = []
    for b, log_alpha in zip(samples, log_alphas_inf):
        log_q_cur_given_prev = tf.reduce_sum(bernoulli_loglikelihood(b, log_alpha), axis=1)
        log_q_bs.append(log_q_cur_given_prev)
    log_q_b = tf.add_n(log_q_bs)
    # compute log[p(b1, ..., bN, x)]
    log_p_x_bs = []
    all_log_alphas_gen = [tf.zeros_like(samples[0])] + log_alphas_gen
    all_samples_gen = samples + [x]
    for b, log_alpha in zip(all_samples_gen, all_log_alphas_gen):
        log_p_next_given_cur = tf.reduce_sum(bernoulli_loglikelihood(b, log_alpha), axis=1)
        log_p_x_bs.append(log_p_next_given_cur)
    log_p_b_x = tf.add_n(log_p_x_bs)

    return -1. * (log_p_b_x - log_q_b)


""" Networks """
def encoder(x, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alpha = tf.layers.dense(2. * x - 1., num_latents, name="log_alpha")
    return log_alpha

def decoder(b, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alpha = tf.layers.dense(2. * b - 1., num_latents, name="out")
    return log_alpha

def inference_network(x, layer, num_layers, num_latents, name, reuse, sampler):
    with tf.variable_scope(name, reuse=reuse):
        log_alphas = []
        samples = []
        for l in range(num_layers):
            if l == 0:
                inp = x
            else:
                inp = samples[-1]
            log_alpha = layer(inp, num_latents, str(l), reuse)
            log_alphas.append(log_alpha)
            sample = sampler.sample(log_alpha, l)
            samples.append(sample)
    return log_alphas, samples

def generator_network(samples, layer, num_layers, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alphas = []
        for l in reversed(range(num_layers)):
            log_alpha = layer(
                samples[l],
                784 if l == 0 else num_latents, str(l), reuse
            )
            log_alphas.append(log_alpha)
    return log_alphas

def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 200, tf.nn.relu, name="q_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="q_2")
    out = tf.layers.dense(h2, 1, name="q_out")
    scale = tf.get_variable(
        "q_scale", shape=[1], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True
    )
    return scale[0] * out

""" Variable Creation """
def create_log_temp(num_latents, l):
    return tf.Variable(
        [np.log(.5) for i in range(num_latents)],
        trainable=True,
        name='log_temperature_{}'.format(l),
        dtype=tf.float32
    )

def create_eta(num_latents, l):
    return tf.Variable(
        [1.0 for i in range(num_latents)],
        trainable=True,
        name='eta_{}'.format(l),
        dtype=tf.float32
    )

class BSampler:
    def __init__(self, u):
        self.u = u
    def sample(self, log_alpha, l):
        z = reparameterize(log_alpha, self.u[l])
        b = tf.to_float(tf.stop_gradient(z > 0))
        return b

class ZSampler:
    def __init__(self, u, temp):
        self.u = u
        self.temp = temp
    def sample(self, log_alpha, l):
        z = reparameterize(log_alpha, self.u[l])
        sig_z = concrete_relaxation(z, self.temp[l], log_alpha)
        return sig_z


def main(use_reinforce=False, relaxed=False, num_epochs=300,
         batch_size=100, num_latents=200, num_layers=2, lr=.0001):
    TRAIN_DIR = "./binary_vae_test"
    if os.path.exists(TRAIN_DIR):
        print("Deleting existing train dir")
        import shutil

        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)
    sess = tf.Session()
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [batch_size, 784])
    x_im = tf.reshape(x, [batch_size, 28, 28, 1])
    tf.summary.image("x_true", x_im)
    x_binary = tf.to_float(x > .5)

    # create rebar specific variables temperature and eta
    log_temperatures = [create_log_temp(num_latents, l) for l in range(num_layers)]
    temperatures = [tf.exp(log_temp) for log_temp in log_temperatures]
    batch_temps = [tf.expand_dims(temp, 0) for temp in temperatures]
    etas = [create_eta(num_latents, l) for l in range(num_layers)]
    batch_etas = [tf.expand_dims(eta, 0) for eta in etas]

    # random uniform samples
    u = [
        tf.random_uniform([batch_size, num_latents], dtype=tf.float32)
        for l in range(num_layers)
    ]
    # create binary sampler
    b_sampler = BSampler(u)
    # generate hard forward pass
    inf_la_b, samples_b = inference_network(
        x_binary, encoder, num_layers,
        num_latents, "encoder", False, b_sampler
    )
    gen_la_b = generator_network(
        samples_b, decoder, num_layers,
        num_latents, "decoder", False
    )
    a = tf.exp(gen_la_b[-1])
    dec_log_theta = a / (1 + a)
    dec_log_theta_im = tf.reshape(dec_log_theta, [batch_size, 28, 28, 1])
    tf.summary.image("x_pred", dec_log_theta_im)

    v = [v_from_u(_u, log_alpha) for _u, log_alpha in zip(u, inf_la_b)]
    # create soft samplers
    sig_z_sampler = ZSampler(u, batch_temps)
    sig_zt_sampler = ZSampler(v, batch_temps)
    # generate soft forward passes
    inf_la_z, samples_z = inference_network(
        x, encoder, num_layers,
        num_latents, "encoder", True, sig_z_sampler
    )
    gen_la_z = generator_network(
        samples_z, decoder, num_layers,
        num_latents, "decoder", True
    )
    inf_la_zt, samples_zt = inference_network(
        x, encoder, num_layers,
        num_latents, "encoder", True, sig_zt_sampler
    )
    gen_la_zt = generator_network(
        samples_zt, decoder, num_layers,
        num_latents, "decoder", True
    )
    # create loss evaluations
    f_b = neg_elbo(x, samples_b, inf_la_b, gen_la_b)
    f_z = neg_elbo(x, samples_z, inf_la_z, gen_la_z)
    f_zt = neg_elbo(x, samples_zt, inf_la_zt, gen_la_zt)
    tf.summary.scalar("fb", tf.reduce_mean(f_b))
    tf.summary.scalar("fz", tf.reduce_mean(f_z))
    tf.summary.scalar("fzt", tf.reduce_mean(f_zt))
    generative_loss = tf.reduce_mean(f_b)
    tf.summary.scalar("loss", generative_loss)
    # create gradient evaluations for rebar
    d_f_z_d_la = [tf.gradients(f_z, la)[0] for la in inf_la_z]
    d_f_zt_d_la = [tf.gradients(f_zt, la)[0] for la in inf_la_zt]
    d_log_p_d_la = [
        bernoulli_loglikelihood_derivitive(b, la)
        for b, la in zip(samples_b, inf_la_b)
    ]
    d_f_b_d_la = [tf.gradients(f_b, la)[0] for la in inf_la_b]
    # create rebar
    batch_f_b = tf.expand_dims(f_b, 1)
    batch_f_zt = tf.expand_dims(f_zt, 1)
    rebars = []
    reinforces = []
    for l in range(num_layers):
        term1 = (batch_f_b - batch_etas[l] * batch_f_zt) * d_log_p_d_la[l]
        term2 = batch_etas[l] * (d_f_z_d_la[l] - d_f_zt_d_la[l]) + d_f_b_d_la[l]
        rebar = term1 + term2
        rebars.append(rebar)
        reinforce = batch_f_b * d_log_p_d_la[l] + d_f_b_d_la[l]
        reinforces.append(reinforce)
        tf.summary.histogram("rebar_{}".format(l), rebar)
        tf.summary.histogram("reinforce_{}".format(l), reinforce)

    # variance reduction objective
    variance_loss = tf.reduce_mean(tf.add_n([tf.square(rebar) for rebar in rebars]))

    # optimizers
    inf_opt = tf.train.AdamOptimizer(lr)
    inf_vars = [v for v in tf.trainable_variables() if "encoder" in v.name]
    # need to scale by batch size cuz tf.gradients sums
    la_grads = []
    for l in range(num_layers):
        if use_reinforce:
            la_grads.append(reinforce / batch_size)
        else:
            la_grads.append(rebar / batch_size)

    inf_grads = []
    for l in range(num_layers):
        inf_grad = tf.gradients(inf_la_b[l], inf_vars, grad_ys=la_grads[l])
        inf_grads.append(inf_grad)
    z_inf_grads = zip(*inf_grads)
    inf_grads = []
    for ig in z_inf_grads:
        grads = [g for g in ig if g is not None]
        grads = tf.add_n(grads)
        inf_grads.append(grads)
    inf_gradvars = zip(inf_grads, inf_vars)

    gen_opt = tf.train.AdamOptimizer(lr)
    gen_vars = [v for v in tf.trainable_variables() if "decoder" in v.name]
    gen_gradvars = gen_opt.compute_gradients(generative_loss, var_list=gen_vars)

    var_opt = tf.train.AdamOptimizer(lr)
    var_vars = etas + log_temperatures
    if relaxed:
        q_vars = [v for v in tf.trainable_variables() if "Q_func" in v.name]
        var_vars = var_vars + q_vars
    var_gradvars = var_opt.compute_gradients(variance_loss, var_list=var_vars)


    print("Inference")
    for g, v in inf_gradvars:
        print("    {}".format(v.name))
        print(g)
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)
    print("Generative")
    for g, v in gen_gradvars:
        print("    {}, {}".format(v.name, g))
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)
    print("Variance")
    for g, v in var_gradvars:
        print("    {}, {}".format(v.name, g))
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)

    inf_train_op = inf_opt.apply_gradients(inf_gradvars)
    gen_train_op = gen_opt.apply_gradients(gen_gradvars)
    var_train_op = var_opt.apply_gradients(var_gradvars)

    if use_reinforce:
        with tf.control_dependencies([gen_train_op, inf_train_op]):
            train_op = tf.no_op()
    else:
        with tf.control_dependencies([gen_train_op, inf_train_op, var_train_op]):
            train_op = tf.no_op()

    #LOOK AT REINFORCE GRADIENTS AND MAKE SURE THEY ARE DIFFERENT

    test_loss = tf.Variable(600, trainable=False, name="test_loss", dtype=tf.float32)
    #rebar_var = tf.Variable(np.zeros([batch_size, num_latents]), trainable=False, name="rebar_variance", dtype=tf.float32)
    #reinforce_var = tf.Variable(np.zeros([batch_size, num_latents]), trainable=False, name="reinforce_variance",
    #                            dtype=tf.float32)
    #est_diffs = tf.Variable(np.zeros([batch_size, num_latents]), trainable=False, name="estimator_differences",
    #                        dtype=tf.float32)
    tf.summary.scalar("test_loss", test_loss)
    #tf.summary.histogram("rebar_variance", rebar_var)
    #tf.summary.histogram("reinforace_variance", reinforce_var)
    #tf.summary.histogram("estimator_diffs", est_diffs)
    summ_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TRAIN_DIR)
    sess.run(tf.global_variables_initializer())

    iters_per_epoch = dataset.train.num_examples // batch_size
    iters = iters_per_epoch * num_epochs
    for i in range(iters):
        batch_xs, _ = dataset.train.next_batch(batch_size)
        if i % 100 == 0:
            loss, _, sum_str = sess.run([generative_loss, train_op, summ_op], feed_dict={x: batch_xs})
            summary_writer.add_summary(sum_str, i)
            print(i, loss)
        else:
            loss, _ = sess.run([generative_loss, train_op], feed_dict={x: batch_xs})

        if i % iters_per_epoch == 0:
            # epoch over, run test data
            losses = []
            for _ in range(dataset.test.num_examples // batch_size):
                batch_xs, _ = dataset.test.next_batch(batch_size)
                losses.append(sess.run(generative_loss, feed_dict={x: batch_xs}))
            tl = np.mean(losses)
            print("Test loss = {}".format(tl))
            sess.run(test_loss.assign(tl))


if __name__ == "__main__":
    main()