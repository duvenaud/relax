from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
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


def v_from_u(u, log_alpha, force_same=True):
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
    if force_same:
        v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors
    return v


def reparameterize(log_alpha, noise):
    return log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)


def concrete_relaxation(z, temp):
    return tf.sigmoid(z / temp)


def mu_prop_relaxation(log_alpha, noise, temp):
    l_u = safe_log_prob(noise) - safe_log_prob(1 - noise)
    return (((tf.square(temp) + temp + 1.) / (temp + 1.)) * log_alpha + l_u) / temp


def neg_elbo(x, samples, log_alphas_inf, log_alphas_gen, prior, log=False):
    assert len(samples) == len(log_alphas_inf) == len(log_alphas_gen)
    # compute log[q(b1|x)q(b2|b1)...q(bN|bN-1)]
    log_q_bs = []
    for b, log_alpha in zip(samples, log_alphas_inf):
        log_q_cur_given_prev = tf.reduce_sum(bernoulli_loglikelihood(b, log_alpha), axis=1)
        log_q_bs.append(log_q_cur_given_prev)
    log_q_b = tf.add_n(log_q_bs)
    # compute log[p(b1, ..., bN, x)]
    log_p_x_bs = []

    all_log_alphas_gen = list(reversed(log_alphas_gen)) + [prior]
    all_samples_gen = [x] + samples
    for b, log_alpha in zip(all_samples_gen, all_log_alphas_gen):
        log_p_next_given_cur = tf.reduce_sum(bernoulli_loglikelihood(b, log_alpha), axis=1)
        log_p_x_bs.append(log_p_next_given_cur)
    log_p_b_x = tf.add_n(log_p_x_bs)

    if log:
        for i, log_q in enumerate(log_q_bs):
            log_p = log_p_x_bs[i+1]
            kl = tf.reduce_mean(log_q - log_p)
            tf.summary.scalar("kl_{}".format(i), kl)
        tf.summary.scalar("log_p_x_given_b", tf.reduce_mean(log_p_x_bs[0]))
    return -1. * tf.reduce_mean((log_p_b_x - log_q_b)), log_q_bs


""" Networks """
def encoder(x, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alpha = tf.layers.dense(2. * x - 1., num_latents, name="log_alpha")
    return log_alpha

def decoder(b, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alpha = tf.layers.dense(2. * b - 1., num_latents, name="log_alpha")
    return log_alpha

def inference_network(x, mean, layer, num_layers, num_latents, name, reuse, sampler):
    with tf.variable_scope(name, reuse=reuse):
        log_alphas = []
        samples = []
        for l in range(num_layers):
            if l == 0:
                inp = ((x - mean) + 1.) / 2.
            else:
                inp = samples[-1]
            log_alpha = layer(inp, num_latents, str(l), reuse)
            log_alphas.append(log_alpha)
            sample = sampler.sample(log_alpha, l)
            samples.append(sample)
    return log_alphas, samples

def generator_network(samples, output_bias, layer, num_layers, num_latents, name, reuse, sampler=None, prior=None):
    with tf.variable_scope(name, reuse=reuse):
        log_alphas = []
        PRODUCE_SAMPLES = False
        if samples is None:
            PRODUCE_SAMPLES = True
            prior_log_alpha = prior
            samples = [None for l in range(num_layers)]
            samples[-1] = sampler.sample(prior_log_alpha, num_layers-1)
        for l in reversed(range(num_layers)):
            log_alpha = layer(
                samples[l],
                784 if l == 0 else num_latents, str(l), reuse
            )
            if l == 0:
                log_alpha = log_alpha + output_bias
            log_alphas.append(log_alpha)
            if l > 0 and PRODUCE_SAMPLES:
                samples[l-1] = sampler.sample(log_alpha, l-1)
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
def create_log_temp():
    return tf.Variable(
        np.log(.5),
        trainable=True,
        name='log_temperature',
        dtype=tf.float32
    )


def create_eta():
    return tf.Variable(
        1.0,
        trainable=True,
        name='eta',
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
        sig_z = concrete_relaxation(z, self.temp[l])
        return sig_z


def main(use_reinforce=False, relaxed=False, learn_prior=True, num_epochs=820,
         batch_size=24, num_latents=200, num_layers=2, lr=.0001, test_bias=False):
    TRAIN_DIR = "./binary_vae_3_der_diff"
    if os.path.exists(TRAIN_DIR):
        print("Deleting existing train dir")
        import shutil

        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)

    sess = tf.Session()
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_binary = (dataset.train.images > .5).astype(np.float32)
    train_mean = np.mean(train_binary, axis=0, keepdims=True)
    train_output_bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.).astype(np.float32)

    x = tf.placeholder(tf.float32, [batch_size, 784])
    x_im = tf.reshape(x, [batch_size, 28, 28, 1])
    tf.summary.image("x_true", x_im)
    x_binary = tf.to_float(x > .5)

    # make prior for top b
    p_prior = tf.Variable(
        tf.zeros([num_latents],
        dtype=tf.float32),
        trainable=learn_prior,
        name='p_prior',
    )
    # create rebar specific variables temperature and eta
    log_temperatures = [create_log_temp() for l in range(num_layers)]
    for lt in log_temperatures:
        tf.summary.scalar(lt.name, lt)
    temperatures = [tf.exp(log_temp) for log_temp in log_temperatures]


    # random uniform samples
    u = [
        tf.random_uniform([batch_size, num_latents], dtype=tf.float32)
        for l in range(num_layers)
    ]
    # create binary sampler
    b_sampler = BSampler(u)
    # generate hard forward pass
    encoder_name = "encoder"
    decoder_name = "decoder"
    inf_la_b, samples_b = inference_network(
        x_binary, train_mean,
        encoder, num_layers,
        num_latents, encoder_name, False, b_sampler
    )
    gen_la_b = generator_network(
        samples_b, train_output_bias,
        decoder, num_layers,
        num_latents, decoder_name, False
    )
    # produce reconstruction summary
    a = tf.exp(gen_la_b[-1])
    dec_log_theta = a / (1 + a)
    dec_log_theta_im = tf.reshape(dec_log_theta, [batch_size, 28, 28, 1])
    tf.summary.image("x_pred", dec_log_theta_im)
    # produce samples
    _samples_la_b = generator_network(
        None, train_output_bias,
        decoder, num_layers,
        num_latents, decoder_name, True, sampler=b_sampler, prior=p_prior
    )
    a = tf.exp(_samples_la_b[-1])
    dec_log_theta = a / (1 + a)
    dec_log_theta_im = tf.reshape(dec_log_theta, [batch_size, 28, 28, 1])
    tf.summary.image("x_sample", dec_log_theta_im)

    v = [v_from_u(_u, log_alpha) for _u, log_alpha in zip(u, inf_la_b)]
    # create soft samplers
    sig_z_sampler = ZSampler(u, temperatures)
    sig_zt_sampler = ZSampler(v, temperatures)
    # generate soft forward passes
    inf_la_z, samples_z = inference_network(
        x_binary, train_mean,
        encoder, num_layers,
        num_latents, encoder_name, True, sig_z_sampler
    )
    gen_la_z = generator_network(
        samples_z, train_output_bias,
        decoder, num_layers,
        num_latents, decoder_name, True
    )
    inf_la_zt, samples_zt = inference_network(
        x_binary, train_mean,
        encoder, num_layers,
        num_latents, encoder_name, True, sig_zt_sampler
    )
    gen_la_zt = generator_network(
        samples_zt, train_output_bias,
        decoder, num_layers,
        num_latents, decoder_name, True
    )
    # create loss evaluations
    f_b, log_q_bs = neg_elbo(x_binary, samples_b, inf_la_b, gen_la_b, p_prior, log=True)
    f_z, _ = neg_elbo(x_binary, samples_z, inf_la_z, gen_la_z, p_prior)
    f_zt, _ = neg_elbo(x_binary, samples_zt, inf_la_zt, gen_la_zt, p_prior)
    log_q_b = tf.add_n([tf.reduce_mean(log_q_b) for log_q_b in log_q_bs])

    encoder_params = [v for v in tf.global_variables() if "encoder" in v.name]
    decoder_params = [v for v in tf.global_variables() if "decoder" in v.name]
    if learn_prior:
        decoder_params.append(p_prior)

    model_opt = tf.train.AdamOptimizer(lr, beta2=.99999)
    # compute gradients and store gradients 50% speed increase
    grads = {}
    vals = [f_b, f_z, f_zt, log_q_b]
    names = ['f_b', 'f_z', 'f_zt', 'log_q_b']
    for val, name in zip(vals, names):
        val_gradvars = model_opt.compute_gradients(val, var_list=encoder_params)
        grads[name] = {}
        for g, v in val_gradvars:
            grads[name][v.name] = g

    grad_vars = []
    etas = []
    variance_objectives = []
    rebars = []
    reinforces = []
    for param in encoder_params:
        print(param.name)
        # create eta
        eta = create_eta()
        tf.summary.scalar(eta.name, eta)

        # # non reinforce gradient
        d_fb_dt = grads['f_b'][param.name]
        d_fz_dt = grads['f_z'][param.name]
        d_fzt_dt = grads['f_zt'][param.name]
        d_log_q_dt = grads['log_q_b'][param.name]

        reinforce = f_b * d_log_q_dt + d_fb_dt
        rebar = (f_b - eta * f_zt) * d_log_q_dt + eta * (d_fz_dt - d_fzt_dt) + d_fb_dt
        tf.summary.histogram(param.name, param)
        tf.summary.histogram(param.name+"_der_diff", (d_fz_dt - d_fzt_dt))
        tf.summary.histogram(param.name+"_reinforce", reinforce)
        tf.summary.histogram(param.name+"_rebar", rebar)
        if use_reinforce:
            grad_vars.append((reinforce, param))
        else:
            grad_vars.append((rebar, param))
        etas.append(eta)
        variance_objectives.append(tf.reduce_mean(tf.square(rebar)))
        rebars.append(rebar)
        reinforces.append(reinforce)

    decoder_gradvars = model_opt.compute_gradients(f_b, var_list=decoder_params)
    for g, v in decoder_gradvars:
        print(v.name)
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)
    grad_vars.extend(decoder_gradvars)

    variance_objective = tf.add_n(variance_objectives)
    model_train_op = model_opt.apply_gradients(grad_vars)
    if use_reinforce:
        train_op = model_train_op
    else:
        variance_opt = tf.train.AdamOptimizer(10. * lr, beta2=.99999)
        variance_gradvars = variance_opt.compute_gradients(variance_objective, var_list=etas+log_temperatures)
        for g, v in variance_gradvars:
            tf.summary.histogram(v.name+"_gradient", g)
        variance_train_op = variance_opt.apply_gradients(variance_gradvars)
        with tf.control_dependencies([model_train_op, variance_train_op]):
            train_op = tf.no_op()

    tf.summary.scalar("fb", f_b)
    tf.summary.scalar("fz", f_z)
    tf.summary.scalar("fzt", f_zt)
    total_loss = f_b
    tf.summary.scalar("loss", total_loss)

    test_loss = tf.Variable(1000, trainable=False, name="test_loss", dtype=tf.float32)
    tf.summary.scalar("test_loss", test_loss)
    summ_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TRAIN_DIR)
    sess.run(tf.global_variables_initializer())

    iters_per_epoch = dataset.train.num_examples // batch_size
    iters = iters_per_epoch * num_epochs
    t = time.time()
    for i in range(iters):
        batch_xs, _ = dataset.train.next_batch(batch_size)
        if i % 1000 == 0:
            loss, _, sum_str = sess.run([total_loss, train_op, summ_op], feed_dict={x: batch_xs})
            summary_writer.add_summary(sum_str, i)
            time_taken = time.time() - t
            t = time.time()
            print(i, loss, "{} / batch".format(time_taken / 1000))
            if test_bias:
                rebs = []
                refs = []
                for _i in range(100000):
                    if _i % 1000 == 0:
                        print(_i)
                    rb, re = sess.run([rebars[3], reinforces[3]], feed_dict={x: batch_xs})
                    rebs.append(rb[:5])
                    refs.append(re[:5])
                rebs = np.array(rebs)
                refs = np.array(refs)
                re_var = np.log(refs.var(axis=0))
                rb_var = np.log(rebs.var(axis=0))
                print("rebar variance     = {}".format(rb_var))
                print("reinforce variance = {}".format(re_var))
                print("rebar     = {}".format(rebs.mean(axis=0)))
                print("reinforce = {}\n".format(refs.mean(axis=0)))
        else:
            loss, _ = sess.run([total_loss, train_op], feed_dict={x: batch_xs})

        if i % iters_per_epoch == 0:
            # epoch over, run test data
            losses = []
            for _ in range(dataset.test.num_examples // batch_size):
                batch_xs, _ = dataset.test.next_batch(batch_size)
                losses.append(sess.run(total_loss, feed_dict={x: batch_xs}))
            tl = np.mean(losses)
            print("Test loss = {}".format(tl))
            sess.run(test_loss.assign(tl))


if __name__ == "__main__":
    main(num_layers=3)