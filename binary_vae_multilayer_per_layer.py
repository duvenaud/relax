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


def concrete_relaxation(log_alpha, noise, temp):
    z = log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)
    return tf.sigmoid(z / temp)


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
    return -1. * (log_p_b_x - log_q_b), log_q_bs


""" Networks """
def encoder(x, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alpha = tf.layers.dense(2. * x - 1., num_latents, name="log_alpha")
    return log_alpha


def decoder(b, num_latents, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        log_alpha = tf.layers.dense(2. * b - 1., num_latents, name="log_alpha")
    return log_alpha


def inference_network(x, mean, layer, num_layers, num_latents, name, reuse, sampler, samples=[], log_alphas=[]):
    with tf.variable_scope(name, reuse=reuse):
        assert len(samples) == len(log_alphas)
        # copy arrays to avoid them being modified
        samples = [s for s in samples]
        log_alphas = [la for la in log_alphas]
        start = len(samples)
        for l in range(start, num_layers):
            if l == 0:
                inp = ((x - mean) + 1.) / 2.
            else:
                inp = samples[-1]
            log_alpha = layer(inp, num_latents, layer_name(l), reuse)
            log_alphas.append(log_alpha)
            sample = sampler.sample(log_alpha, l)
            samples.append(sample)
    assert len(log_alphas) == len(samples) == num_layers
    return log_alphas, samples


def layer_name(l):
    return "layer_{}".format(l)


def Q_name(l):
    return "Q_{}".format(l)

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
                784 if l == 0 else num_latents, layer_name(l), reuse
            )
            if l == 0:
                log_alpha = log_alpha + output_bias
            log_alphas.append(log_alpha)
            if l > 0 and PRODUCE_SAMPLES:
                samples[l-1] = sampler.sample(log_alpha, l-1)
    return log_alphas


def Q_func(x, bs, name, reuse):
    inp = tf.concat([x] + bs, 1)
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(inp, 200, tf.tanh, name="1")
        h2 = tf.layers.dense(h1, 200, tf.tanh, name="2")
        out = tf.layers.dense(h2, 1, name="out")[:, 0]
    return out


""" Variable Creation """
def create_log_temp(num):
    return tf.Variable(
        [np.log(.5) for i in range(num)],
        trainable=True,
        name='log_temperature',
        dtype=tf.float32
    )


def create_eta(num):
    return tf.Variable(
        [1.0 for i in range(num)],
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
    def __init__(self, u):
        self.u = u
    def sample(self, log_alpha, l):
        z = reparameterize(log_alpha, self.u[l])
        return z


class SIGZSampler:
    def __init__(self, u, temp):
        self.u = u
        self.temp = temp
    def sample(self, log_alpha, l):
        sig_z = concrete_relaxation(log_alpha, self.u[l], self.temp[l])
        return sig_z


def log_image(im_vec, name):
    # produce reconstruction summary
    a = tf.exp(im_vec)
    dec_log_theta = a / (1 + a)
    dec_log_theta_im = tf.reshape(dec_log_theta, [-1, 28, 28, 1])
    tf.summary.image(name, dec_log_theta_im)


def get_variables(tag, arr=None):
    if arr is None:
        return [v for v in tf.global_variables() if tag in v.name]
    else:
        return [v for v in arr if tag in v.name]


def main(relaxation=None, learn_prior=True, num_epochs=840,
         batch_size=24, num_latents=200, num_layers=2, lr=.0001, test_bias=False):
    TRAIN_DIR = "./binary_vae_test_per_layer_relaxed"
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
    log_temperatures = [create_log_temp(1) for l in range(num_layers)]
    temperatures = [tf.exp(log_temp) for log_temp in log_temperatures]
    batch_temperatures = [tf.reshape(temp, [1, -1]) for temp in temperatures]
    etas = [create_eta(1) for l in range(num_layers)]
    batch_etas = [tf.reshape(eta, [1, -1]) for eta in etas]

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
    log_image(gen_la_b[-1], "x_pred")
    # produce samples
    _samples_la_b = generator_network(
        None, train_output_bias,
        decoder, num_layers,
        num_latents, decoder_name, True, sampler=b_sampler, prior=p_prior
    )
    log_image(_samples_la_b[-1], "x_sample")

    # hard loss evaluation and log probs
    f_b, log_q_bs = neg_elbo(x_binary, samples_b, inf_la_b, gen_la_b, p_prior, log=True)
    batch_f_b = tf.expand_dims(f_b, 1)
    total_loss = tf.reduce_mean(f_b)
    tf.summary.scalar("fb", total_loss)
    # optimizer for model parameters
    model_opt = tf.train.AdamOptimizer(lr, beta2=.99999)
    # optimizer for variance reducing parameters
    variance_opt = tf.train.AdamOptimizer(10. * lr, beta2=.99999)
    # get encoder and decoder variables
    encoder_params = get_variables(encoder_name)
    decoder_params = get_variables(decoder_name)
    if learn_prior:
        decoder_params.append(p_prior)
    # compute and store gradients of hard loss with respect to encoder_parameters
    encoder_loss_grads = {}
    for g, v in model_opt.compute_gradients(total_loss, var_list=encoder_params):
        encoder_loss_grads[v.name] = g
    # get gradients for decoder parameters
    decoder_gradvars = model_opt.compute_gradients(total_loss, var_list=decoder_params)
    # will hold all gradvars for the model (non-variance adjusting variables)
    model_gradvars = [gv for gv in decoder_gradvars]

    # conditional samples
    v = [v_from_u(_u, log_alpha) for _u, log_alpha in zip(u, inf_la_b)]
    # need to create soft samplers
    sig_z_sampler = SIGZSampler(u, batch_temperatures)
    sig_zt_sampler = SIGZSampler(v, batch_temperatures)

    rebars = []
    reinforces = []
    variance_objectives = []
    # have to produce 2 forward passes for each layer for z and zt samples
    for l in range(num_layers):
        cur_la_b = inf_la_b[l]

        cur_z_sample = sig_z_sampler.sample(cur_la_b, l)
        prev_samples_z = samples_b[:l] + [cur_z_sample]

        cur_zt_sample = sig_zt_sampler.sample(cur_la_b, l)
        prev_samples_zt = samples_b[:l] + [cur_zt_sample]

        prev_log_alphas = inf_la_b[:l] + [cur_la_b]

        # soft forward passes
        inf_la_z, samples_z = inference_network(
            x_binary, train_mean,
            encoder, num_layers,
            num_latents, encoder_name, True, sig_z_sampler,
            samples=prev_samples_z, log_alphas=prev_log_alphas
        )
        gen_la_z = generator_network(
            samples_z, train_output_bias,
            decoder, num_layers,
            num_latents, decoder_name, True
        )
        inf_la_zt, samples_zt = inference_network(
            x_binary, train_mean,
            encoder, num_layers,
            num_latents, encoder_name, True, sig_zt_sampler,
            samples=prev_samples_zt, log_alphas=prev_log_alphas
        )
        gen_la_zt = generator_network(
            samples_zt, train_output_bias,
            decoder, num_layers,
            num_latents, decoder_name, True
        )
        # soft loss evaluataions
        f_z, _ = neg_elbo(x_binary, samples_z, inf_la_z, gen_la_z, p_prior)
        f_zt, _ = neg_elbo(x_binary, samples_zt, inf_la_zt, gen_la_zt, p_prior)
        if relaxation:
            q_z = Q_func(x_binary, prev_samples_z, Q_name(l), False)
            q_zt = Q_func(x_binary, prev_samples_zt, Q_name(l), True)
            tf.summary.scalar("q_z_{}".format(l), tf.reduce_mean(q_z))
            tf.summary.scalar("q_zt_{}".format(l), tf.reduce_mean(q_zt))
            f_z = f_z + q_z
            f_zt = f_zt + q_zt
        tf.summary.scalar("f_z_{}".format(l), tf.reduce_mean(f_z))
        tf.summary.scalar("f_zt_{}".format(l), tf.reduce_mean(f_zt))
        cur_samples_b = samples_b[l]
        # get gradient of sample log-likelihood wrt current parameter
        d_log_q_d_la = bernoulli_loglikelihood_derivitive(cur_samples_b, cur_la_b)
        # get gradient of soft-losses wrt current parameter
        d_f_z_d_la = tf.gradients(f_z, cur_la_b)[0]
        d_f_zt_d_la = tf.gradients(f_zt, cur_la_b)[0]
        """
        The problem is that im not using the same log alpha for fz and fzt
        """
        tf.summary.histogram("d_f_z_d_la_{}".format(l), d_f_z_d_la)
        tf.summary.histogram("d_f_zt_d_la_{}".format(l), d_f_zt_d_la)
        batch_f_zt = tf.expand_dims(f_zt, 1)
        eta = batch_etas[l]
        # compute rebar and reinforce
        tf.summary.histogram("der_diff_{}".format(l), d_f_z_d_la - d_f_zt_d_la)
        tf.summary.histogram("d_log_q_d_la_{}".format(l), d_log_q_d_la)
        rebar = ((batch_f_b - eta * batch_f_zt) * d_log_q_d_la + eta * (d_f_z_d_la - d_f_zt_d_la)) / batch_size
        reinforce = batch_f_b * d_log_q_d_la / batch_size
        rebars.append(rebar)
        reinforces.append(reinforce)
        tf.summary.histogram("rebar_{}".format(l), rebar)
        tf.summary.histogram("reinforce_{}".format(l), reinforce)
        # backpropogate rebar to individual layer parameters
        layer_params = get_variables(layer_name(l), arr=encoder_params)
        layer_rebar_grads = tf.gradients(cur_la_b, layer_params, grad_ys=rebar)
        # get direct loss grads for each parameter
        layer_loss_grads = [encoder_loss_grads[v.name] for v in layer_params]
        # each param's gradient should be rebar + the direct loss gradient
        layer_grads = [rg + lg for rg, lg in zip(layer_rebar_grads, layer_loss_grads)]
        for rg, lg, v in zip(layer_rebar_grads, layer_loss_grads, layer_params):
            tf.summary.histogram(v.name+"_grad_rebar", rg)
            tf.summary.histogram(v.name+"_grad_loss", lg)
        layer_gradvars = list(zip(layer_grads, layer_params))
        model_gradvars.extend(layer_gradvars)
        variance_objective = tf.reduce_mean(tf.reduce_sum(tf.square(rebar), axis=1))
        variance_objectives.append(variance_objective)

    variance_objective = tf.add_n(variance_objectives)
    variance_vars = log_temperatures + etas
    if relaxation:
        q_vars = get_variables("Q_")
        variance_vars = variance_vars + q_vars
    variance_gradvars = variance_opt.compute_gradients(variance_objective, var_list=variance_vars)
    variance_train_op = variance_opt.apply_gradients(variance_gradvars)
    model_train_op = model_opt.apply_gradients(model_gradvars)
    with tf.control_dependencies([model_train_op, variance_train_op]):
        train_op = tf.no_op()

    for g, v in model_gradvars + variance_gradvars:
        print(g, v.name)
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name+"_grad", g)

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
        if i % 100 == 0:
            loss, _, sum_str = sess.run([total_loss, train_op, summ_op], feed_dict={x: batch_xs})
            summary_writer.add_summary(sum_str, i)
            time_taken = time.time() - t
            t = time.time()
            print(i, loss, "{} / batch".format(time_taken / 100))
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
    main(num_layers=2, relaxation=True)