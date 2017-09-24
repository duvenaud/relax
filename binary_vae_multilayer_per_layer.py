from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
import os
import datasets


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


def reparameterize(log_alpha, noise, name=None):
    return tf.identity(log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise), name=name)


def concrete_relaxation(log_alpha, noise, temp, name):
    z = log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)
    return tf.sigmoid(z / temp, name=name)


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


def Q_func(x, x_mean, z, bs, name, reuse):
    inp = tf.concat([x - x_mean, z] + [2. * b - 1 for b in bs], 1)
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(inp, 200, tf.tanh, name="1")
        h2 = tf.layers.dense(h1, 200, tf.tanh, name="2")
        h3 = tf.layers.dense(h2, 200, tf.tanh, name="3")
        h4 = tf.layers.dense(h3, 200, tf.tanh, name="4")
        out = tf.layers.dense(h4, 1, name="out")[:, 0]
        # scale = tf.get_variable(
        #     "q_scale", shape=[1], dtype=tf.float32,
        #     initializer=tf.constant_initializer(0), trainable=True
        # )
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
    def __init__(self, u, name):
        self.u = u
        self.name = name
    def sample(self, log_alpha, l):
        z = reparameterize(log_alpha, self.u[l])
        b = tf.to_float(tf.stop_gradient(z > 0), name="{}_{}".format(self.name, l))
        return b


class ZSampler:
    def __init__(self, u, name):
        self.u = u
        self.name = name
    def sample(self, log_alpha, l):
        z = reparameterize(log_alpha, self.u[l], name="{}_{}".format(self.name, l))
        return z


class SIGZSampler:
    def __init__(self, u, temp, name):
        self.u = u
        self.temp = temp
        self.name = name
    def sample(self, log_alpha, l):
        sig_z = concrete_relaxation(log_alpha, self.u[l], self.temp[l], name="{}_{}".format(self.name, l))
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


def main(relaxation=None, learn_prior=True, max_iters=2000000,
         batch_size=24, num_latents=200, num_layers=2, lr=.0001,
         test_bias=False, train_dir=None, iwae_samples=100, dataset="mnist"):

    if os.path.exists(train_dir):
        print("Deleting existing train dir")
        import shutil

        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    sess = tf.Session()
    if dataset == "mnist":
        X_tr, X_va, X_te = datasets.load_mnist()
    elif dataset == "omni":
        X_tr, X_va, X_te = datasets.load_omniglot()
    else:
        assert False
    train_mean = np.mean(X_tr, axis=0, keepdims=True)
    train_output_bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.).astype(np.float32)

    x = tf.placeholder(tf.float32, [batch_size, 784])
    x_im = tf.reshape(x, [batch_size, 28, 28, 1])
    tf.summary.image("x_true", x_im)

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
    b_sampler = BSampler(u, "b_sampler")
    gen_b_sampler = BSampler(u, "gen_b_sampler")
    # generate hard forward pass
    encoder_name = "encoder"
    decoder_name = "decoder"
    inf_la_b, samples_b = inference_network(
        x, train_mean,
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
        num_latents, decoder_name, True, sampler=gen_b_sampler, prior=p_prior
    )
    log_image(_samples_la_b[-1], "x_sample")

    # hard loss evaluation and log probs
    f_b, log_q_bs = neg_elbo(x, samples_b, inf_la_b, gen_la_b, p_prior, log=True)
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
    sig_z_sampler = SIGZSampler(u, batch_temperatures, "sig_z_sampler")
    sig_zt_sampler = SIGZSampler(v, batch_temperatures, "sig_zt_sampler")

    z_sampler = ZSampler(u, "z_sampler")
    zt_sampler = ZSampler(v, "zt_sampler")

    rebars = []
    reinforces = []
    variance_objectives = []
    # have to produce 2 forward passes for each layer for z and zt samples
    for l in range(num_layers):
        cur_la_b = inf_la_b[l]

        # if standard rebar or additive relaxation
        if relaxation is None or relaxation == "add":
            # compute soft samples and soft passes through model and soft elbos
            cur_z_sample = sig_z_sampler.sample(cur_la_b, l)
            prev_samples_z = samples_b[:l] + [cur_z_sample]

            cur_zt_sample = sig_zt_sampler.sample(cur_la_b, l)
            prev_samples_zt = samples_b[:l] + [cur_zt_sample]

            prev_log_alphas = inf_la_b[:l] + [cur_la_b]

            # soft forward passes
            inf_la_z, samples_z = inference_network(
                x, train_mean,
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
                x, train_mean,
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
            f_z, _ = neg_elbo(x, samples_z, inf_la_z, gen_la_z, p_prior)
            f_zt, _ = neg_elbo(x, samples_zt, inf_la_zt, gen_la_zt, p_prior)

        if relaxation is not None:
            # sample z and zt
            prev_bs = samples_b[:l]
            cur_z_sample = z_sampler.sample(cur_la_b, l)
            cur_zt_sample = zt_sampler.sample(cur_la_b, l)

            q_z = Q_func(x, train_mean, cur_z_sample, prev_bs, Q_name(l), False)
            q_zt = Q_func(x, train_mean, cur_zt_sample, prev_bs, Q_name(l), True)
            tf.summary.scalar("q_z_{}".format(l), tf.reduce_mean(q_z))
            tf.summary.scalar("q_zt_{}".format(l), tf.reduce_mean(q_zt))
            if relaxation == "add":
                f_z = f_z + q_z
                f_zt = f_zt + q_zt
            elif relaxation == "all":
                f_z = q_z
                f_zt = q_zt
            else:
                assert False
        tf.summary.scalar("f_z_{}".format(l), tf.reduce_mean(f_z))
        tf.summary.scalar("f_zt_{}".format(l), tf.reduce_mean(f_zt))
        cur_samples_b = samples_b[l]
        # get gradient of sample log-likelihood wrt current parameter
        d_log_q_d_la = bernoulli_loglikelihood_derivitive(cur_samples_b, cur_la_b)
        # get gradient of soft-losses wrt current parameter
        d_f_z_d_la = tf.gradients(f_z, cur_la_b)[0]
        d_f_zt_d_la = tf.gradients(f_zt, cur_la_b)[0]
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
    if relaxation is not None:
        q_vars = get_variables("Q_")
        variance_vars = variance_vars + q_vars
    variance_gradvars = variance_opt.compute_gradients(variance_objective, var_list=variance_vars)
    variance_train_op = variance_opt.apply_gradients(variance_gradvars)
    model_train_op = model_opt.apply_gradients(model_gradvars)
    with tf.control_dependencies([model_train_op, variance_train_op]):
        train_op = tf.no_op()

    for g, v in model_gradvars + variance_gradvars:
        print(g, v.name)
        if g is not None:
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name+"_grad", g)

    test_loss = tf.Variable(1000, trainable=False, name="test_loss", dtype=tf.float32)
    train_loss = tf.Variable(1000, trainable=False, name="train_loss", dtype=tf.float32)
    tf.summary.scalar("test_loss", test_loss)
    tf.summary.scalar("train_loss", train_loss)
    summ_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(train_dir)
    sess.run(tf.global_variables_initializer())

    iters_per_epoch = X_tr.shape[0] // batch_size
    print("Train set has {} examples".format(X_tr.shape[0]))
    t = time.time()
    for epoch in range(10000000):
        train_losses = []
        for i in range(iters_per_epoch):
            cur_iter = epoch * iters_per_epoch + i
            if cur_iter > max_iters:
                print("Training Completed")
                return
            batch_xs = X_tr[i*batch_size: (i+1) * batch_size]
            if i % 100 == 0:
                loss, _, sum_str = sess.run([total_loss, train_op, summ_op], feed_dict={x: batch_xs})
                summary_writer.add_summary(sum_str, cur_iter)
                time_taken = time.time() - t
                t = time.time()
                print(cur_iter, loss, "{} / batch".format(time_taken / 100))
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

            train_losses.append(loss)

        # epoch over, run test data
        test_losses = []
        for _it in range(X_te.shape[0] // batch_size - 1):
            batch_xs = X_te[_it*batch_size: (_it+1)*batch_size]
            test_losses.append(sess.run(total_loss, feed_dict={x: batch_xs}))
        trl = np.mean(train_losses)
        tel = np.mean(test_losses)
        print("Test loss = {}, Train loss = {}".format(tel, trl))
        sess.run([test_loss.assign(tel), train_loss.assign(trl)])

        np.random.shuffle(X_tr)


if __name__ == "__main__":
    main(num_layers=3, relaxation="add", train_dir="/ais/gobi5/wgrathwohl/rebar_experiments/binary_var_3_layer_add_big_Q", dataset="mnist", lr=.0005)
