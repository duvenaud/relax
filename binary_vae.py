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


def neg_elbo(x, b, log_alpha, pred_x_log_alpha):
    log_q_b_given_x = tf.reduce_sum(bernoulli_loglikelihood(b, log_alpha), axis=1)
    log_p_b = tf.reduce_sum(bernoulli_loglikelihood(b, tf.zeros_like(log_alpha)), axis=1)
    log_p_x_given_b = tf.reduce_sum(bernoulli_loglikelihood(x, pred_x_log_alpha), axis=1)
    return -1. * (log_p_x_given_b + log_p_b - log_q_b_given_x)


""" Networks """
def encoder(x, num_latents):
    h1 = tf.layers.dense(2. * x - 1., 200, tf.nn.relu, name="encoder_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="encoder_2")
    log_alpha = tf.layers.dense(h2, num_latents, name="encoder_out")
    return log_alpha

def decoder(b):
    h1 = tf.layers.dense(2. * b - 1., 200, tf.nn.relu, name="decoder_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="decoder_2")
    log_alpha = tf.layers.dense(h2, 784, name="decoder_out")
    return log_alpha

def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 200, tf.nn.relu, name="q_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="q_2")
    out = tf.layers.dense(h2, 1, name="q_out")
    scale = tf.get_variable(
        "q_scale", shape=[1], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True
    )
    return scale[0] * out


def main():
    TRAIN_DIR = "./binary_vae_test"
    use_reinforce = False
    relaxed = False
    if os.path.exists(TRAIN_DIR):
        print("Deleting existing train dir")
        import shutil

        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)
    sess = tf.Session()
    num_epochs = 300
    batch_size = 100
    num_latents = 200
    lr = .0001
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [batch_size, 784])
    x_im = tf.reshape(x, [batch_size, 28, 28, 1])
    tf.summary.image("x_true", x_im)
    x_binary = tf.to_float(x > .5)

    # encode data
    with tf.variable_scope("encoder"):
        log_alpha = encoder(x_binary, num_latents)

    # reparameterization variables
    u = tf.random_uniform([batch_size, num_latents], dtype=tf.float32)
    z = reparameterize(log_alpha, u)
    b = tf.to_float(tf.stop_gradient(z > 0))
    v = v_from_u(u, log_alpha)
    z_tilde = reparameterize(log_alpha, v)

    # rebar variables
    eta = tf.Variable(
        [1.0 for i in range(num_latents)],
        trainable=True,
        name='eta',
        dtype=tf.float32
    )
    log_temperature = tf.Variable(
        [np.log(.5) for i in range(num_latents)],
        trainable=True,
        name='log_temperature',
        dtype=tf.float32
    )
    temperature = tf.exp(log_temperature)

    # relaxation variables
    batch_temp = tf.expand_dims(temperature, 0)
    sig_z = concrete_relaxation(z, batch_temp, log_alpha)
    sig_z_tilde = concrete_relaxation(z_tilde, batch_temp, log_alpha)

    # decoder evaluations
    with tf.variable_scope("decoder"):
        dec_b = decoder(b)
        a = tf.exp(dec_b)
        dec_log_theta = a / (1 + a)
        dec_log_theta_im = tf.reshape(dec_log_theta, [batch_size, 28, 28, 1])
        tf.summary.image("x_pred", dec_log_theta_im)
    with tf.variable_scope("decoder", reuse=True):
        dec_z = decoder(sig_z)
        dec_z_tilde = decoder(sig_z_tilde)

    # loss function evaluations
    f_b = neg_elbo(x_binary, b, log_alpha, dec_b)
    f_z = neg_elbo(x_binary, sig_z, log_alpha, dec_z)
    f_z_tilde = neg_elbo(x_binary, sig_z_tilde, log_alpha, dec_z_tilde)
    if relaxed:
        print("Relaxed Model")
        with tf.variable_scope("Q_func"):
            q_z = Q_func(sig_z)[:, 0]
        with tf.variable_scope("Q_func", reuse=True):
            q_z_tilde = Q_func(sig_z_tilde)[:, 0]
        f_z = f_z + q_z
        f_z_tilde = f_z_tilde + q_z_tilde


    tf.summary.scalar("fb", tf.reduce_mean(f_b))
    tf.summary.scalar("fz", tf.reduce_mean(f_z))
    tf.summary.scalar("fzt", tf.reduce_mean(f_z_tilde))
    # loss function for generative model
    generative_loss = tf.reduce_mean(f_b)
    tf.summary.scalar("loss", generative_loss)

    # rebar construction
    d_f_z_d_log_alpha = tf.gradients(f_z, log_alpha)[0]
    d_f_z_tilde_d_log_alpha = tf.gradients(f_z_tilde, log_alpha)[0]
    d_log_p_d_log_alpha = bernoulli_loglikelihood_derivitive(b, log_alpha)
    # check shapes are alright
    assert_same_shapes(d_f_z_d_log_alpha, d_f_z_tilde_d_log_alpha, d_log_p_d_log_alpha)
    assert_same_shapes(f_b, f_z_tilde)
    batch_eta = tf.expand_dims(eta, 0)
    batch_f_b = tf.expand_dims(f_b, 1)
    batch_f_z_tilde = tf.expand_dims(f_z_tilde, 1)
    d_f_b_d_log_alpha = tf.gradients(f_b, log_alpha)[0]
    tf.summary.histogram("df/dtheta", d_f_b_d_log_alpha)
    rebar = (batch_f_b - batch_eta * batch_f_z_tilde) * d_log_p_d_log_alpha + batch_eta * (d_f_z_d_log_alpha - d_f_z_tilde_d_log_alpha) + d_f_b_d_log_alpha
    reinforce = batch_f_b * d_log_p_d_log_alpha + d_f_b_d_log_alpha
    tf.summary.histogram("rebar", rebar)
    tf.summary.histogram("reinforce", reinforce)

    # variance reduction objective
    variance_loss = tf.reduce_mean(tf.square(rebar))

    # optimizers
    inf_opt = tf.train.AdamOptimizer(lr)
    inf_vars = [v for v in tf.trainable_variables() if "encoder" in v.name]
    # need to scale by batch size cuz tf.gradients sums
    log_alpha_grads = (reinforce if use_reinforce else rebar) / batch_size
    inf_grads = tf.gradients(log_alpha, inf_vars, grad_ys=log_alpha_grads)
    inf_gradvars = zip(inf_grads, inf_vars)
    inf_train_op = inf_opt.apply_gradients(inf_gradvars)

    gen_opt = tf.train.AdamOptimizer(lr)
    gen_vars = [v for v in tf.trainable_variables() if "decoder" in v.name]
    gen_gradvars = gen_opt.compute_gradients(generative_loss, var_list=gen_vars)
    gen_train_op = gen_opt.apply_gradients(gen_gradvars)

    var_opt = tf.train.AdamOptimizer(lr)
    var_vars = [eta, log_temperature]
    if relaxed:
        q_vars = [v for v in tf.trainable_variables() if "Q_func" in v.name]
        var_vars = var_vars + q_vars
    var_gradvars = var_opt.compute_gradients(variance_loss, var_list=var_vars)
    var_train_op = var_opt.apply_gradients(var_gradvars)

    print("Inference")
    for g, v in inf_gradvars:
        print("    {}".format(v.name))
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)
    print("Generative")
    for g, v in gen_gradvars:
        print("    {}".format(v.name))
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)
    print("Variance")
    for g, v in var_gradvars:
        print("    {}".format(v.name))
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "_grad", g)

    if use_reinforce:
        with tf.control_dependencies([gen_train_op, inf_train_op]):
            train_op = tf.no_op()
    else:
        with tf.control_dependencies([gen_train_op, inf_train_op, var_train_op]):
            train_op = tf.no_op()

    test_loss = tf.Variable(600, trainable=False, name="test_loss", dtype=tf.float32)
    rebar_var = tf.Variable(np.zeros([batch_size, num_latents]), trainable=False, name="rebar_variance", dtype=tf.float32)
    reinforce_var = tf.Variable(np.zeros([batch_size, num_latents]), trainable=False, name="reinforce_variance", dtype=tf.float32)
    est_diffs = tf.Variable(np.zeros([batch_size, num_latents]), trainable=False, name="estimator_differences", dtype=tf.float32)
    tf.summary.scalar("test_loss", test_loss)
    tf.summary.histogram("rebar_variance", rebar_var)
    tf.summary.histogram("reinforace_variance", reinforce_var)
    tf.summary.histogram("estimator_diffs", est_diffs)
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
        #     # bias test
        #     rebars = []
        #     reinforces = []
        #     for _ in range(100000):
        #         rb, re = sess.run([rebar, reinforce], feed_dict={x: batch_xs})
        #         rebars.append(rb)
        #         reinforces.append(re)
        #     rebars = np.array(rebars)
        #     reinforces = np.array(reinforces)
        #     re_var = reinforces.var(axis=0)
        #     rb_var = rebars.var(axis=0)
        #     diffs = np.abs(rebars.mean(axis=0) - reinforces.mean(axis=0))
        #     percent_diffs = diffs / rebars.mean(axis=0)
        #     print("rebar variance", rb_var.mean())
        #     print("reinforce variance", re_var.mean())
        #     print("diffs", diffs.mean())
        #     print("percent diffs", percent_diffs.mean())
        #     print(rebars.mean(axis=0)[0])
        #     print(reinforces.mean(axis=0)[0])
        #     sess.run(
        #         [rebar_var.assign(rb_var), reinforce_var.assign(re_var), est_diffs.assign(diffs)]
        #     )



if __name__ == "__main__":
    main()