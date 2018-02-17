from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ITERS = 5000
RESOLUTION = 10

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


def logistic_loglikelihood(z, loc, scale=1):
    return tf.log(tf.exp(-(z - loc) / scale) / scale * tf.square((1 + tf.exp(-(z - loc) / scale))))


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-softplus(-log_alpha)) + (1 - b) * (-log_alpha - softplus(-log_alpha))


def bernoulli_loglikelihood_derivitive(b, log_alpha):
    assert gs(b) == gs(log_alpha)
    sna = tf.sigmoid(-log_alpha)
    return b * sna - (1 - b) * (1 - sna)


def v_from_u(u, log_alpha, force_same=True, b=None, v_prime=None):
    u_prime = tf.nn.sigmoid(-log_alpha)
    if not force_same:
        v = b * (u_prime + v_prime * (1 - u_prime)) + (1 - b) * v_prime * u_prime
    else:
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


def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 10, tf.nn.tanh, name="q_1", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)
    return out


def loss_func(b, t):
    return tf.reduce_mean(tf.square(b - t), axis=1)


def main(t=0.499, rand_seed=42, use_reinforce=False, relaxed=False,
         log_var=False, tf_log=False, force_same=False):
    with tf.Session() as sess:
        TRAIN_DIR = "./toy_problem"
        if os.path.exists(TRAIN_DIR):
            print("Deleting existing train dir")
            import shutil

            shutil.rmtree(TRAIN_DIR)
        os.makedirs(TRAIN_DIR)
        iters = ITERS  # todo: change back
        batch_size = 1
        num_latents = 1
        target = np.array([[t for i in range(num_latents)]], dtype=np.float32)
        print("Target is {}".format(target))
        lr = .01

        # encode data
        log_alpha = tf.Variable(
            [[0.0 for i in range(num_latents)]],
            trainable=True,
            name='log_alpha',
            dtype=tf.float32
        )
        a = tf.exp(log_alpha)
        theta = a / (1 + a)

        tf.set_random_seed(rand_seed)  # fix for repeatable experiments

        # reparameterization variables
        u = tf.random_uniform([batch_size, num_latents], dtype=tf.float32)
        v_p = tf.random_uniform([batch_size, num_latents], dtype=tf.float32)
        z = reparameterize(log_alpha, u)  # z(u)
        b = tf.to_float(tf.stop_gradient(z > 0))
        v = v_from_u(u, log_alpha, force_same, b, v_p)
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

        # loss function evaluations
        f_b = loss_func(b, target)

        # if we are relaxing the relaxation
        if relaxed == "THETA_U":
            z_inp = tf.concat([theta, u], 1)
            z_tilde_inp = tf.concat([theta, v], 1)
            with tf.variable_scope("Q_func"):
                f_z = Q_func(z_inp)[:, 0]
            with tf.variable_scope("Q_func", reuse=True):
                f_z_tilde = Q_func(z_tilde_inp)[:, 0]

        else:
            # relaxation variables
            batch_temp = tf.expand_dims(temperature, 0)
            sig_z = concrete_relaxation(z, batch_temp)
            sig_z_tilde = concrete_relaxation(z_tilde, batch_temp)

            if relaxed:
                with tf.variable_scope("Q_func"):
                    f_z = Q_func(sig_z)[:, 0]
                with tf.variable_scope("Q_func", reuse=True):
                    f_z_tilde = Q_func(sig_z_tilde)[:, 0]
            else:
                f_z = loss_func(sig_z, target)
                f_z_tilde = loss_func(sig_z_tilde, target)

        tf.summary.scalar("fb", tf.reduce_mean(f_b))
        tf.summary.scalar("fz", tf.reduce_mean(f_z))
        tf.summary.scalar("fzt", tf.reduce_mean(f_z_tilde))
        # loss function for generative model
        loss = tf.reduce_mean(f_b)
        tf.summary.scalar("loss", loss)

        # rebar construction
        d_f_z_d_log_alpha = tf.gradients(f_z, log_alpha)[0]
        d_f_z_tilde_d_log_alpha = tf.gradients(f_z_tilde, log_alpha)[0]
        #        d_log_pb_d_log_alpha = bernoulli_loglikelihood_derivitive(b, log_alpha)
        d_log_pb_d_log_alpha = tf.gradients(bernoulli_loglikelihood(b, log_alpha), log_alpha)[0]
        # check shapes are alright
        assert_same_shapes(d_f_z_d_log_alpha, d_f_z_tilde_d_log_alpha, d_log_pb_d_log_alpha)
        assert_same_shapes(f_b, f_z_tilde)
        batch_eta = tf.expand_dims(eta, 0)
        batch_f_b = tf.expand_dims(f_b, 1)
        batch_f_z_tilde = tf.expand_dims(f_z_tilde, 1)
        # do one of LAX, BAR, relaxed-REBAR, or REBAR
        if relaxed:
            rebar = (batch_f_b - batch_f_z_tilde) * d_log_pb_d_log_alpha + (d_f_z_d_log_alpha - d_f_z_tilde_d_log_alpha)
        else:
            rebar = (batch_f_b - batch_eta * batch_f_z_tilde) * d_log_pb_d_log_alpha + batch_eta * (
            d_f_z_d_log_alpha - d_f_z_tilde_d_log_alpha)
        reinforce = batch_f_b * d_log_pb_d_log_alpha
        tf.summary.histogram("rebar", rebar)
        tf.summary.histogram("reinforce", reinforce)

        # variance reduction objective
        variance_loss = tf.reduce_mean(tf.square(rebar))

        # optimizers
        inf_opt = tf.train.AdamOptimizer(lr)

        # need to scale by batch size cuz tf.gradients sums
        if use_reinforce:
            log_alpha_grads = reinforce / batch_size
        else:
            log_alpha_grads = rebar / batch_size

        inf_train_op = inf_opt.apply_gradients([(log_alpha_grads, log_alpha)])

        var_opt = tf.train.AdamOptimizer(lr)
        var_vars = [eta, log_temperature]
        if relaxed:
            print("Relaxed model")
            q_vars = [v for v in tf.trainable_variables() if "Q_func" in v.name]
            var_vars = var_vars + q_vars
        var_gradvars = var_opt.compute_gradients(variance_loss, var_list=var_vars)
        var_train_op = var_opt.apply_gradients(var_gradvars)

        print("Variance")
        for g, v in var_gradvars:
            print("    {}".format(v.name))
            if g is not None:
                tf.summary.histogram(v.name, v)
                tf.summary.histogram(v.name + "_grad", g)

        if use_reinforce:
            with tf.control_dependencies([inf_train_op]):
                train_op = tf.no_op()
        else:
            with tf.control_dependencies([inf_train_op, var_train_op]):
                train_op = tf.no_op()

        summ_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(TRAIN_DIR)
        sess.run(tf.global_variables_initializer())

        variances = []
        losses = []
        thetas = []

        print("Collecting {} samples".format(ITERS // RESOLUTION))
        for i in tqdm(range(iters)):
            if (i + 1) % RESOLUTION == 0:
                if tf_log:
                    loss_value, _, sum_str, theta_value = sess.run([loss, train_op, summ_op, theta])
                    summary_writer.add_summary(sum_str, i)
                else:
                    loss_value, _, theta_value, temp = sess.run([loss, train_op, theta, temperature])

                tv = theta_value[0][0]
                thetas.append(tv)
                losses.append(tv * (1 - target[0][0]) ** 2 + (1 - tv) * target[0][0] ** 2)
                print(i, loss_value, [t for t in theta_value[0]], [tmp for tmp in temp])

                if log_var:
                    grads = [sess.run([rebar, reinforce]) for i in tqdm(range(100))]
                    rebars, reinforces = zip(*grads)
                    re_m, re_v = np.mean(rebars), np.std(rebars)
                    rf_m, rf_v = np.mean(reinforces), np.std(reinforces)
                    if use_reinforce:
                        variances.append(re_v)
                    print("Reinforce mean = {}, Reinforce std = {}".format(rf_m, rf_v))
                    print("Rebar mean     = {}, Rebar std     = {}".format(re_m, re_v))


            else:
                _, = sess.run([train_op])

        tv = None
        print(tv)
        return tv, thetas, losses, variances


if __name__ == "__main__":
    t = 0.499
    rand_seed = np.random.randint(1, 1000)

    main(t=t, relaxed="THETA_U", rand_seed=rand_seed)
