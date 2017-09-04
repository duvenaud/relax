from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

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
def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 10, tf.nn.relu, name="q_1", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)
    scale = tf.get_variable(
        "q_scale", shape=[1], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True
    )
    return scale[0] * out

def loss_func(b, t):
    return tf.reduce_mean(tf.square(b - t), axis=1)


def main(use_reinforce=False, relaxed=False, visualize=False, log_var=False, tf_log=False):
    with tf.Session() as sess:
        TRAIN_DIR = "./toy_problem"
        if os.path.exists(TRAIN_DIR):
            print("Deleting existing train dir")
            import shutil

            shutil.rmtree(TRAIN_DIR)
        os.makedirs(TRAIN_DIR)
        iters = 5000
        batch_size = 1
        num_latents = 1
        #target = np.array([[float(i) / num_latents for i in range(num_latents)]], dtype=np.float32)
        target = np.array([[.499 for i in range(num_latents)]], dtype=np.float32)
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

        # loss function evaluations
        f_b = loss_func(b, target)
        f_z = loss_func(sig_z, target)
        f_z_tilde = loss_func(sig_z_tilde, target)
        if relaxed != False:
            print("Relaxed Model")
            with tf.variable_scope("Q_func"):
                q_z = Q_func(sig_z)[:, 0]
            with tf.variable_scope("Q_func", reuse=True):
                q_z_tilde = Q_func(sig_z_tilde)[:, 0]
            if relaxed == True:
                f_z = f_z + q_z
                f_z_tilde = f_z_tilde + q_z_tilde
            elif relaxed == "super":
                f_z = q_z
                f_z_tilde = q_z_tilde


        tf.summary.scalar("fb", tf.reduce_mean(f_b))
        tf.summary.scalar("fz", tf.reduce_mean(f_z))
        tf.summary.scalar("fzt", tf.reduce_mean(f_z_tilde))
        # loss function for generative model
        loss = tf.reduce_mean(f_b)
        tf.summary.scalar("loss", loss)

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
        rebar = (batch_f_b - batch_eta * batch_f_z_tilde) * d_log_p_d_log_alpha + batch_eta * (d_f_z_d_log_alpha - d_f_z_tilde_d_log_alpha)
        reinforce = batch_f_b * d_log_p_d_log_alpha
        tf.summary.histogram("rebar", rebar)
        tf.summary.histogram("reinforce", reinforce)

        # variance reduction objective
        variance_loss = tf.reduce_mean(tf.square(rebar))

        # optimizers
        inf_opt = tf.train.AdamOptimizer(lr)
        # need to scale by batch size cuz tf.gradients sums
        log_alpha_grads = (reinforce if use_reinforce else rebar) / batch_size
        inf_train_op = inf_opt.apply_gradients([(log_alpha_grads, log_alpha)])

        var_opt = tf.train.AdamOptimizer(lr)
        var_vars = [eta, log_temperature]
        if relaxed:
            q_vars = [v for v in tf.trainable_variables() if "Q_func" in v.name]
            var_vars = var_vars + q_vars
        var_gradvars = var_opt.compute_gradients(variance_loss, var_list=var_vars)
        var_train_op = var_opt.apply_gradients(var_gradvars)

        print("Variance")
        for g, v in var_gradvars:
            print("    {}".format(v.name))
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name + "_grad", g)

        if use_reinforce:
            with tf.control_dependencies([inf_train_op]):
                train_op = tf.no_op()
        else:
            with tf.control_dependencies([inf_train_op, var_train_op]):
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

        for i in range(iters):
            if i % 100 == 0:
                if tf_log:
                    loss_value, _, sum_str, theta_value = sess.run([loss, train_op, summ_op, theta])
                    summary_writer.add_summary(sum_str, i)
                else:
                    loss_value, _, theta_value = sess.run([loss, train_op, theta])

                print(i, loss_value, [t for t in theta_value[0]])


                if log_var:
                    grads = [sess.run([rebar, reinforce]) for i in range(1000)]
                    rebars, reinforces = zip(*grads)
                    re_m, re_v = np.mean(rebars), np.std(rebars)
                    rf_m, rf_v = np.mean(reinforces), np.std(reinforces)
                    print("Reinforce mean = {}, Reinforce std = {}".format(rf_m, rf_v))
                    print("Rebar mean     = {}, Rebar std     = {}".format(re_m, re_v))
            else:
                _, = sess.run([train_op])

        if visualize:
            X = [float(i) / 100 for i in range(100)]
            FZ = []
            for x in X:
                fz = sess.run(f_z, feed_dict={sig_z: [[x]]})
                FZ.append(fz)
            plt.plot(X, FZ)
            plt.show()
        tv = theta_value[0][0]
        print(tv)
        return theta_value[0][0]

            # if i % 100 == 0:
            #     # bias test
            #     rebars = []
            #     reinforces = []
            #     for _ in range(10000):
            #         rb, re = sess.run([rebar, reinforce])
            #         rebars.append(rb)
            #         reinforces.append(re)
            #     rebars = np.array(rebars)
            #     reinforces = np.array(reinforces)
            #     re_var = reinforces.var(axis=0)
            #     rb_var = rebars.var(axis=0)
            #     diffs = np.abs(rebars.mean(axis=0) - reinforces.mean(axis=0))
            #     sess.run([rebar_var.assign(rb_var), reinforce_var.assign(re_var), est_diffs.assign(diffs)])
            #     print("rebar variance", rb_var.mean())
            #     print("reinforce variance", re_var.mean())
            #     print(rebars.mean(axis=0)[0])
            #     print(reinforces.mean(axis=0)[0])
            #     print()




if __name__ == "__main__":
    thetas = []
    for i in range(10):
        tf.reset_default_graph()
        thetas.append(main(relaxed="super"))
    print(np.mean(thetas), np.std(thetas))