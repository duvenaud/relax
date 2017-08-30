import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
ion()
import matplotlib

matplotlib.rc("savefig") #, dpi=300)


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


class REBAROptimizer(object):
    def __init__(self, sess, loss, log_alpha=None, dim=None, name="REBAR", learning_rate=.01, n_samples=1):
        self.name = name
        self.sess = sess
        self.loss = loss
        self.dim = dim
        self.log_alpha = log_alpha
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.variance_optimizer = tf.train.AdamOptimizer(learning_rate)

        """ model parameters """
        self._create_model_parameters()
        """ reparameterization noise """
        self._create_reparam_variables()
        """ relaxed loss evaluations """
        self._create_loss_evaluations()
        """ gradvars for optimizers """
        self._create_gradvars()
        """ variance reduction optimization operation """
        self.variance_reduction_op = self.variance_optimizer.apply_gradients(self.variance_gradvars)

    def _create_model_parameters(self):
        # alpha = theta / (1 - theta)
        if self.log_alpha is None:
            "no log alpha given, creating here"
            self.log_alpha = tf.Variable(
                [0.0 for i in range(self.dim)],  # initial value
                name='log_alpha', dtype=tf.float32
            )
            self.batch_log_alpha = self.log_alpha
            self.batch_size = 1
        else:
            sh = gs(self.log_alpha)
            if len(sh) > 1:
                self.batch_log_alpha = self.log_alpha
                self.log_alpha = tf.reshape(self.batch_log_alpha, [-1])
            self.dim = gs(self.log_alpha)[0]
            self.batch_size = sh[0]
        a = tf.exp(self.log_alpha)
        theta = a / (1 + a)
        tf.summary.histogram("theta", theta)
        # expanded version for internal purposes
        self._log_alpha = tf.expand_dims(self.log_alpha, 0)
        n_vars = self.dim / self.batch_size
        self.n_vars = n_vars
        self.batch_log_temperature = tf.Variable(
            [np.log(.5) for i in range(n_vars)],
            trainable=False,
            name='log_temperature',
            dtype=tf.float32
        )
        self.log_temperature = tf.reshape(tf.tile(tf.expand_dims(self.batch_log_temperature, 0), [self.batch_size, 1]), [-1])
        self.tiled_log_temperature = tf.tile([self.log_temperature], [self.n_samples, 1])
        self.temperature = tf.exp(self.tiled_log_temperature)
        tf.summary.histogram("temp", self.temperature)
        self.batch_eta = tf.Variable(
            [1.0 for i in range(n_vars)],
            trainable=False,
            name='eta',
            dtype=tf.float32
        )
        self.eta = tf.reshape(tf.tile(tf.expand_dims(self.batch_eta, 0), [self.batch_size, 1]), [-1])

    def _create_reparam_variables(self, eps=1e-8):
        # noise for generating z
        u = tf.random_uniform([self.n_samples, self.dim], dtype=tf.float32)
        log_alpha = self._log_alpha
        # logistic reparameterization z = g(u, log_alpha)
        z = log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)
        # b = H(z)
        b = tf.to_float(tf.stop_gradient(z > 0))
        # g(u', log_alpha) = 0
        u_prime = tf.nn.sigmoid(-log_alpha)
        v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1)
        v_1 = tf.clip_by_value(v_1, 0, 1)
        v_1 = tf.stop_gradient(v_1)
        v_1 = v_1 * (1 - u_prime) + u_prime
        v_0 = u / tf.clip_by_value(u_prime, eps, 1)
        v_0 = tf.clip_by_value(v_0, 0, 1)
        v_0 = tf.stop_gradient(v_0)
        v_0 = v_0 * u_prime

        v = tf.where(u > u_prime, v_1, v_0)
        v = tf.check_numerics(v, 'v sampling is not numerically stable.')
        v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors
        tf.summary.histogram("u-v", u-v)

        z_tilde = log_alpha + safe_log_prob(v) - safe_log_prob(1 - v)
        self.b = b
        self.z = z
        self.z_tilde = z_tilde

    def _create_loss_evaluations(self):
        """
        produces f(b), f(sig(z)), f(sig(z_tilde))
        """
        # relaxed inputs
        log_alpha = self._log_alpha
        sig_z = tf.nn.sigmoid(self.z / self.temperature + log_alpha)
        sig_z_tilde = tf.nn.sigmoid(self.z_tilde / self.temperature + log_alpha)
        # evaluate loss
        f_b = tf.reshape(self.loss(tf.reshape(self.b, [self.batch_size, -1])), [-1])
        f_z = tf.reshape(self.loss(tf.reshape(sig_z, [self.batch_size, -1])), [-1])
        f_z_tilde = tf.reshape(self.loss(tf.reshape(sig_z_tilde, [self.batch_size, -1])), [-1])
        self.f_b = f_b
        self.f_z = f_z
        self.f_z_tilde = f_z_tilde

    def _create_gradvars(self):
        """
        produces d[log p(b)]/d[log_alpha], d[f(sigma_theta(z))]/d[log_alpha], d[f(sigma_theta(z_tilde))]/d[log_alpha]
        """
        log_alpha = self._log_alpha
        eta = tf.expand_dims(self.eta, 0)
        f_b = tf.expand_dims(self.f_b, 1)
        f_z_tilde = tf.expand_dims(self.f_z_tilde, 1)
        d_log_p_d_log_alpha = bernoulli_loglikelihood_derivitive(self.b, log_alpha)
        term1 = ((f_b - eta * f_z_tilde) * d_log_p_d_log_alpha)[0]
        # d[f(sigma_theta(z))]/d[log_alpha] - eta * d[f(sigma_theta(z_tilde))]/d[log_alpha]
        term2 = tf.gradients(
            tf.reduce_mean(self.f_z - self.f_z_tilde),
            self.log_alpha
        )[0]
        # rebar gradient estimator
        rebar = term1 + self.eta * term2
        reinforce = (f_b * d_log_p_d_log_alpha)[0]
        # now compute gradients of the variance of this wrt other parameters
        # eta
        d_var_d_eta = tf.gradients(
            tf.reduce_sum(tf.square(rebar)) / self.batch_size,
            self.batch_eta
        )[0]
        # temperature
        d_var_d_temperature = tf.gradients(
            tf.reduce_sum(tf.square(rebar)) / self.batch_size,
            self.batch_log_temperature
        )[0]
        self._rebar = rebar
        self.rebar = tf.reshape(rebar, [self.batch_size, -1])
        self.reinforce = tf.reshape(reinforce, [self.batch_size, -1])
        tf.summary.histogram("rebar_gradient", rebar)
        tf.summary.histogram("reinforce_gradient", reinforce)
        self.rebar_gradvars = [(rebar, self.log_alpha)]
        self.variance_gradvars = [(d_var_d_eta, self.batch_eta), (d_var_d_temperature, self.batch_log_temperature)]

    def train(self, n_steps=10000):
        self.sess.run(tf.global_variables_initializer())
        ave_loss = tf.reduce_mean(self.f_b)
        summ_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("/tmp/rebar")
        for iter in xrange(n_steps):
            if iter % 100 == 0:
                _, sum_str, loss_val, g_val, g_val_r, la, t, e = sess.run(
                    [self.train_op, summ_op, ave_loss, self.rebar, self.reinforce, self.log_alpha, self.log_temperature, self.eta])
                summary_writer.add_summary(sum_str, iter)
            else:
                _, loss_val, g_val, g_val_r, la, t, e = sess.run(
                [self.train_op, ave_loss, self.rebar, self.reinforce, self.log_alpha, self.log_temperature, self.eta])


class RelaxedREBAROptimizer(REBAROptimizer):
    def __init__(self, sess, loss, q_func, log_alpha=None, dim=None, name="REBAR", learning_rate=.01, n_samples=1):
        self.q_func = q_func
        super(RelaxedREBAROptimizer, self).__init__(sess, loss, log_alpha, dim, name, learning_rate, n_samples)
        self.Q_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.Q_vars = [v for v in tf.trainable_variables() if "Q_func" in v.name]
        self._Q_gradvars()
        self.Q_opt_op = self.Q_optimizer.apply_gradients(self.Q_gradvars)
        old_var_op = self.variance_reduction_op
        with tf.control_dependencies([self.Q_opt_op, old_var_op]):
            self.variance_reduction_op = tf.no_op()


    def _create_loss_evaluations(self):
        """
        produces f(b), f(sig(z)), f(sig(z_tilde))
        """
        # relaxed inputs
        log_alpha = self._log_alpha
        sig_z = tf.nn.sigmoid(self.z / self.temperature + log_alpha)
        sig_z_tilde = tf.nn.sigmoid(self.z_tilde / self.temperature + log_alpha)
        # evaluate loss
        f_b = tf.reshape(self.loss(tf.reshape(self.b, [self.batch_size, -1])), [-1])
        z_inp = tf.reshape(sig_z, [self.batch_size, -1])
        z_tilde_inp = tf.reshape(sig_z_tilde, [self.batch_size, -1])
        l_z = self.loss(z_inp)
        l_z_tilde = self.loss(z_tilde_inp)
        with tf.variable_scope("Q_func"):
            f_z = tf.reshape(self.q_func(z_inp) + l_z, [-1])
        with tf.variable_scope("Q_func", reuse=True):
            f_z_tilde = tf.reshape(self.q_func(z_tilde_inp) + l_z_tilde, [-1])

        self.f_b = f_b
        self.f_z = f_z
        self.f_z_tilde = f_z_tilde
        tf.summary.scalar("f_b", tf.reduce_mean(self.f_b))
        tf.summary.scalar("f_z_tilde", tf.reduce_mean(self.f_z_tilde))
        tf.summary.scalar("f_z", tf.reduce_mean(self.f_z))

    def _Q_gradvars(self):
        """
        produces d[log p(b)]/d[log_alpha], d[f(sigma_theta(z))]/d[log_alpha], d[f(sigma_theta(z_tilde))]/d[log_alpha]
        """
        print(gs(self._rebar), gs(self.rebar))
        self.Q_gradvars = []
        for var in self.Q_vars:
            d_var_d_v = tf.gradients(
                tf.reduce_sum(tf.square(self._rebar)) / self.batch_size,
                var
            )[0]
            # d_var_d_v = tf.gradients(
            #     tf.reduce_mean(tf.square(self.f_b - self.f_z_tilde) + tf.square(self.f_b - self.f_z)),
            #     var
            # )[0]
            print(var.name, gs(d_var_d_v))
            self.Q_gradvars.append((d_var_d_v, var))



if __name__ == "__main__":
    def loss(b):
        bs, dim = gs(b)
        t = np.expand_dims(np.array(range(dim+2)[1:-1], dtype=np.float32) / (dim+2), 0)
        return tf.reduce_sum(tf.square(b - t), axis=1)
    sess = tf.Session()
    r_opt = REBAROptimizer(sess, loss, dim=10, learning_rate=.1, n_samples=1)

    """
    Bias and Variance test
    """
    r_opt.sess.run(tf.global_variables_initializer())
    summ_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("/tmp/rebar")
    percent_dims = []
    rebar_vars = []
    rebar_means = []
    reinforce_vars = []
    reinforce_means = []
    for iter in xrange(100):
        rebars = []
        reinforces = []
        for i in range(10000):
            [reb, ref] = sess.run([r_opt.rebar, r_opt.reinforce])
            rebars.append(reb)
            reinforces.append(ref)
        rebar_vars.append(np.var(rebars, axis=0))
        reinforce_vars.append(np.var(reinforces, axis=0))
        rebar_means.append(np.mean(rebars, axis=0))
        reinforce_means.append(np.mean(reinforces, axis=0))
        print("vars", np.mean(rebar_vars[-1]), np.mean(reinforce_vars[-1]))
        print("means", rebar_means[-1][3], reinforce_means[-1][3])
        print()
        sess.run(r_opt.train_op)




#
#     def train(self, n_steps=100000):
#         sess.run(tf.global_variables_initializer())
#         loss_vals = []
#         ave_loss = tf.reduce_mean(self.f_b)
#         summ_op = tf.summary.merge_all()
#         summary_writer = tf.summary.FileWriter("/tmp/rebar")
#         for iter in xrange(n_steps):
#             if iter % 100 == 0:
#                 _, sum_str, loss_val, g_val, la, t, s, ls, e = sess.run(
#                     [self.train_op, summ_op, ave_loss, self.rebar, self.log_alpha, self.log_temperature, self.log_scale,
#                      self.len_scale, self.eta])
#                 summary_writer.add_summary(sum_str, iter)
#             else:
#                 _, loss_val, g_val, la, t, s, ls, e = sess.run(
#                 [self.train_op, ave_loss, self.rebar, self.log_alpha, self.log_temperature, self.log_scale, self.len_scale, self.eta])
#             loss_vals.append(loss_val)
#             a = np.exp(la)
#             theta = a/(1+a)
#             if iter % 100 == 0:
#                 print(
#                     "iter {}, loss = {}\n grad = {}\n theta = {}\n temp = {}\n scale = {}\n len_scale = {}\n eta = {}\n".format(
#                         iter, loss_val, g_val, theta, np.exp(t), np.exp(s), np.exp(ls), e
#                     )
#                 )
#