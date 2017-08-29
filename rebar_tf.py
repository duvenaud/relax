import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
ion()
import matplotlib

matplotlib.rc("savefig") #, dpi=300)


def safe_log_prob(x, eps=1e-8):
    return tf.log(tf.clip_by_value(x, eps, 1.0))


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


class REBAROptimizer:
    def __init__(self, sess, loss, log_alpha=None, dim=None, name="REBAR", learning_rate=.1, n_samples=1):
        self.name = name
        self.sess = sess
        self.loss = loss
        self.dim = dim
        self.log_alpha = log_alpha
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.variance_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        """ model parameters """
        self._create_model_parameters()
        """ reparameterization noise """
        self._create_reparam_variables()
        """ relaxed loss evaluations """
        self._create_loss_evaluations()
        """ gradvars for optimizers """
        self._create_gradvars()
        """ optimization operation """
        depends = [gv[0] for gv in self.rebar_gradvars + self.variance_gradvars]
        with tf.control_dependencies(depends):
            self.train_op = tf.group(
                self.optimizer.apply_gradients(self.rebar_gradvars),
                self.variance_optimizer.apply_gradients(self.variance_gradvars)
            )

    def _create_model_parameters(self):
        # alpha = theta / (1 - theta)
        if self.log_alpha is None:
            "no log alpha given, creating here"
            self.log_alpha = tf.Variable(
                [0.0 for i in range(self.dim)],  # initial value
                name='log_alpha', dtype=tf.float64
            )
        else:
            self.dim = gs(self.log_alpha)[0]
        a = tf.exp(self.log_alpha)
        theta = a / (1 + a)
        tf.summary.histogram("theta", theta)
        # expanded version for internal purposes
        self._log_alpha = tf.expand_dims(self.log_alpha, 0)
        self.log_temperature = tf.Variable(
            [np.log(.5) for i in range(self.dim)],
            trainable=False,
            name='log_temperature',
            dtype=tf.float64
        )
        self.tiled_log_temperature = tf.tile([self.log_temperature], [self.n_samples, 1])
        self.temperature = tf.exp(self.tiled_log_temperature)
        tf.summary.histogram("temp", self.temperature)
        self.eta = tf.Variable(
            [1.0 for i in range(self.dim)],
            trainable=False,
            name='eta',
            dtype=tf.float64
        )

    def _create_reparam_variables(self):
        # noise for generating z
        u = tf.random_uniform([self.n_samples, self.dim], dtype=tf.float64)
        log_alpha = self._log_alpha
        # logistic reparameterization z = g(u, log_alpha)
        z = log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)
        # b = H(z)
        b = tf.to_double(tf.stop_gradient(z > 0))
        # g(u', log_alpha) = 0
        u_prime = tf.nn.sigmoid(-log_alpha)
        proto_v = tf.random_uniform([self.n_samples, self.dim], dtype=tf.float64)
        proto_v_1 = proto_v * (1 - u_prime) + u_prime
        proto_v_0 = proto_v * u_prime
        v = b * proto_v_1 + (1 - b) * proto_v_0
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
        f_b = self.loss(self.b)
        f_z = self.loss(sig_z)
        f_z_tilde = self.loss(sig_z_tilde)
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
        log_p = bernoulli_loglikelihood(self.b, log_alpha)
        # [f(b) - n * f(sig(z_tilde))] * d[log p(b)]/d[log_alpha]
        l = tf.reduce_mean((tf.stop_gradient(f_b) - tf.stop_gradient(eta * f_z_tilde)) * log_p, axis=0)
        l_reinforce = tf.reduce_mean(tf.stop_gradient(f_b) * log_p, axis=0)
        reinforce = tf.gradients(
            l_reinforce,
            self.log_alpha
        )[0]
        term1 = tf.gradients(
            l,
            self.log_alpha
        )[0]
        # d[f(sigma_theta(z))]/d[log_alpha] - eta * d[f(sigma_theta(z_tilde))]/d[log_alpha]
        term2 = tf.gradients(
            tf.reduce_mean(self.f_z - self.f_z_tilde),
            self.log_alpha
        )[0]
        # rebar gradient estimator
        rebar = term1 + self.eta * term2
        # now compute gradients of the variance of this wrt other parameters
        # eta
        d_term1_d_eta = tf.gradients(
            tf.reduce_mean(-1. * tf.stop_gradient(f_z_tilde) * log_p, axis=0),
            self.log_alpha
        )[0]
        d_rebar_d_eta = d_term1_d_eta + term2
        d_var_d_eta = 2. * rebar * d_rebar_d_eta
        # temperature
        d_f_z_tilde_d_tilded_temperature = tf.gradients(
            f_z_tilde,
            self.tiled_log_temperature
        )[0]

        d_term1_d_temperature = tf.gradients(
            tf.reduce_mean(-1. * self.eta * tf.stop_gradient(d_f_z_tilde_d_tilded_temperature) * log_p, axis=0),
            self.log_alpha
        )[0]
        d_term2_d_temperature = self.eta * tf.gradients(
            tf.reduce_mean(term2),
            self.log_temperature
        )[0]
        d_rebar_d_temperature = d_term1_d_temperature + d_term2_d_temperature
        d_var_d_temperature = 2. * rebar * d_rebar_d_temperature
        self.rebar = rebar
        self.reinforce = reinforce
        tf.summary.histogram("rebar_gradient", rebar)
        self.rebar_gradvars = [(rebar, self.log_alpha)]
        self.variance_gradvars = [(d_var_d_eta, self.eta), (d_var_d_temperature, self.log_temperature)]

    def train(self, n_steps=10000):
        self.sess.run(tf.global_variables_initializer())
        loss_vals = []
        ave_loss = tf.reduce_mean(self.f_b)
        summ_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("/tmp/rebar")
        percent_dims = []
        for iter in xrange(n_steps):
            if iter % 100 == 0:
                _, sum_str, loss_val, g_val, g_val_r, la, t, e = sess.run(
                    [self.train_op, summ_op, ave_loss, self.rebar, self.reinforce, self.log_alpha, self.log_temperature, self.eta])
                summary_writer.add_summary(sum_str, iter)
            else:
                _, loss_val, g_val, g_val_r, la, t, e = sess.run(
                [self.train_op, ave_loss, self.rebar, self.reinforce, self.log_alpha, self.log_temperature, self.eta])


class RelaxedREBAROptimizer(REBAROptimizer):
    """
    We have a more heaviliy parameterized continuous relaxation
    instead of sig(x / t + log_alpha) we have s * sig(x / t + log_alpha)
    """
    def _create_model_parameters(self):
        # alpha = theta / (1 - theta)
        self.log_alpha = tf.Variable(
            [0.0 for i in range(self.dim)],  # initial value
            name='log_alpha', dtype=tf.float64
        )
        a = tf.exp(self.log_alpha)
        theta = a/(1+a)
        tf.summary.histogram("theta", theta)
        # expanded version for internal purposes
        self._log_alpha = tf.expand_dims(self.log_alpha, 0)
        self.log_temperature = tf.Variable(
            [np.log(.5) for i in range(self.dim)],
            trainable=False,
            name='log_temperature',
            dtype=tf.float64
        )
        self.tiled_log_temperature = tf.tile([self.log_temperature], [self.n_samples, 1])
        self.temperature = tf.exp(self.tiled_log_temperature)
        tf.summary.histogram("temp", self.temperature)
        self.eta = tf.Variable(
            [1.0 for i in range(self.dim)],
            trainable=False,
            name='eta',
            dtype=tf.float64
        )
        self.log_scale = tf.Variable(
            [np.log(.5) for i in range(self.dim)],
            trainable=False,
            name='log_scale',
            dtype=tf.float64
        )
        self.tiled_log_scale = tf.tile([self.log_scale], [self.n_samples, 1])
        self.scale = tf.exp(self.tiled_log_scale)
        tf.summary.histogram("scale", self.scale)
        self.log_len_scale = tf.Variable(
            [np.log(1.) for i in range(self.dim)],
            trainable=False,
            name='log_len_scale',
            dtype=tf.float64
        )
        self.tiled_log_len_scale = tf.tile([self.log_len_scale], [self.n_samples, 1])
        self.len_scale = tf.exp(self.tiled_log_len_scale)
        tf.summary.histogram("len_scale", self.len_scale)

    @staticmethod
    def relaxation(input, temp, log_alpha, scale, len_scale):
        relax = scale * tf.nn.sigmoid(input / len_scale + log_alpha) - ((scale - 1.) / 2.)
        true = tf.to_double(tf.stop_gradient(input > 0))
        # as temp -> 0 we want relaxation to converge to heavy-side
        r = tf.sigmoid(1. / temp)
        return r * true + (1 - r) * relax

    def _create_loss_evaluations(self):
        """
        produces f(b), f(sig(z)), f(sig(z_tilde))
        """
        # relaxed inputs
        log_alpha = self._log_alpha
        sig_z = self.relaxation(self.z, self.temperature, log_alpha, self.scale, self.len_scale)
        sig_z_tilde = self.relaxation(self.z_tilde, self.temperature, log_alpha, self.scale, self.len_scale)
        # evaluate loss
        f_b = self.loss(self.b)
        f_z = self.loss(sig_z)
        f_z_tilde = self.loss(sig_z_tilde)
        self.f_b = f_b
        self.f_z = f_z
        self.f_z_tilde = f_z_tilde

    def variance_gradient(self, param, tiled_param, f_z_tilde, log_p, term2, rebar):
        d_f_z_tilde_d_tiled_param = tf.gradients(
            f_z_tilde,
            tiled_param
        )[0]
        d_term1_d_param = tf.gradients(
            tf.reduce_mean(-1. * self.eta * tf.stop_gradient(d_f_z_tilde_d_tiled_param) * log_p, axis=0),
            self.log_alpha
        )[0]
        d_term2_d_param = self.eta * tf.gradients(
            tf.reduce_mean(term2),
            param
        )[0]
        d_rebar_d_param = d_term1_d_param + d_term2_d_param
        d_var_d_param = 2. * rebar * d_rebar_d_param
        return d_var_d_param

    def _create_gradvars(self):
        """
        produces d[log p(b)]/d[log_alpha], d[f(sigma_theta(z))]/d[log_alpha], d[f(sigma_theta(z_tilde))]/d[log_alpha]
        """
        log_alpha = self._log_alpha
        eta = tf.expand_dims(self.eta, 0)
        f_b = tf.expand_dims(self.f_b, 1)
        f_z_tilde = tf.expand_dims(self.f_z_tilde, 1)
        log_p = (self.b * (-softplus(-log_alpha)) + (1 - self.b) * (-log_alpha - softplus(-log_alpha)))
        # [f(b) - n * f(sig(z_tilde))] * d[log p(b)]/d[log_alpha]
        l = tf.reduce_mean((tf.stop_gradient(f_b) - tf.stop_gradient(eta * f_z_tilde)) * log_p, axis=0)
        term1 = tf.gradients(
            l,
            self.log_alpha
        )[0]
        # d[f(sigma_theta(z))]/d[log_alpha] - eta * d[f(sigma_theta(z_tilde))]/d[log_alpha]
        term2 = tf.gradients(
            tf.reduce_mean(self.f_z - self.f_z_tilde),
            self.log_alpha
        )[0]
        # rebar gradient estimator
        rebar = term1 + self.eta * term2

        # now compute gradients of the variance of this wrt other parameters
        # eta
        d_term1_d_eta = tf.gradients(
            tf.reduce_mean(-1. * tf.stop_gradient(f_z_tilde) * log_p, axis=0),
            self.log_alpha
        )[0]
        d_rebar_d_eta = d_term1_d_eta + term2
        d_var_d_eta = 2. * rebar * d_rebar_d_eta
        # relaxation parameters
        d_var_d_temperature = self.variance_gradient(
            self.log_temperature, self.tiled_log_temperature, f_z_tilde, log_p, term2, rebar
        )
        d_var_d_scale = self.variance_gradient(
            self.log_scale, self.tiled_log_scale, f_z_tilde, log_p, term2, rebar
        )
        d_var_d_len_scale = self.variance_gradient(
            self.log_len_scale, self.tiled_log_len_scale, f_z_tilde, log_p, term2, rebar
        )

        self.rebar = rebar
        tf.summary.histogram("rebar_gradient", rebar)
        self.rebar_gradvars = [(rebar, self.log_alpha)]
        self.variance_gradvars = [
            (d_var_d_eta, self.eta),
            (d_var_d_temperature, self.log_temperature),
            (d_var_d_scale, self.log_scale),
            (d_var_d_len_scale, self.log_len_scale)
        ]

    def train(self, n_steps=100000):
        sess.run(tf.global_variables_initializer())
        loss_vals = []
        ave_loss = tf.reduce_mean(self.f_b)
        summ_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("/tmp/rebar")
        for iter in xrange(n_steps):
            if iter % 100 == 0:
                _, sum_str, loss_val, g_val, la, t, s, ls, e = sess.run(
                    [self.train_op, summ_op, ave_loss, self.rebar, self.log_alpha, self.log_temperature, self.log_scale,
                     self.len_scale, self.eta])
                summary_writer.add_summary(sum_str, iter)
            else:
                _, loss_val, g_val, la, t, s, ls, e = sess.run(
                [self.train_op, ave_loss, self.rebar, self.log_alpha, self.log_temperature, self.log_scale, self.len_scale, self.eta])
            loss_vals.append(loss_val)
            a = np.exp(la)
            theta = a/(1+a)
            if iter % 100 == 0:
                print(
                    "iter {}, loss = {}\n grad = {}\n theta = {}\n temp = {}\n scale = {}\n len_scale = {}\n eta = {}\n".format(
                        iter, loss_val, g_val, theta, np.exp(t), np.exp(s), np.exp(ls), e
                    )
                )



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