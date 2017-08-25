import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
ion()
import matplotlib

matplotlib.rc("savefig") #, dpi=300)


def safe_log_prob(x, eps=1e-8):
    return tf.log(tf.clip_by_value(x, eps, 1.0))


def softplus(x):
    '''
    Let m = max(0, x), then,

    sofplus(x) = log(1 + e(x)) = log(e(0) + e(x)) = log(e(m)(e(-m) + e(x-m)))
                         = m + log(e(-m) + e(x - m))

    The term inside of the log is guaranteed to be between 1 and 2.
    '''
    m = tf.maximum(tf.zeros_like(x), x)
    return m + tf.log(tf.exp(-m) + tf.exp(x - m))


class REBAROptimizer:
    def __init__(self, sess, loss, name="REBAR", learning_rate=.1, n_samples=1):
        self.name = name
        self.sess = sess
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.variance_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        """ model parameters """
        # alpha = theta / (1 - theta)
        self.log_alpha = tf.Variable(0.0, name='log_alpha', dtype=tf.float64)  # initial value
        self.log_temperature = tf.Variable(
            np.log(.5),
            trainable=False,
            name='log_temperature',
            dtype=tf.float64
        )
        self.tiled_log_temperature = tf.tile([self.log_temperature], [n_samples])
        self.temperature = tf.exp(self.tiled_log_temperature)
        self.eta = tf.Variable(
            1.0,
            trainable=False,
            name='eta_logit',
            dtype=tf.float64
        )
        """ reparameterization noise """
        self.b, self.z, self.z_tilde = self._create_reparam_variables()
        """ relaxed loss evaluations """
        self.f_b, self.f_z, self.f_z_tilde = self._create_loss_evaluations()
        """ gradvars for optimizers """
        self.rebar_gradvars, self.variance_gradvars = self._create_gradvars()
        """ optimization operation """
        depends = [gv[0] for gv in self.rebar_gradvars + self.variance_gradvars]
        with tf.control_dependencies(depends):
            self.train_op = tf.group(
                self.optimizer.apply_gradients(self.rebar_gradvars),
                self.variance_optimizer.apply_gradients(self.variance_gradvars)
            )

    def _create_reparam_variables(self):
        # noise for generating z
        u = tf.random_uniform([self.n_samples], dtype=tf.float64)
        # logistic reparameterization z = g(u, log_alpha)
        z = self.log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)
        # b = H(z)
        b = tf.to_double(tf.stop_gradient(z > 0))
        # g(u', log_alpha) = 0
        u_prime = tf.nn.sigmoid(-self.log_alpha)
        proto_v = tf.random_uniform([self.n_samples], dtype=tf.float64)
        proto_v_1 = proto_v * (1 - u_prime) + u_prime
        proto_v_0 = proto_v * u_prime
        v = b * proto_v_1 + (1 - b) * proto_v_0
        z_tilde = self.log_alpha + safe_log_prob(v) - safe_log_prob(1 - v)
        return b, z, z_tilde


    def _create_loss_evaluations(self):
        """
        produces f(b), f(sig(z)), f(sig(z_tilde))
        """
        # relaxed inputs
        sig_z = tf.nn.sigmoid(self.z / self.temperature + self.log_alpha)
        sig_z_tilde = tf.nn.sigmoid(self.z_tilde / self.temperature + self.log_alpha)
        # evaluate loss
        f_b = self.loss(self.b)
        f_z = self.loss(sig_z)
        f_z_tilde = self.loss(sig_z_tilde)
        return f_b, f_z, f_z_tilde

    def _create_gradvars(self):
        """
        produces d[log p(b)]/d[log_alpha], d[f(sigma_theta(z))]/d[log_alpha], d[f(sigma_theta(z_tilde))]/d[log_alpha]
        """
        log_p = (self.b * (-softplus(-self.log_alpha)) + (1 - self.b) * (-self.log_alpha - softplus(-self.log_alpha)))
        # [f(b) - n * f(sig(z_tilde))] * d[log p(b)]/d[log_alpha]
        term1 = tf.gradients(
            tf.reduce_mean((tf.stop_gradient(self.f_b) - tf.stop_gradient(self.eta * self.f_z_tilde)) * log_p),
            self.log_alpha
        )[0]
        # eta * d[f(sigma_theta(z))]/d[log_alpha] - eta * d[f(sigma_theta(z_tilde))]/d[log_alpha]
        term2 = tf.gradients(
            tf.reduce_mean(self.f_z - self.f_z_tilde),
            self.log_alpha
        )[0]
        # rebar gradient estimator
        rebar = term1 + self.eta * term2
        # now compute gradients of the variance of this wrt other parameters
        # eta
        d_term1_d_eta = tf.gradients(
            tf.reduce_mean(-1. * tf.stop_gradient(self.f_z_tilde) * log_p),
            self.log_alpha
        )[0]
        d_rebar_d_eta = d_term1_d_eta + term2
        d_var_d_eta = 2. * rebar * d_rebar_d_eta
        # temperature
        d_f_z_tilde_d_tilded_temperature = tf.gradients(
            self.f_z_tilde,
            self.tiled_log_temperature
        )[0]
        d_term1_d_temperature = tf.gradients(
            tf.reduce_mean(-1. * self.eta * tf.stop_gradient(d_f_z_tilde_d_tilded_temperature) * log_p),
            self.log_alpha
        )[0]
        d_term2_d_temperature = self.eta * tf.gradients(
            tf.reduce_mean(term2),
            self.log_temperature
        )[0]
        d_rebar_d_temperature = d_term1_d_temperature + d_term2_d_temperature
        d_var_d_temperature = 2. * rebar * d_rebar_d_temperature
        self.rebar = rebar
        return [(rebar, self.log_alpha)], [(d_var_d_eta, self.eta), (d_var_d_temperature, self.log_temperature)]

    def train(self, n_steps=5000):
        sess.run(tf.global_variables_initializer())
        p_vals = []
        loss_vals = []
        g_var = []
        for iter in xrange(n_steps):
            _, loss_val, g_val, la, t, e = sess.run(
                [self.train_op, self.f_b, self.rebar, self.log_alpha, self.log_temperature, self.eta])
            loss_vals.append(loss_val)
            a = np.exp(la)
            theta = a/(1+a)
            if iter % 100 == 0:
                print("iter {}, loss = {}, grad = {:.5f}, theta = {:.5f}, temp = {:.5f}, eta = {:.5f}".format(
                    iter, loss_val[0], g_val, theta, np.exp(t), e
                ))

if __name__ == "__main__":
    def loss(b, t=.49):
        return tf.square(b - t)
    sess = tf.Session()
    rebar_optimizer = REBAROptimizer(sess, loss)
    rebar_optimizer.train()
