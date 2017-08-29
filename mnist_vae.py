from tensorflow.examples.tutorials.mnist import input_data
from rebar_tf import *
import tensorflow as tf
import numpy as np
tf.nn.rnn_cell.BasicLSTMCell
def encoder(x):
    if len(gs(x)) > 2:
        p = np.prod(gs(x)[1:])
        x = tf.reshape(x, [-1, p])
    h1 = tf.layers.dense(x, 200, tf.nn.relu, name="encoder_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="encoder_2")
    log_alpha = tf.layers.dense(h2, 200, name="encoder_out")
    return log_alpha

def decoder(b):
    h1 = tf.layers.dense(2. * b - 1., 200, tf.nn.relu, name="decoder_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="decoder_2")
    log_theta = tf.layers.dense(h2, 200, name="decoder_out")
    return log_alpha



if __name__ == "__main__":
    sess = tf.Session()
    batch_size = 100
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    batch_xs, batch_ys = dataset.train.next_batch(100)
    def to_vec(t):
        return tf.reshape(t, [-1])
    def from_vec(t):
        return tf.reshape(t, [batch_size, -1])

    x = tf.placeholder(tf.float32, [batch_size, 784])
    x_binary = tf.to_float(x > .5)
    log_alpha = encoder(x_binary)
    log_alpha_v = tf.reshape(log_alpha, [-1])
    def loss(b):
        log_q_b_given_x = bernoulli_loglikelihood(b, log_alpha_v)
        log_q_b_given_x = tf.reduce_mean(tf.reduce_sum(from_vec(log_q_b_given_x), axis=1))

        log_p_b = bernoulli_loglikelihood(b, tf.zeros_like(log_alpha_v))
        log_p_b = tf.reduce_mean(tf.reduce_sum(from_vec(log_p_b)), axis=1)

        b_batch = from_vec(b)
        log_alpha_x_batch = decoder(b_batch)
        log_alpha_x = to_vec(log_alpha_x_batch)
        x_v = to_vec(x_binary)
        log_p_x_given_b = bernoulli_loglikelihood(x_v, log_alpha_x)
        log_p_x_given_b = tf.reduce_mean(tf.reduce_sum(from_vec(log_p_x_given_b), axis=1))
        return log_p_x_given_b + log_p_b - log_q_b_given_x

    rebar_optimzer = REBAROptimizer(sess, loss, log_alpha_v)
    gen_loss = rebar_optimizer.f_b
    gen_opt = tf.train.AdamOptimizer(.0001)
    gen_vars = [v for v in tf.trainable_variables() if "decoder" in v.name]
    gen_train_op = gen_opt.minimize(gen_loss, var_list=gen_vars)


