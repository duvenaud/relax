from tensorflow.examples.tutorials.mnist import input_data
from rebar_tf import *
import tensorflow as tf
import numpy as np
import os
def encoder(x):
    if len(gs(x)) > 2:
        p = np.prod(gs(x)[1:])
        x = tf.reshape(x, [-1, p])
    h1 = tf.layers.dense(2. * x - 1., 200, tf.nn.relu, name="encoder_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="encoder_2")
    log_alpha = tf.layers.dense(h2, 200, name="encoder_out")
    return log_alpha

def decoder(b):
    h1 = tf.layers.dense(2. * b - 1., 200, tf.nn.relu, name="decoder_1")
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="decoder_2")
    log_alpha = tf.layers.dense(h2, 784, name="decoder_out")
    return log_alpha

def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 50, tf.nn.relu, name="q_1", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)
    scale = tf.get_variable(
        "q_scale", shape=[1], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True
    )
    return scale[0] * out


if __name__ == "__main__":
    TRAIN_DIR = "./rebar_new_u_and_v"
    reinforce = False
    relaxed = False
    if os.path.exists(TRAIN_DIR):
        print("Deleting existing train dir")
        import shutil

        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)
    sess = tf.Session()
    batch_size = 100
    lr = .0001
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def to_vec(t):
        return tf.reshape(t, [-1])
    def from_vec(t):
        return tf.reshape(t, [batch_size, -1])

    x = tf.placeholder(tf.float32, [batch_size, 784])
    x_im = tf.reshape(x, [batch_size, 28, 28, 1])
    tf.summary.image("x_true", x_im)
    x_binary = tf.to_float(x > .5)
    log_alpha = encoder(x_binary)
    log_alpha_v = tf.reshape(log_alpha, [-1])
    evals = 0
    def loss(b):
        log_q_b_given_x = bernoulli_loglikelihood(b, log_alpha)
        log_q_b_given_x = tf.reduce_mean(tf.reduce_sum(log_q_b_given_x, axis=1))

        log_p_b = bernoulli_loglikelihood(b, tf.zeros_like(log_alpha))
        log_p_b = tf.reduce_mean(tf.reduce_sum(log_p_b, axis=1))

        with tf.variable_scope("decoder", reuse=evals>0):
            log_alpha_x_batch = decoder(b)
        log_p_x_given_b = bernoulli_loglikelihood(x_binary, log_alpha_x_batch)
        log_p_x_given_b = tf.reduce_mean(tf.reduce_sum(log_p_x_given_b, axis=1))
        # HACKY BS
        global evals
        if evals == 0:
            # if first eval make image summary
            a = tf.exp(log_alpha_x_batch)
            log_theta_x = a / (1 + a)
            log_theta = tf.reshape(log_theta_x, [batch_size, 28, 28, 1])
            tf.summary.image("x_pred", log_theta)
        evals += 1
        return -tf.expand_dims(log_p_x_given_b + log_p_b - log_q_b_given_x, 0)
    if relaxed:
        rebar_optimizer = RelaxedREBAROptimizer(sess, loss, Q_func, log_alpha=log_alpha, learning_rate=lr)
    else:
        rebar_optimizer = REBAROptimizer(sess, loss, log_alpha=log_alpha, learning_rate=lr)
    gen_loss = rebar_optimizer.f_b
    tf.summary.scalar("loss", gen_loss[0])
    gen_opt = tf.train.AdamOptimizer(lr)
    gen_vars = [v for v in tf.trainable_variables() if "decoder" in v.name]
    gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
    gen_train_op = gen_opt.apply_gradients(gen_gradvars)

    alpha_grads = rebar_optimizer.reinforce if reinforce else rebar_optimizer.rebar
    inf_vars = [v for v in tf.trainable_variables() if "encode" in v.name]
    inf_grads = tf.gradients(log_alpha, inf_vars, grad_ys=alpha_grads)
    inf_gradvars = zip(inf_grads, inf_vars)
    inf_opt = tf.train.AdamOptimizer(lr)
    inf_train_op = inf_opt.apply_gradients(inf_gradvars)
    if relaxed:
        gradvars = inf_gradvars + gen_gradvars + rebar_optimizer.variance_gradvars + rebar_optimizer.Q_gradvars
    else:
        gradvars = inf_gradvars + gen_gradvars + rebar_optimizer.variance_gradvars
    for g, v in gradvars:
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name+"_grad", g)

    if reinforce:
        with tf.control_dependencies([gen_train_op, inf_train_op]):
            train_op = tf.no_op()
    else:
        with tf.control_dependencies([gen_train_op, inf_train_op, rebar_optimizer.variance_reduction_op]):
            train_op = tf.no_op()

    summ_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TRAIN_DIR)
    sess.run(tf.global_variables_initializer())
    for i in range(250000):
        batch_xs, _ = dataset.train.next_batch(100)
        if i % 100 == 0:
            loss, _, sum_str = sess.run([gen_loss, train_op, summ_op], feed_dict={x: batch_xs})
            summary_writer.add_summary(sum_str, i)
            print(i, loss[0])
        else:
            loss, _ = sess.run([gen_loss, train_op], feed_dict={x: batch_xs})

