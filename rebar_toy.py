from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import pandas
import seaborn as sns
sns.set()
sns.set_style("white", {"axes.edgecolor": ".7"})
sns.set_style("ticks")
# sns.set_style("whitegrid")

# Tableau 20 Colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


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
    return tf.log(tf.exp(-(z-loc)/scale)/scale*tf.square((1+tf.exp(-(z-loc)/scale))))


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-softplus(-log_alpha)) + (1 - b) * (-log_alpha - softplus(-log_alpha))


def bernoulli_loglikelihood_derivitive(b, log_alpha):
    assert gs(b) == gs(log_alpha)
    sna = tf.sigmoid(-log_alpha)
    return b * sna - (1-b) * (1 - sna)


def v_from_u(u, log_alpha, force_same=True, b=None, v_prime=None):
    u_prime = tf.nn.sigmoid(-log_alpha)
    if not force_same:
        v = b*(u_prime+v_prime*(1-u_prime)) + (1-b)*v_prime*u_prime
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
    #h2 = tf.layers.dense(h1, 10, tf.nn.relu, name="q_2", use_bias=True)
    #h3 = tf.layers.dense(h2, 10, tf.nn.relu, name="q_3", use_bias=True)
    #h4 = tf.layers.dense(h3, 10, tf.nn.relu, name="q_4", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)
    scale = tf.get_variable(
        "q_scale", shape=[1], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True
    )
    return scale[0] * out
#    return out

def loss_func(b, t):
    return tf.reduce_mean(tf.square(b - t), axis=1)


def main(t=0.499, rand_seed=42, use_reinforce=False, relaxed=False, visualize=False,
         log_var=False, tf_log=False, force_same=False, test_bias=False,
         train_to_completion=False, use_exact_gradient=False, BAR=False, LAX=False, train_theta=True, square_loss=False):
    with tf.Session() as sess:
        TRAIN_DIR = "./toy_problem"
        if os.path.exists(TRAIN_DIR):
            print("Deleting existing train dir")
            import shutil

            shutil.rmtree(TRAIN_DIR)
        os.makedirs(TRAIN_DIR)
        iters = ITERS # todo: change back
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
        z = reparameterize(log_alpha, u) # z(u)
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
        if relaxed == "relaxation":
            with tf.variable_scope("Q_func"):
                sig_z = Q_func(z)
            with tf.variable_scope("Q_func", reuse=True):
                sig_z_tilde = Q_func(z_tilde)
            f_z = loss_func(sig_z, target)
            f_z_tilde = loss_func(sig_z_tilde, target)

        else:
            # relaxation variables
            batch_temp = tf.expand_dims(temperature, 0)
            sig_z = concrete_relaxation(z, batch_temp)
            sig_z_tilde = concrete_relaxation(z_tilde, batch_temp)

            f_z = loss_func(sig_z, target)
            f_z_tilde = loss_func(sig_z_tilde, target)

            if relaxed != False:
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
#        d_log_pb_d_log_alpha = bernoulli_loglikelihood_derivitive(b, log_alpha)
        d_log_pb_d_log_alpha = tf.gradients(bernoulli_loglikelihood(b, log_alpha), log_alpha)[0]
        d_log_pz_d_log_alpha = tf.gradients(logistic_loglikelihood(z, log_alpha), log_alpha)[0]
        # check shapes are alright
        assert_same_shapes(d_f_z_d_log_alpha, d_f_z_tilde_d_log_alpha, d_log_pb_d_log_alpha, d_log_pz_d_log_alpha)
        assert_same_shapes(f_b, f_z_tilde)
        batch_eta = tf.expand_dims(eta, 0)
        batch_f_b = tf.expand_dims(f_b, 1)
        batch_f_z_tilde = tf.expand_dims(f_z_tilde, 1)
        # do one of LAX, BAR, relaxed-REBAR, or REBAR
        if LAX or BAR:
            batch_f_z = tf.expand_dims(f_z, 1)
            rebar = batch_f_b*d_log_pb_d_log_alpha - batch_eta*batch_f_z*d_log_pz_d_log_alpha + batch_eta*d_f_z_d_log_alpha
#            rebar = (batch_f_b - batch_f_z) * d_log_pb_d_log_alpha + (d_f_z_d_log_alpha)
        elif relaxed == "super":
            rebar = (batch_f_b - batch_f_z_tilde) * d_log_pb_d_log_alpha + (d_f_z_d_log_alpha - d_f_z_tilde_d_log_alpha)
        else:
            rebar = (batch_f_b - batch_eta * batch_f_z_tilde) * d_log_pb_d_log_alpha + batch_eta * (d_f_z_d_log_alpha - d_f_z_tilde_d_log_alpha)
        reinforce = batch_f_b * d_log_pb_d_log_alpha
        exact_gradient = tf.stop_gradient(tf.square(1 - target) - tf.square(-target)) * tf.nn.sigmoid(log_alpha)
        tf.summary.histogram("rebar", rebar)
        tf.summary.histogram("reinforce", reinforce)

        # variance reduction objective
        variance_loss = tf.reduce_mean(tf.square(rebar))

        # optimizers
        inf_opt = tf.train.AdamOptimizer(lr)

        # need to scale by batch size cuz tf.gradients sums
        if use_reinforce:
            log_alpha_grads = reinforce/batch_size
        elif use_exact_gradient:
            log_alpha_grads = exact_gradient/batch_size
        else:
            log_alpha_grads = rebar/batch_size
          
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

        if use_reinforce or use_exact_gradient:
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
        
        variances = []
        losses = []
        thetas = []
        FBs = []
        FZs = []
        print("Collecting {} samples".format(ITERS//RESOLUTION))
        for i in tqdm(range(iters)):
            if (i+1) % RESOLUTION == 0:
                if train_to_completion:
                    for _ in tqdm(range(1000)):
                        sess.run(var_train_op)
                        
                if tf_log:
                    if train_theta:
                        loss_value, _, sum_str, theta_value = sess.run([loss, train_op, summ_op, theta])
                    else:
                        loss_value, _, sum_str, theta_value = sess.run([loss, var_train_op, summ_op, theta]) # just train eta and temp
                    summary_writer.add_summary(sum_str, i)
                else:
                    if train_theta:
                        loss_value, _, theta_value, temp = sess.run([loss, train_op, theta, temperature])
                    else:
                        loss_value, _, theta_value, temp = sess.run([loss, var_train_op, theta, temperature]) # just train eta and temp
                    
                tv = theta_value[0][0]
                thetas.append(tv)
                losses.append(tv*(1-target[0][0])**2+(1-tv)*target[0][0]**2)
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

                if test_bias:
                    n_variance_samples = 1000
                    rebars = []
                    reinforces = []
                    for _ in tqdm(range(n_variance_samples)):
                        rb, re = sess.run([rebar, reinforce])
                        rebars.append(rb)
                        reinforces.append(re)
                    rebars = np.array(rebars)
                    reinforces = np.array(reinforces)
                    re_var = np.log(reinforces.var(axis=0))
                    rb_var = np.log(rebars.var(axis=0))
                    if use_reinforce:
                      variances.append(np.mean(re_var))
                    else:
                      variances.append(np.mean(rb_var))
                    diffs = np.abs(rebars.mean(axis=0) - reinforces.mean(axis=0))
                    sess.run([rebar_var.assign(rb_var), reinforce_var.assign(re_var), est_diffs.assign(diffs)])
                    print("rebar variance = {}".format(rb_var.mean()))
                    print("reinforce variance = {}".format(re_var.mean()))
                    print("rebar     = {}".format(rebars.mean(axis=0)[0]))
                    print("reinforce = {}\n".format(reinforces.mean(axis=0)[0]))

                if visualize == "f":
                    # run variance reduction operation
                    for i in range(1000):
                        sess.run(var_train_op)
                    X = [float(i) / 100 for i in range(100)]
                    FZ = []
                    for x in X:
                        fz = sess.run(f_z, feed_dict={sig_z: [[x]]})
                        FZ.append(fz)
                    plt.plot(X, FZ)
                    plt.show()
                elif visualize == "sig":
                    us = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
                    FB = []
                    FZ = []
                    for _u in us:
                        fb = sess.run(f_b, feed_dict={u: [[_u]], log_alpha:[[0.0]]})
                        fz = sess.run(f_z, feed_dict={u: [[_u]], log_alpha:[[0.0]]})
                        FB.append(fb)
                        FZ.append(fz)
                    FBs.append(FB)
                    FZs.append(FZ)
#                    plt.plot(us, FB, 'red', label='f(b=H(z))')
#                    if not relaxed:
#                      plt.plot(us, FZ, 'blue', label='f(sigmoid(z/temp))')
#                    elif relaxed in [True, "super"]:
#                      plt.plot(us, FZ, 'blue', label='Q(z)')
#                    plt.xlabel('u')
#                    plt.legend(bbox_to_anchor=(1.0,0.5))
#                    plt.show()
                    #plt.savefig('/home/damichoi/ml/relaxed-rebar/test.png')
                        

            else:
                if train_to_completion:
                    for _ in tqdm(range(100)):
                        sess.run(var_train_op)
                if train_theta:
                    _, = sess.run([train_op])
                else:
                    _, = sess.run([var_train_op])
                
        tv = None
        print(tv)
#        return tv, thetas, losses, variances, FBs, FZs
        return tv, thetas, losses, variances, FBs, FZs


if __name__ == "__main__":
    t = 0.499
    rand_seed = 42

    print("INFO: Collecting data for plots...")

## FIGURE 3
    output_file_name = "relaxations_plot_new_{}_{}.pkl".format(ITERS, ITERS / RESOLUTION)  # file to save results
    try:
        with open(output_file_name, 'r') as f:
            FB, FZ, QZ, us = pickle.load(f)
            print("\nINFO: Loaded precomputed data.")
    except IOError:
        print("INFO: No precomputed data found, running toy example experiments.")
        _,relax_thetas,relax_losses,relax_variances,_,QZ = main(t=t, relaxed="super", visualize="sig", force_same=True, test_bias=False, train_to_completion=False, train_theta=False)
        _,rebar_thetas,rebar_losses,rebar_variances,FB,FZ = main(t=t, relaxed=False, visualize="sig", force_same=True, test_bias=False, train_to_completion=False, train_theta=False)
        us = np.linspace(0,1,len(FB[0]))
        with open(output_file_name, 'w') as f:
            pickle.dump([FB, FZ, QZ, us], f)

    # Outputs ###.png for composition into gif
    print("INFO: Generating and saving plots...")
    try:
        os.stat('./gif')
    except:
        os.mkdir('./gif')

    for which in tqdm(range(len(FB))):
        with sns.axes_style("whitegrid"):
            sns.set_style("ticks")
            sns.set_context("poster")

            f, (ax1, ax2) = plt.subplots(2, sharex=True)
            # ax1.plot(us, FB[which], color=tableau20[0],  label=r'$f(b=H(z(u)))$',)
            # ax1.legend(loc='best')
            # plt.xlabel(r"$u \sim \operatorname{Unif}[0,1]$")

            ax1.plot(us,  FZ[which], color=tableau20[4],label=r'$f(\sigma_\lambda(z(u)))$')
            # ax1.legend(loc='best')
            ax2.plot(us, QZ[which], color=tableau20[6], label=r'$c_\phi(z(u))$')
            # ax2.legend(loc='best')
        # Fine-tune figure; make subplots close to each other and hide x ticks for
        # all but bottom plot.
            f.subplots_adjust(hspace=0.5)

            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            plt.xlabel(r"$u$")
            plt.ylim([0.24, 0.26])
            sns.despine(offset=0.125, trim=True)
            # plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
            plt.tight_layout()
            plt.savefig("./gif/{}.png".format(str(which).zfill(3), which), format='png')
            plt.close(f)
#
#
#
    #
    # f, axes = plt.subplots(3,3)
    # for i, ax in enumerate(axes.flatten()):
    #     with sns.axes_style("darkgrid"):
    #         plt.subplot(311)
    #         ax.plot(us, (FB[i]-np.mean(FB[i]))/np.std(FB[i])-3, ls='-', color='red', alpha=0.8, label=r'$f(b=H(z(u)))$')
    #         ax.plot(us, (FZ[i]-np.mean(FZ[i]))/np.std(FZ[i])-1, ls=':', color='blue', alpha=0.8, label=r'$f(\sigma_\lambda(z(u)))$')
    #         ax.plot(us, (QZ[i]-np.mean(QZ[i]))/np.std(QZ[i])+1, ls='-.',  color='green', alpha=0.8, label=r'$Q(\sigma(z(u)))$')
    #         # ax.legend(loc='best')#bbox_to_anchor=(1.0, 0.65))
    #     # ax.xlabel('u')
    # plt.tight_layout()
    # plt.savefig("relaxations_t_{}_which_{}.pdf".format(t, which), format='pdf')
#     #
##    fig1_dict = {}
##    fig1_dict["relax_losses_t0.1"] = relax_losses
##    fig1_dict[]
#    us = np.linspace(0,1,len(FB[0]))
#    for i in range(len(FB)):
#        plt.figure(i)
##        plt.subplot(2, 2, 1)
#        plt.plot(us, FB[i], 'red', label=r'$f(b=H(z(u)))$')
#        plt.plot(us, FZ[i], 'blue', label=r'$f(\sigma_\lambda(z(u)))$')
#        plt.plot(us, QZ[i], 'green', label=r'$Q(\sigma(z(u)))$')
#        plt.legend(bbox_to_anchor=(1.0,0.65))
#        plt.xlabel('u')
#        plt.savefig('/home/damichoi/ml/relaxed-rebar/toy_problem/test'+str(i)+'.png', bbox_inches='tight')
        
#        plt.subplot(2, 2, 2)
#        plt.xlim(0,10000)
#        #plt.ylim(0.2489,0.2503)
#        x = np.arange(0,100*(i+1),100)
#        plt.plot(x, rebar_losses[:i+1], 'blue', label='REBAR')
#        plt.plot(x, relax_losses[:i+1], 'green', label='RELAX')
#        plt.legend(bbox_to_anchor=(1.7,0.65))
#        plt.xlabel('steps')
#        plt.ylabel('loss')
#        plt.tight_layout()
#        plt.subplots_adjust(wspace = 1.2)
        
#        plt.subplot(2, 2, 2)
#        plt.xlim(0,10000)
#        plt.ylim(-0.02,0.7)
#        x = np.arange(0,100*(i+1),100)
#        plt.plot(x, rebar_thetas[:i+1], 'blue', label='REBAR')
#        plt.plot(x, relax_thetas[:i+1], 'green', label='RELAX')
#        plt.legend(bbox_to_anchor=(1.65,0.65))
#        plt.xlabel('steps')
#        plt.ylabel('theta')
#        plt.tight_layout()
#        plt.subplots_adjust(wspace = 1.0)
#        
#        plt.subplot(2, 2, 2)
#        plt.xlim(0,10000)
##        plt.ylim(-19,-3)
#        x = np.arange(0,100*(i+1),100)
#        plt.plot(x, rebar_variances[:i+1], 'blue', label='REBAR')
#        plt.plot(x, relax_variances[:i+1], 'green', label='RELAX')
#        plt.legend(bbox_to_anchor=(1.65,0.65))
#        plt.xlabel('steps')
#        plt.ylabel('log variance')
#        plt.tight_layout()
#        plt.subplots_adjust(wspace = 1.0)
#        plt.savefig('/home/damichoi/ml/relaxed-rebar/toy_problem/test'+str(i)+'.png', bbox_inches='tight')
#        plt.show()
# uncomment once to collect losses for Figs 1 and 2:
#     file_name = "toy_losses_{}_{}".format(ITERS, t)
#     try:
#         with open(file_name+'.pkl', 'r') as f:
#             unloaded = pickle.load(f)
#     except IOError:
#         _,ext_thetas, ext_losses, ext_variances,__,___ = main(t=t, use_reinforce=False, log_var=False, relaxed=False, visualize=None, force_same=True, test_bias=True, use_exact_gradient=True)
#         tf.reset_default_graph()
#         _,reinf_thetas, reinf_losses, reinf_variances,__,___ = main(t=t, use_reinforce=True,  log_var=False, relaxed=False, visualize=None, force_same=True, test_bias=True)
#         tf.reset_default_graph()
#         _,rebar_thetas, rebar_losses, rebar_variances,__,___ = main(t=t, relaxed=False, log_var=False, visualize=None, force_same=True, test_bias=True)
#         tf.reset_default_graph()
#     #    _,rebar_thetas_ttc, rebar_losses_ttc, rebar_variances_ttc,__,___ = main(relaxed=False, visualize=None, force_same=True, test_bias=False, train_to_completion=True)
#     #    tf.reset_default_graph()
#         _,relax_thetas, relax_losses, relax_variances,__,___ = main(t=t, relaxed="super", log_var=False, visualize=None, force_same=True, test_bias=True)
#         tf.reset_default_graph()
#     #    _,relax_thetas_ttc, relax_losses_ttc, relax_variances_ttc,__,___ = main(relaxed="super", visualize=None, force_same=True, test_bias=False, train_to_completion=True)
#     #     tf.reset_default_graph()
#     #     _,lax_thetas, lax_losses, lax_variances,__,___ = main(t=t, relaxed="super", visualize=None, force_same=True, test_bias=False, train_to_completion=False, LAX=True)
#         # tf.reset_default_graph()
#     #    _,lax_thetas_ttc, lax_losses_ttc, lax_variances_ttc,__,___ = main(relaxed="super", visualize=None, force_same=True, test_bias=False, train_to_completion=True, LAX=True)
#     #     tf.reset_default_graph()
#     #    _,bar_thetas, bar_losses, bar_variances,__,___ = main(relaxed=False, visualize=None, force_same=True, test_bias=False, train_to_completion=False, BAR=True)
#     #     tf.reset_default_graph()
#     #    _,bar_thetas_ttc, bar_losses_ttc, bar_variances_ttc,__,___ = main(relaxed=False, visualize=None, force_same=True, test_bias=False, train_to_completion=True, BAR=True)
#
#         with open(file_name+'.pkl', 'w') as f:
#             pickle.dump([ext_losses, reinf_losses, rebar_losses, relax_losses, rebar_variances, reinf_variances, relax_variances, ext_variances], f)
# #
#     # UNCOMMENT HERE FOR FIGURE 1:
#     sns.set_style("white", {"axes.edgecolor": ".7",
#                         "font_scale": "1.2"})
#     sns.set_context("talk")
#     x = np.arange(0, ITERS, RESOLUTION) #len(rebar_losses))
#     ext_losses, reinf_losses, rebar_losses, relax_losses, _ = unloaded
#     print("rebar_losses {}".format(len(rebar_losses)))
#     max_plot_iters = 10000
#     plt.figure(1)
#     plt.xlim(0,max_plot_iters)
#     alpha=1.0
#     sns.set_context("paper")
#     sns.set(font_scale=1.2)
#
#     plt.plot(x, reinf_losses, color=tableau20[0],label="REINFORCE", alpha=alpha)
#     plt.plot(x, rebar_losses,color=tableau20[4], label="REBAR", alpha=alpha)
# #    plt.plot(x, rebar_losses_ttc, 'orange', label="REBAR trained to completion")
#     plt.plot(x, relax_losses, color=tableau20[6],label="RELAX (ours)", alpha=alpha)
#
#     plt.plot(x, ext_losses, color='black', ls='-.', label="Exact gradient", alpha=0.5)
# #    plt.plot(x, relax_losses_ttc, 'purple', label="RELAX trained to completion")
# #     plt.plot(x, lax_losses,color=tableau20[5], label="LAX", alpha=alpha)
#
#     fill_alpha=0.25
#     # plt.fill_between(x, reinf_losses, lax_losses, where=np.array(lax_losses) >= np.array(reinf_losses),
#     #                  color=tableau20[5], alpha=fill_alpha, interpolate=True)
#     # plt.fill_between(x, relax_losses, rebar_losses, facecolor=tableau20[3], alpha=fill_alpha)
#     # # plt.fill_between(x, relax_losses, ext_losses, facecolor=tableau20[2], alpha=fill_alpha)
#     # plt.fill_between(x, reinf_losses, rebar_losses,  where=np.array(reinf_losses) >= np.array(rebar_losses),
#     #                  facecolor=tableau20[1], alpha=fill_alpha)
#
# #    plt.plot(x, lax_losses_ttc, 'black', label="LAX trained to completion")
# #    plt.plot(x, bar_losses, 'pink', label="BAR")
# #    plt.plot(x, bar_losses_ttc, 'yellow', label="BAR trained to completion")
#     plt.legend(loc="best")#bbox_to_anchor=(1.0, 0.75))
#     plt.xlabel("Steps")
#     plt.ylabel(("Loss"))
#     # plt.rc('grid', linestyle="--", color='black')
#     # plt.grid(True)
#     # plt.ylabel("Loss")
#     # plt.xlabel("Iteration")
#     ylims = plt.gca().get_ylim()
#     sns.despine()
#     plt.savefig(os.path.join('toy_problem', file_name +'no_envelope' + '.pdf'), format='pdf', bbox_inches='tight')

# END FIGURE 1

#    plt.figure(2)
#    plt.xlim(0,10000)
#    plt.plot(x, ext_thetas, 'green', label="exact_gradient")
#    plt.plot(x, reinf_thetas, 'magenta', label="REINFORCE")
#    plt.plot(x, rebar_thetas, 'red', label="REBAR")
##    plt.plot(x, rebar_thetas_ttc, 'orange', label="REBAR trained to completion")
#    plt.plot(x, relax_thetas, 'blue', label="RELAX")
#    plt.plot(x, relax_thetas_ttc, 'purple', label="RELAX trained to completion")
#    plt.plot(x, lax_thetas, 'cyan', label="LAX")
#    plt.plot(x, lax_thetas_ttc, 'black', label="LAX trained to completion")
##    plt.plot(x, bar_thetas, 'pink', label="BAR")
##    plt.plot(x, bar_thetas_ttc, 'yellow', label="BAR trained to completion")
#    plt.legend(bbox_to_anchor=(1.0,0.75))
#    plt.rc('grid', linestyle="--", color='black')
#    plt.grid(True)
#    plt.ylabel("theta")
#    plt.xlabel("Steps")
#    plt.savefig('/home/damichoi/ml/relaxed-rebar/theta.png', bbox_inches='tight')
#    
#    plt.figure(3)
#    plt.xlim(0,10000)
##    plt.plot(x, reinf_variances, 'magenta', label="REINFORCE")
#    plt.plot(x, rebar_variances, 'red', label="REBAR")
##    plt.plot(x, rebar_variances_ttc, 'orange', label="REBAR trained to completion")
#    plt.plot(x, relax_variances, 'blue', label="RELAX")
##    plt.plot(x, relax_variances_ttc, 'purple', label="RELAX trained to completion")
#    plt.plot(x, lax_variances, 'cyan', label="LAX")
##    plt.plot(x, lax_variances_ttc, 'black', label="LAX trained to completion")
##    plt.plot(x, bar_variances, 'pink', label="BAR")
##    plt.plot(x, bar_variances_ttc, 'yellow', label="BAR trained to completion")
#    plt.legend(bbox_to_anchor=(1.0,0.75))
#    plt.rc('grid', linestyle="--", color='black')
#    plt.grid(True)
#    plt.ylabel("log(Var(gradient estimator))")
#    plt.xlabel("Steps")
#    plt.savefig('/home/damichoi/ml/relaxed-rebar/variance.png', bbox_inches='tight')


# UNCOMMENT HERE FOR LOG VARIANCE PLOT:
#     _x = np.arange(0, ITERS, RESOLUTION)
#     # _reinf_variances = [reinf_variances[i] for i in range(10000) if (i+1)%10 == 0]
#     # _rebar_variances = [rebar_variances[i] for i in range(ITERS) if i%RESOLUTION == 0]
#     # _relax_variances = [relax_variances[i] for i in range(ITERS) if i%RESOLUTION == 0]
#     # _lax_variances =   [lax_variances[i] for i in range(10000) if (i+1)%10 == 0]
#     # _bar_variances =   [bar_variances[i] for i in range(10000) if (i+1)%10 == 0]
#
#     plt.figure(4)
#     # sns.set(font_scale=1.2)
#     window = 50
#     sns.set_style("white", {"axes.edgecolor": ".7",
#                             "font_scale" : "1.2"})
#     sns.set_context("talk")
#
#     A = pandas.Series(reinf_variances, _x)
#     B = pandas.Series(rebar_variances, _x)
#     C = pandas.Series(relax_variances, _x)
#     point_alpha=0.1
#     line_alpha=0.8
#     plt.plot(_x, pandas.rolling_mean(A, window),  color=tableau20[0], alpha=line_alpha, label="REINFORCE")
#     plt.plot(_x, pandas.rolling_mean(B, window), color=tableau20[4], alpha=line_alpha,  label="REBAR")
#     plt.plot(_x, pandas.rolling_mean(C, window), color=tableau20[6], alpha=line_alpha,  label="RELAX (ours)")
#     plt.legend(loc='best')# bbox_to_anchor=(1.0, 0.75))
#     plt.ylabel("Log Variance of Gradient Estimates")
#     plt.xlabel("Steps")
#     plt.xlim([500, ITERS])
#     sns.despine()
#     plt.savefig(os.path.join('variance_100_{}.pdf'.format(t)), format='pdf', bbox_inches='tight')
