from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit
from autograd.misc.optimizers import adam

from relax import init_nn_params, nn_predict, relax_all

def make_one_d(f, d, full_d_input):
    def oned(one_d_input):
        c = full_d_input.copy()
        c[d] = one_d_input
        return f(c)
    return oned

def map_and_stack(f):
    def mapped(inputs):
        return np.stack([f(a) for a in inputs])
    return mapped

if __name__ == '__main__':

    D = 100
    slice_dim = D // 2 - 1
    num_hidden_units = 5
    rs = npr.RandomState(0)
    num_samples = 10
    init_est_params = (0.0, init_nn_params(0.1, [D, num_hidden_units, 1]))
    init_model_params = np.zeros(D)
    init_combined_params = (init_model_params, init_est_params)

    def objective(b):
        return np.sum((b - np.linspace(0, 1, D))**2, axis=-1, keepdims=True)

    def mc_objective_and_var(combined_params, t):
        params, est_params = combined_params
        params_rep = np.tile(params, (num_samples, 1))
        rs = npr.RandomState(t)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        return relax_all(params_rep, est_params, noise_u, noise_v, objective)

    def combined_grad(combined_params, t):
        obj_value, grad_obj, grad_var = mc_objective_and_var(combined_params, t)
        return (np.mean(grad_obj, axis=0), grad_var)

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(511, frameon=False)
    ax2 = fig.add_subplot(512, frameon=False)
    ax3 = fig.add_subplot(513, frameon=False)
    ax4 = fig.add_subplot(514, frameon=False)
    ax5 = fig.add_subplot(515, frameon=False)

    plt.ion()
    plt.show(block=False)

    temperatures = []
    nn_scales = []
    def callback(combined_params, t, combined_gradient):
        params, est_params = combined_params
        grad_params, grad_est = combined_gradient
        log_temperature, nn_params = est_params
        temperatures.append(np.exp(log_temperature))
        if t % 10 == 0:
            objective_val, grads, est_grads = mc_objective_and_var(combined_params, t)
            print("Iteration {} objective {}".format(t, np.mean(objective_val)))
            ax1.cla()
            ax1.plot(expit(params), 'r')
            ax1.set_ylabel('parameter values')
            ax1.set_xlabel('parameter index')
            ax1.set_ylim([0, 1])
            ax2.cla()
            ax2.plot(grad_params, 'g')
            ax2.set_ylabel('average gradient')
            ax2.set_xlabel('parameter index')
            ax3.cla()
            ax3.plot(np.var(grads), 'b')
            ax3.set_ylabel('gradient variance')
            ax3.set_xlabel('parameter index')
            ax4.cla()
            ax4.plot(temperatures, 'b')
            ax4.set_ylabel('temperature')
            ax4.set_xlabel('iteration')

            ax5.cla()
            xrange = np.linspace(0, 1, 200)
            f_tilde = lambda x: nn_predict(nn_params, x)
            f_tilde_map = map_and_stack(make_one_d(f_tilde, slice_dim, params))
            ax5.plot(xrange, f_tilde_map(logit(xrange)), 'b')
            ax5.set_ylabel('1d slide of surrogate')
            ax5.set_xlabel('relaxed sample')
            plt.draw()
            plt.pause(1.0/30.0)

    print("Optimizing...")
    adam(combined_grad, init_combined_params, step_size=0.1, num_iters=2000, callback=callback)
    plt.pause(10.0)
