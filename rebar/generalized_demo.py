from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit

from autograd import grad, value_and_grad
from autograd.optimizers import adam

from rebar import simple_mc_generalized_rebar, init_nn_params, func_plus_nn

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
    num_hidden_units = 5
    rs = npr.RandomState(0)
    num_samples = 50
    init_est_params = (1.0, 1.0, -1.0, init_nn_params(0.1, [D, num_hidden_units, 1]))
    init_model_params = np.zeros(D)
    init_combined_params = (init_model_params, init_est_params)

    def objective(params, b):
        return np.sum((b - np.linspace(0, 1, D))**2, axis=-1, keepdims=True)

    def mc_objective_and_var(combined_params, t):
        params, est_params = combined_params
        params_rep = np.tile(params, (num_samples, 1))
        rs = npr.RandomState(t)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        objective_vals, grads = \
            value_and_grad(simple_mc_generalized_rebar)(params_rep, est_params, noise_u, noise_v, objective)
        return np.mean(objective_vals), np.var(grads, axis=0)

    def combined_obj(combined_params, t):
        # Combines objective value and variance of gradients.
        # However, model_params shouldn't affect variance (in expectation),
        # and est_params shouldn't affect objective (in expectation).
        obj_value, grad_variances = mc_objective_and_var(combined_params, t)
        return obj_value + grad_variances

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(611, frameon=False)
    ax2 = fig.add_subplot(612, frameon=False)
    ax3 = fig.add_subplot(613, frameon=False)
    ax4 = fig.add_subplot(614, frameon=False)
    ax5 = fig.add_subplot(615, frameon=False)
    ax6 = fig.add_subplot(616, frameon=False)

    plt.ion()
    plt.show(block=False)

    etas = []
    temperatures = []
    nn_scales = []
    def callback(combined_params, t, gradient):
        params, est_params = combined_params
        log_eta, log_temperature, log_nn_scale, nn_params = est_params
        etas.append(np.exp(log_eta))
        temperatures.append(np.exp(log_temperature))
        nn_scales.append(np.exp(log_nn_scale))
        grad_params = gradient[:D]
        if t % 10 == 0:
            objective_val, grad_vars = mc_objective_and_var(combined_params, t)
            print("Iteration {} objective {}".format(t, objective_val))
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
            ax3.plot(grad_vars, 'b')
            ax3.set_ylabel('gradient variance')
            ax3.set_xlabel('parameter index')
            ax4.cla()
            ax4.plot(temperatures, 'b')
            ax4.set_ylabel('temperature')
            ax5.cla()
            ax5.plot(etas, 'b')
            ax5.set_ylabel('eta')
            ax5.set_xlabel('iteration')
            ax6.cla()
            xrange = np.linspace(0, 1, 200)
            slide_d = D / 2 - 1
            f = lambda b: objective(params, b)
            f_tilde = lambda x: func_plus_nn(params, x, np.exp(log_nn_scale), nn_params, objective)
            f_map       = map_and_stack(make_one_d(f,       slide_d, params))
            f_tilde_map = map_and_stack(make_one_d(f_tilde, slide_d, params))
            ax6.plot(xrange, f_map(xrange) - np.mean(f_map(xrange)),       'g')
            ax6.plot(xrange, f_tilde_map(xrange) - np.mean(f_tilde_map(xrange)), 'b')
            ax6.set_ylabel('mean of f tilde 1d')
            ax6.set_xlabel('relaxed sample')
            plt.draw()
            plt.pause(1.0/30.0)

    print("Optimizing...")
    adam(grad(combined_obj), init_combined_params, step_size=0.1, num_iters=2000, callback=callback)
    plt.pause(10.0)