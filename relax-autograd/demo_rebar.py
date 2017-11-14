from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit
from autograd.misc.optimizers import adam

from relax import rebar_all

if __name__ == '__main__':

    D = 100
    rs = npr.RandomState(0)
    num_samples = 10
    init_params = (np.zeros(D), (1.0, 1.0))

    def objective(b):
        return np.sum((b - np.linspace(0, 1, D))**2, axis=-1, keepdims=True)

    def mc_objective_and_var(combined_params, t):
        params, est_params = combined_params
        params_rep = np.tile(params, (num_samples, 1))
        rs = npr.RandomState(t)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        return rebar_all(params_rep, est_params, noise_u, noise_v, objective)

    def combined_grad(combined_params, t):
        obj_value, grad_obj, grad_var = mc_objective_and_var(combined_params, t)
        return (np.mean(grad_obj, axis=0), grad_var)

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(411, frameon=False)
    ax2 = fig.add_subplot(412, frameon=False)
    ax3 = fig.add_subplot(413, frameon=False)
    ax4 = fig.add_subplot(414, frameon=False)
    plt.ion()
    plt.show(block=False)

    temperatures = []
    etas = []
    def callback(combined_params, t, combined_gradient):
        params, est_params = combined_params
        grad_params, grad_est = combined_gradient
        log_temperature, log_eta = est_params
        temperatures.append(np.exp(log_temperature))
        etas.append(np.exp(log_eta))
        if t % 10 == 0:
            objective_val, grads, est_grads = mc_objective_and_var(combined_params, t)
            print("Iteration {} objective {}".format(t, np.mean(objective_val)))
            ax1.cla()
            ax1.plot(expit(params), 'r')
            ax1.set_ylabel('parameter values')
            ax1.set_ylim([0, 1])
            ax2.cla()
            ax2.plot(grad_params, 'g')
            ax2.set_ylabel('average gradient')
            ax3.cla()
            ax3.plot(temperatures, 'b')
            ax3.set_ylabel('temperature')
            ax4.cla()
            ax4.plot(etas, 'b')
            ax4.set_ylabel('eta')
            ax4.set_xlabel('iteration')

            plt.draw()
            plt.pause(1.0/30.0)

    print("Optimizing...")
    adam(combined_grad, init_params, step_size=0.1, num_iters=2000, callback=callback)
    plt.pause(10.0)