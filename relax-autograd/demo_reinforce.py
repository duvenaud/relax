from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit

from autograd import grad
from autograd.misc.optimizers import adam

from relax import bernoulli_sample, reinforce

if __name__ == '__main__':

    D = 100
    rs = npr.RandomState(0)
    num_samples = 50
    init_params = np.zeros(D)

    def objective(b):
        return np.sum((b - np.linspace(0, 1, D))**2, axis=-1, keepdims=True)

    def mc_objective_and_var(params, t):
        params_rep = np.tile(params, (num_samples, 1))
        rs = npr.RandomState(t)
        noise_u = rs.rand(num_samples, D)
        samples = bernoulli_sample(params_rep, noise_u)
        objective_vals = objective(samples)
        grads = reinforce(params_rep, noise_u, objective_vals)
        return np.mean(objective_vals), np.mean(grads, axis=0), np.var(grads, axis=0)

    def obj_grads(params, t):
        obj_value, grads, grad_variances = mc_objective_and_var(params, t)
        return grads

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(311, frameon=False)
    ax2 = fig.add_subplot(312, frameon=False)
    ax3 = fig.add_subplot(313, frameon=False)
    plt.ion()
    plt.show(block=False)

    temperatures = []
    def callback(params, t, gradient):
        grad_params = gradient[:D]
        if t % 10 == 0:
            objective_val, grads, grad_vars = mc_objective_and_var(params, t)
            print("Iteration {} objective {}".format(t, objective_val))
            ax1.cla()
            ax1.plot(expit(params), 'r')
            ax1.set_ylabel('parameter values')
            ax1.set_ylim([0, 1])
            ax2.cla()
            ax2.plot(grad_params, 'g')
            ax2.set_ylabel('average gradient')
            ax3.cla()
            ax3.plot(grad_vars, 'b')
            ax3.set_ylabel('gradient variance')
            ax3.set_xlabel('parameter index')

            plt.draw()
            plt.pause(1.0/30.0)

    print("Optimizing...")
    adam(obj_grads, init_params, step_size=0.1, num_iters=2000, callback=callback)
    plt.pause(10.0)