from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit

from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam

from relax import concrete

if __name__ == '__main__':

    D = 100
    rs = npr.RandomState(0)
    num_samples = 50
    init_params = (np.zeros(D), 1.0)

    def objective(b):
        return np.sum((b - np.linspace(0, 1, D))**2, axis=-1, keepdims=True)

    def mc_objective_and_var(combined_params, t):
        params, est_params = combined_params
        params_rep = np.tile(params, (num_samples, 1))
        rs = npr.RandomState(t)
        noise_u = rs.rand(num_samples, D)
        objective_vals, grads = \
            value_and_grad(concrete)(params_rep, est_params, noise_u, objective)
        return np.mean(objective_vals), np.var(grads, axis=0)

    def combined_obj(combined_params, t):
        # Combines objective value and variance of gradients.
        obj_value, grad_variances = mc_objective_and_var(combined_params, t)
        return obj_value

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(411, frameon=False)
    ax2 = fig.add_subplot(412, frameon=False)
    ax3 = fig.add_subplot(413, frameon=False)
    ax4 = fig.add_subplot(414, frameon=False)
    plt.ion()
    plt.show(block=False)

    temperatures = []
    def callback(combined_params, t, combined_grads):
        params, temperature = combined_params
        grad_params, grad_temperature = combined_grads
        temperatures.append(temperature)
        if t % 10 == 0:
            objective_val, grad_vars = mc_objective_and_var(combined_params, t)
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
            ax4.cla()
            ax4.plot(temperatures, 'b')
            ax4.set_ylabel('temperature')

            plt.draw()
            plt.pause(1.0/30.0)

    print("Optimizing...")
    adam(grad(combined_obj), init_params, step_size=0.1, num_iters=2000, callback=callback)
    plt.pause(10.0)