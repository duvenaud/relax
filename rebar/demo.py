from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit

from autograd import grad
from autograd.optimizers import adam

from rebar import reinforce, concrete, rebar

if __name__ == '__main__':

    D = 100

    rs = npr.RandomState(0)
    num_samples = 500
    init_params = np.zeros(D)

    def objective(params, b):
        return (b - np.linspace(0, 1, D))**2

    def objective_reinforce(params, t):
        params_rep = np.tile(params, (num_samples, 1))
        return objective_tiled_reinforce(params_rep, t)

    def objective_tiled_reinforce(params_rep, t):
        rs=npr.RandomState(t)
        noise = rs.rand(num_samples, D)
        noise2 = rs.rand(num_samples, D)
        #objective_vals = reinforce(params_rep, noise, objective)
        #objective_vals = concrete(params_rep, 0.1, noise, objective)
        objective_vals = rebar(params_rep, 1.01, noise, noise2, objective)

        return np.mean(objective_vals)

    def grad_var_objective_reinforce(params, t):
        params_rep = np.tile(params, (num_samples, 1))
        grads = grad(objective_tiled_reinforce)(params_rep, t)
        return np.sqrt(np.var(grads, axis=0)) * num_samples

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(311, frameon=False)
    ax2 = fig.add_subplot(312, frameon=False)
    ax3 = fig.add_subplot(313, frameon=False)
    plt.ion()
    plt.show(block=False)

    grad_variance_est = np.zeros(D)
    def callback(params, t, gradient):
        if t % 1 == 0:
            print("Iteration {} objective {}".format(t, objective_reinforce(params, t)))
            ax1.cla()
            ax1.plot(expit(params), 'r')
            ax2.cla()
            ax2.plot(gradient, 'g')
            ax3.cla()
            ax3.plot(grad_var_objective_reinforce(params, t), 'b')
            plt.draw()
            plt.pause(1.0/30.0)

    print("Optimizing...")
    adam(grad(objective_reinforce), init_params, step_size=0.1,
         num_iters=2000, callback=callback)
    plt.pause(10.0)