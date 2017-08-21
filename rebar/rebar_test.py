from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit
from autograd import grad, getval
from autograd.util import nd

from rebar import simple_mc_reinforce, simple_mc_concrete,\
    simple_mc_rebar, rebar_variance, simple_mc_simple_rebar,\
    simple_mc_generalized_rebar, init_nn_params

if __name__ == '__main__':
    rs = npr.RandomState(0)
    num_samples = 100000
    D = 3
    params = logit(rs.rand(D))
    targets = rs.randn(D)

    def objective(params, b):
        return (b - targets + expit(params))**2

    def mc(params, estimator=simple_mc_reinforce):
        rs = npr.RandomState(0)
        noise = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        objective_vals = estimator(params_rep, noise, objective)
        return np.mean(objective_vals)

    print("Gradient estimators:")
    print("Numerical          : {}".format(np.array(nd(mc, params))[0]))
    print("Reinforce          : {}".format(grad(mc)(params, simple_mc_reinforce)))
    print("Concrete, temp = 0 : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_concrete(p, 0.01, n, o))))
    print("Concrete, temp = 1 : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_concrete(p, 1.0, n, o))))
    print("Rebar, temp = 0    : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (0.01, 0.3), n, rs.rand(num_samples, D), o))))
    print("Rebar, temp = 1    : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (1.0,  0.3), n, rs.rand(num_samples, D), o))))
    print("Rebar, eta = 0     : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (1.0,  0.0), n, rs.rand(num_samples, D), o))))
    print("Simple Rebar       : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_simple_rebar(p, n, rs.rand(num_samples, D), o))))
    nn_params = init_nn_params(0.1, [D, 5, 1])
    print("Generalized Rebar  : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_generalized_rebar(p, nn_params, n, rs.rand(num_samples, D), o))))

    def rebar_var_naive(est_params):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        grad_vals = grad(simple_mc_rebar)(params_rep, est_params, noise_u, noise_v, objective)
        return np.var(grad_vals)

    def rebar_var_fancy(est_params):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        var_vals = rebar_variance(est_params, params_rep, noise_u, noise_v, objective)
        return np.mean(var_vals)

    print("\n\nGradient of variance:")
    print("Numerical naive    : {}".format(np.array(nd(rebar_var_naive, (1.0,  0.3)))[0]))
    print("Numerical fancy    : {}".format(np.array(nd(rebar_var_fancy, (1.0,  0.3)))[0]))
    print("Autodiff           : {}".format(grad(rebar_var_naive)((1.0,  0.3))))
    #print("Formula from paper : {}".format(grad(rebar_var_fancy)((1.0,  0.3))))
