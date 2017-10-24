from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit
from autograd import grad, getval
from autograd.util import nd

from rebar import simple_mc_reinforce, concrete,\
    simple_mc_rebar, simple_mc_rebar_grads_var, simple_mc_simple_rebar,\
    simple_mc_generalized_rebar, init_nn_params, obj_rebar_estgrad_var,\
    simple_mc_bar

if __name__ == '__main__':
    rs = npr.RandomState(0)
    num_samples = 10000
    D = 1
    params = logit(rs.rand(D))
    targets = rs.rand(D)

    def objective(params, b):
        #return np.sum((b - targets + expit(params))**2, axis=-1, keepdims=True)
        return np.sum((b - targets)**2, axis=-1, keepdims=True)

    def exact_objective(params):
        return expit(-params) * objective(params, 0) + expit(params) * objective(params, 1)

    def mc(params, estimator=simple_mc_reinforce):
        rs = npr.RandomState(0)
        noise = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        objective_vals = estimator(params_rep, noise, objective)
        return np.mean(objective_vals)

    print("Gradient estimators:")
    print("Exact              : {}".format(grad(exact_objective)(params)))
    print("Reinforce          : {}".format(grad(mc)(params, simple_mc_reinforce)))
    print("Concrete, temp = 1 : {}".format(grad(mc)(params, lambda p, n, o: concrete(p, np.log(1.0), n, o))))
    print("Rebar, temp = 1    : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (np.log(1.0),  np.log(0.3)), n, rs.rand(num_samples, D), o))))
    print("Rebar, temp = 10   : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (np.log(10.0), np.log(0.3)), n, rs.rand(num_samples, D), o))))
    print("Rebar, eta = 0     : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (np.log(1.0),  np.log(0.0)), n, rs.rand(num_samples, D), o))))
    print("bar, temp = 1    : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_bar(p, (np.log(1.0),  np.log(0.3)), n, o))))
    print("bar, temp = 10   : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_bar(p, (np.log(10.0), np.log(0.3)), n, o))))
    print("bar, eta = 0     : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_bar(p, (np.log(1.0),  np.log(0.0)), n, o))))
    print("Simple Rebar       : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_simple_rebar(p, n, rs.rand(num_samples, D), o))))
    nn_params = init_nn_params(0.1, [D, 5, 1])
    print("Generalized Rebar  : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_generalized_rebar(p, (0, 0, 0, nn_params), n, rs.rand(num_samples, D), o))))

    def rebar_var_naive(est_params):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        grad_vals = grad(simple_mc_rebar)(params_rep, est_params, noise_u, noise_v, objective)
        return np.mean(np.var(grad_vals, axis=0))

    def rebar_var_fancy(est_params):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        obj, grads, estgrad, var = obj_rebar_estgrad_var(params_rep, est_params, noise_u, noise_v, objective)
        return np.mean(var)

    def rebar_estgrads_direct(est_params):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        obj, grads, estgrad, var = obj_rebar_estgrad_var(params_rep, est_params, noise_u, noise_v, objective)
        return np.mean(estgrad, axis=0)

    print("\n\nGradient of variance:")
    print("Numerical naive    : {}".format(np.array(nd(rebar_var_naive, (1.0,  0.3)))[0]))
    print("Numerical fancy    : {}".format(np.array(nd(rebar_var_fancy, (1.0,  0.3)))[0]))
    print("Autodiff           : {}".format(grad(rebar_var_naive)((1.0,  0.3))))
    print("Formula from paper : {}".format(grad(rebar_var_fancy)((1.0,  0.3))))
    print("Formula combined   : {}".format(rebar_estgrads_direct((1.0,  0.3))))
    #print("Numerical implicit : {}".format(np.array(nd(rebar_var_implicit, (1.0,  0.3)))[0]))
    #print("Autodiff implicit  : {}".format(grad(rebar_var_implicit)((1.0,  0.3))))
