from __future__ import absolute_import
from __future__ import print_function
import itertools

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit
from autograd import grad, elementwise_grad

from relax import simple_mc_reinforce, concrete,\
    simple_mc_relax, relax_all,\
    simple_mc_rebar, init_nn_params, rebar_all


if __name__ == '__main__':
    rs = npr.RandomState(0)
    num_samples = 10000
    D = 3
    params = logit(rs.rand(D))

    def objective(params, b):
        return np.sum((b - np.linspace(0.2, 0.9, D))**2, axis=-1, keepdims=True)

    def expected_objective(params):
        lst = list(itertools.product([0.0, 1.0], repeat=D))
        return sum([objective(params, np.array(b)) \
                    * np.prod([expit(params[i] * (b[i] * 2.0 - 1.0))
                               for i in range(D)]) for b in lst])

    def mc(params, estimator):  # Simple Monte Carlo
        print("Calling MC")
        rs = npr.RandomState(0)
        noise = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        objective_vals = estimator(params_rep, noise, objective)
        return np.mean(objective_vals)

    print("Gradient estimators:")
    print("Exact              : {}".format(grad(expected_objective)(params)))
    print("Reinforce          : {}".format(grad(mc)(params, simple_mc_reinforce)))
    print("Concrete, temp = 1 : {}".format(grad(mc)(params, lambda p, n, o: concrete(p, np.log(1), n, o))))
    print("Rebar, temp = 1    : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (np.log(1.0),  np.log(0.3)), n, rs.rand(num_samples, D), o))))
    print("Rebar, temp = 10   : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (np.log(10.0), np.log(0.3)), n, rs.rand(num_samples, D), o))))
    print("Rebar, eta = 0     : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_rebar(p, (np.log(1.0),  np.log(0.0001)), n, rs.rand(num_samples, D), o))))
    nn_params = init_nn_params(0.1, [D, 5, 1])
    print("Relax              : {}".format(grad(mc)(params, lambda p, n, o: simple_mc_relax(p, (0, 0, nn_params), n, rs.rand(num_samples, D), o))))

    def var_naive(est_params, method):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        grad_vals = elementwise_grad(method)(params_rep, est_params, noise_u, noise_v, objective)
        return np.mean(np.var(grad_vals, axis=0))

    def var_grads(est_params, method):
        rs = npr.RandomState(0)
        noise_u = rs.rand(num_samples, D)
        noise_v = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        obj, grads, vargrads = method(params_rep, est_params, noise_u, noise_v, objective)
        return vargrads

    print("\n\nGradient of variance of REBAR gradient:")
    print("Autodiff  : {}".format(grad(var_naive)((1.0,  0.3), simple_mc_rebar)))
    print("Direct    : {}".format(var_grads((1.0,  0.3), rebar_all)))

    print("\n\nGradient of variance of RELAX gradient:")
    print("Autodiff  : {}".format(grad(var_naive)((0.0, 0.0, nn_params), simple_mc_relax)))
    print("Direct    : {}".format(var_grads((0.0, 0.0, nn_params), relax_all)))
