from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit
from autograd import grad
from autograd.util import nd

from rebar import reinforce, concrete, rebar

if __name__ == '__main__':
    rs = npr.RandomState(0)
    num_samples = 100000
    D = 2
    params = logit(rs.rand(D))
    targets = rs.randn(D)

    def objective(params, b):
        return (b - targets + expit(params))**2

    def mc(params, estimator, rs=rs):
        noise = rs.rand(num_samples, D)
        params_rep = np.tile(params, (num_samples, 1))
        objective_vals = estimator(params_rep, noise, objective)
        return np.mean(objective_vals)

    def fixed_seed_objective(params):
        return mc(params, reinforce, rs=npr.RandomState(0))

    print("Numerical          : {}".format(np.array(nd(fixed_seed_objective, params))[0]))
    print("Reinforce          : {}".format(grad(mc)(params, reinforce)))
    print("Concrete, temp = 0 : {}".format(grad(mc)(params, lambda p, n, o: concrete(p, 0.01, n, o))))
    print("Concrete, temp = 1 : {}".format(grad(mc)(params, lambda p, n, o: concrete(p, 1.0, n, o))))
    print("Rebar, temp = 0    : {}".format(grad(mc)(params, lambda p, n, o: rebar(p, 0.01, n, rs.rand(num_samples, D), o))))
    print("Rebar, temp = 1    : {}".format(grad(mc)(params, lambda p, n, o: rebar(p, 1.0,  n, rs.rand(num_samples, D), o))))
