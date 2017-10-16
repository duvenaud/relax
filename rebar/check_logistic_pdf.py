from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
import numpy.random as npr
from scipy.special import expit, logit


def logistic_sample(logit_theta, noise):  # REBAR's z = g(theta, u)
    return logit_theta + logit(noise)

def logistic_logpdf(x, logit_mu=0, scale=1):
    mu = expit(logit_mu)
    y = (x - mu) / (2 * scale)
    return 2 * np.logaddexp(-y, y) + np.log(scale)

if __name__ == '__main__':

    rs = npr.RandomState(0)
    num_samples = 50000

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')

    samples = logistic_sample(1.1, rs.rand(num_samples))

    plt.hist(samples)

    plt.plot(np.exp(logistic_logpf(params, 1.1, 1.0)), 'r')
    plt.pause(10.0)