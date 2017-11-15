import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import expit, logit
from autograd import elementwise_grad, value_and_grad, make_vjp


def heaviside(z):
    return z >= 0

def softmax(z, log_temperature):
    temperature = np.exp(log_temperature)
    return expit(z / temperature)

def logistic_sample(noise, mu=0, sigma=1):
    return mu + logit(noise) * sigma

def logistic_logpdf(x, mu=0, scale=1):
    y = (x - mu) / (2 * scale)
    return -2 * np.logaddexp(y, -y) - np.log(scale)

def bernoulli_sample(logit_theta, noise):
    return logit(noise) < logit_theta

def relaxed_bernoulli_sample(logit_theta, noise, log_temperature):
    return softmax(logistic_sample(noise, expit(logit_theta)), log_temperature)

def conditional_noise(logit_theta, samples, noise):
    # Computes p(u|b), where b = H(z), z = logit_theta + logit(noise), p(u) = U(0, 1)
    uprime = expit(-logit_theta)  # u' = 1 - theta
    return samples * (noise * (1 - uprime) + uprime) + (1 - samples) * noise * uprime

def bernoulli_logprob(logit_theta, targets):
    # log Bernoulli(targets | theta), targets are 0 or 1.
    return -np.logaddexp(0, -logit_theta * (targets * 2 - 1))


############### REINFORCE ##################

def reinforce(params, noise, func_vals):
    samples = bernoulli_sample(params, noise)
    return func_vals * elementwise_grad(bernoulli_logprob)(params, samples)


############### CONCRETE ###################

def concrete(params, log_temperature, noise, f):
    relaxed_samples = relaxed_bernoulli_sample(params, noise, log_temperature)
    return f(relaxed_samples)


############### REBAR ######################

def rebar(params, est_params, noise_u, noise_v, f):
    log_temperature, log_eta = est_params
    eta = np.exp(log_eta)
    samples = bernoulli_sample(params, noise_u)

    def concrete_cond(params):
        cond_noise = conditional_noise(params, samples, noise_v)
        return concrete(params, log_temperature, cond_noise, f)

    grad_concrete = elementwise_grad(concrete)(params, log_temperature, noise_u, f)
    f_cond, grad_concrete_cond = value_and_grad(concrete_cond)(params)
    return reinforce(params, noise_u, f(samples) - eta * f_cond) + \
           eta * (grad_concrete - grad_concrete_cond)

def rebar_all(params, est_params, noise_u, noise_v, f):
    # Returns objective, gradients, and gradients of variance of gradients.
    func_vals = f(bernoulli_sample(params, noise_u))
    var_vjp, grads = make_vjp(rebar, argnum=1)(params, est_params, noise_u, noise_v, f)
    d_var_d_est = var_vjp(2 * grads / grads.shape[0])
    return func_vals, grads, d_var_d_est


############### RELAX ######################
# Uses a neural network for control variate instead of original objective

def init_nn_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

relu = lambda x: np.maximum(0, x)

def nn_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs

def relax(params, est_params, noise_u, noise_v, func_vals):
    samples = bernoulli_sample(params, noise_u)
    log_temperature, nn_params = est_params

    def surrogate(relaxed_samples):
        return nn_predict(nn_params, relaxed_samples)

    def surrogate_cond(params):
        cond_noise = conditional_noise(params, samples, noise_v)  # z tilde
        return concrete(params, log_temperature, cond_noise, surrogate)

    grad_surrogate = elementwise_grad(concrete)(params, log_temperature, noise_u, surrogate)
    surrogate_cond, grad_surrogate_cond = value_and_grad(surrogate_cond)(params)
    return reinforce(params, noise_u, func_vals - surrogate_cond) + \
           grad_surrogate - grad_surrogate_cond

def relax_all(params, est_params, noise_u, noise_v, f):
    # Returns objective, gradients, and gradients of variance of gradients.
    func_vals = f(bernoulli_sample(params, noise_u))
    var_vjp, grads = make_vjp(relax, argnum=1)(params, est_params, noise_u, noise_v, func_vals)
    d_var_d_est = var_vjp(2 * grads / grads.shape[0])
    return func_vals, grads, d_var_d_est
