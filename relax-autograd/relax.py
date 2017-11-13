import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import expit, logit
from autograd.extend import primitive, defvjp
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

def reinforce(params, noise, f):
    samples = bernoulli_sample(params, noise)
    func_vals, grad_func_vals = value_and_grad(f)(params, samples)
    return grad_func_vals + func_vals * elementwise_grad(bernoulli_logprob)(params, samples)


############### CONCRETE ###################

def concrete(params, log_temperature, noise, f):
    relaxed_samples = relaxed_bernoulli_sample(params, noise, log_temperature)
    return f(params, relaxed_samples)


############### REBAR ######################

def rebar(params, est_params, noise_u, noise_v, f):
    log_temperature, log_eta = est_params
    eta = np.exp(log_eta)
    samples = bernoulli_sample(params, noise_u)

    def concrete_cond(params):
        cond_noise = conditional_noise(params, samples, noise_v)  # z tilde
        return concrete(params, log_temperature, cond_noise, f)

    grad_concrete = elementwise_grad(concrete)(params, log_temperature, noise_u, f)  # d_f(z) / d_theta
    f_cond, grad_concrete_cond = value_and_grad(concrete_cond)(params)  # d_f(ztilde) / d_theta
    controlled_f = lambda params, samples: f(params, samples) - eta * f_cond

    return reinforce(params, noise_u, controlled_f) + eta * (grad_concrete - grad_concrete_cond)

def grad_of_var_of_grads(grads):
    # For an unbiased gradient estimator, gives an unbiased
    # single-sample estimate of the gradient of the variance of the gradients.
    return 2 * grads / grads.shape[0]



############### RELAX ######################
# Uses a neural network for control variate instead of original objective

def init_nn_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs


def relax(params, est_params, noise_u, noise_v, f):
    samples = bernoulli_sample(params, noise_u)
    log_eta, log_temperature, nn_params = est_params

    def surrogate(params, relaxed_samples):
        return np.exp(log_eta) * nn_predict(nn_params, relaxed_samples)

    def surrogate_cond(params):
        cond_noise = conditional_noise(params, samples, noise_v)  # z tilde
        return concrete(params, log_temperature, cond_noise, surrogate)

    grad_surrogate = elementwise_grad(concrete)(params, log_temperature, noise_u, surrogate)
    surrogate_cond, grad_surrogate_cond = value_and_grad(surrogate_cond)(params)
    controlled_f = lambda params, samples: f(params, samples) - surrogate_cond

    return reinforce(params, noise_u, controlled_f) + grad_surrogate - grad_surrogate_cond




############## Hooks into autograd ##############

###Set up Simple Monte Carlo functions that have different gradient estimators
@primitive
def simple_mc_reinforce(params, noise, f):
    samples = bernoulli_sample(params, noise)
    return f(params, samples)

def reinforce_vjp(ans, *args):
    return lambda g: g * reinforce(*args)
defvjp(simple_mc_reinforce, reinforce_vjp, argnums=[0])


######## REBAR hooks ############

@primitive
def simple_mc_rebar(params, est_params, noise_u, noise_v, f):
    samples = bernoulli_sample(params, noise_u)
    return f(params, samples)

def rebar_vjp(ans, *args):
    return lambda g: g * rebar(*args)
defvjp(simple_mc_rebar, rebar_vjp, None, argnums=[0, 1])


@primitive
def rebar_grads_var(*args):
    # Returns estimates of objective, gradients, and variance of gradients.
    obj, grads = value_and_grad(simple_mc_rebar)(*args)
    return obj, grads, np.var(grads, axis=0)

def rebar_var_vjp(ans, *args):  # Unbiased estimator of grad of variance of rebar.
    obj, grads, var = ans
    def inner_grad((_1, _2, var_g)):
        est_params_vjp, _ = make_vjp(rebar, argnum=1)(*args)
        return est_params_vjp(var_g * grad_of_var_of_grads(grads))
    return inner_grad

def rebar_obj_vjp((obj, grads, var), *args):
    return lambda (obj_g, rebar_g, var_g): obj_g * grads
defvjp(rebar_grads_var, rebar_obj_vjp, rebar_var_vjp, argnums=[0, 1])


######### RELAX hooks ##########

@primitive
def simple_mc_relax(params, est_params, noise_u, noise_v, f):
    samples = bernoulli_sample(params, noise_u)
    return f(params, samples)

def relax_vjp(ans, *args):
    return lambda g: g * relax(*args)
defvjp(simple_mc_relax, relax_vjp, None, argnums=[0, 1])


@primitive
def relax_grads_var(*args):
    # Returns estimates of objective, gradients, and variance of gradients.
    obj, grads = value_and_grad(simple_mc_relax)(*args)
    return obj, grads, np.var(grads, axis=0)

def relax_var_vjp(ans, *args):  # Unbiased estimator of grad of variance of rebar.
    obj, grads, var = ans
    def inner_grad((_1, _2, var_g)):
        est_params_vjp, _ = make_vjp(relax, argnum=1)(*args)
        return est_params_vjp(var_g * grad_of_var_of_grads(grads))
    return inner_grad

def relax_obj_vjp((obj, grads, var), *args):
    return lambda (obj_g, _1, _2): obj_g * grads
defvjp(relax_grads_var, relax_obj_vjp, relax_var_vjp, argnums=[0, 1])
