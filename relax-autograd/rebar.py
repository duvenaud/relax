import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import expit, logit
from autograd.extend import primitive, defvjp
from autograd import grad, elementwise_grad, value_and_grad, make_vjp


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
    return logit(noise) < logit_theta  # heaviside(logistic_sample(logit_theta, noise))

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

    return reinforce(params, noise_u, controlled_f) \
           + eta * grad_concrete - eta * grad_concrete_cond

def grad_of_var_of_grads(grads):
    # For an unbiased gradient estimator, gives an unbiased
    # single-sample estimate of the gradient of the variance of the gradients.
    return 2 * grads / grads.shape[0]


############### BAR ######################

def bar(params, est_params, noise, f):
    log_temperature, log_eta = est_params
    eta = np.exp(log_eta)
    z = logistic_sample(noise, params)

    def f_relax(params):
        z = logistic_sample(noise, params)
        relaxed_samples = softmax(z, log_temperature)
        return f(params, relaxed_samples)

    f_relaxed_eval, f_relax_grad = value_and_grad(f_relax)(params)

    return reinforce(params, noise, f) \
        - eta * f_relaxed_eval * elementwise_grad(logistic_logpdf, argnum=1)(z, params) \
        + eta * f_relax_grad


# Experimental variant

def rbar(params, est_params, noise, f):
    log_temperature, log_eta = est_params
    eta = np.exp(log_eta)
    z = logistic_sample(noise, params)

    def f_relax(params):
        z = logistic_sample(noise, params)
        relaxed_samples = softmax(z, log_temperature)
        return f(params, relaxed_samples)

    samples = bernoulli_sample(params, noise)
    f_relaxed_eval, f_relax_grad = value_and_grad(f_relax)(params)
    return (f(params, samples) + eta * f_relaxed_eval) * elementwise_grad(logistic_logpdf)(z, params) + eta * f_relax_grad


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

#def func_plus_nn(params, relaxed_samples, nn_scale, nn_params, f):
#    return f(params, relaxed_samples) \
#           + nn_scale * nn_predict(nn_params, relaxed_samples)

def relax(params, est_params, noise_u, noise_v, f):
    samples = bernoulli_sample(params, noise_u)
    log_eta, log_temperature, log_nn_scale, nn_params = est_params
    eta = np.exp(log_eta)
    nn_scale = np.exp(log_nn_scale)

    def f_relaxed(params, relaxed_samples):
        return nn_scale * nn_predict(nn_params, relaxed_samples)

    def concrete_cond(params):
        cond_noise = conditional_noise(params, samples, noise_v)  # z tilde
        return concrete(params, log_temperature, cond_noise, f_relaxed)

    grad_concrete = elementwise_grad(concrete)(params, log_temperature, noise_u, f_relaxed)
    f_cond, grad_concrete_cond = value_and_grad(concrete_cond)(params)
    controlled_f = lambda params, samples: f(params, samples) - eta * f_cond

    return reinforce(params, noise_u, controlled_f) \
           + eta * grad_concrete - eta * grad_concrete_cond




############## Hooks into autograd ##############

###Set up Simple Monte Carlo functions that have different gradient estimators
@primitive
def simple_mc_reinforce(params, noise, f):
    samples = bernoulli_sample(params, noise)
    return f(params, samples)

def reinforce_vjp(ans, *args):
    return lambda g: g * reinforce(*args)
defvjp(simple_mc_reinforce, reinforce_vjp, argnums=[0])


@primitive
def simple_mc_rebar(params, est_params, noise_u, noise_v, f):
    samples = bernoulli_sample(params, noise_u)
    return f(params, samples)

def rebar_vjp(ans, *args):
    return lambda g: g * rebar(*args)
defvjp(simple_mc_rebar, rebar_vjp, None, argnums=[0, 1])
#simple_mc_rebar.defvjp_is_zero(argnums=(1,))

@primitive
def simple_mc_bar(params, est_params, noise_u, f):
    samples = bernoulli_sample(params, noise_u)
    return f(params, samples)

def bar_vjp(ans, *args):
    return lambda g: g * bar(*args)
defvjp(simple_mc_bar, bar_vjp, argnums=[0])


@primitive
def simple_mc_rbar(params, est_params, noise_u, f):
    samples = bernoulli_sample(params, noise_u)
    return f(params, samples)

def rbar_vjp(ans, *args):
    return lambda g: g * rbar(*args)
defvjp(simple_mc_rbar, rbar_vjp, argnum=0)


def rebar_v_vjp(ans, *args):
    # Unbiased estimator of grad of variance of rebar.
    grads = rebar(*args)
    grad_est = grad_of_var_of_grads(grads)
    est_params_vjp, _ = make_vjp(rebar, argnum=1)(*args)
    return est_params_vjp(grad_est)

#defvjp(simple_mc_rebar, rebar_v_vjp, argnum=1)

def obj_rebar_estgrad_var(params, est_params, noise_u, noise_v, f):
    # To avoid recomputing things, here's a function that computes everything together
    rebar_list = []
    def value_and_rebar(est_params):
        o, r = value_and_grad(simple_mc_rebar)(params, est_params, noise_u, noise_v, f)
        rebar_list.append(r)
        return r


    obj, grads = rebar_list[0]
    vargrad = make_vjp(value_and_rebar)(est_params)(grad_of_var_of_grads(grads))

    var = np.var(grads, axis=0)
    return obj, grads, vargrad, var



def rebar_var_vjp(ans, *args):
    # Unbiased estimator of grad of variance of rebar.
    _, grads, _ = ans
    _, _, var_g = g
    grad_est = grad_of_var_of_grads(grads)
    est_params_vjp, _ = make_vjp(rebar, argnum=1)(*args)
    return est_params_vjp(grad_est)

    def double_val_fun(*args):
        val = fun(*args)
        return make_tuple(val, unbox_if_possible(val))
    gradval_and_val = grad_and_aux(double_val_fun, argnum)
    flip = lambda x, y: make_tuple(y, x)
    return lambda *args: flip(*gradval_and_val(*args))

# This wrapper lets us implement the single-sample estimator
# of the gradient of the variance of the gradient estimate from the paper.
@primitive
def simple_mc_rebar_grads_var(*args):
    # Returns estimates of objective, gradients, and variance of gradients.
    obj, grads = value_and_grad(simple_mc_rebar)(*args)
    return obj, grads, np.var(grads, axis=0)

def rebar_obj_vjp((obj, grads, var), *args):
    return lambda (obj_g, rebar_g, var_g): obj_g * grads

def rebar_var_vjp(ans, *args):
    # Unbiased estimator of grad of variance of rebar.
    _, grads, _ = ans
    def inner_grad((_1, _2, var_g)):
        grad_est = 2 * var_g * grads / grads.shape[0]  # Formula from paper
        est_params_vjp, _ = make_vjp(rebar, argnum=1)(*args)
        return est_params_vjp(grad_est)
    return inner_grad
#simple_mc_rebar_grads_var.defvjp(rebar_obj_vjp, argnum=0)
#simple_mc_rebar_grads_var.defvjp(rebar_var_vjp, argnum=1)


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

def relax_var_vjp(g, ans, *args):  # Unbiased estimator of grad of variance of rebar.
    _, grads, _ = ans
    def inner_grad((_1, _2, var_g)):
        est_params_vjp, _ = make_vjp(relax, argnum=1)(*args)
        return est_params_vjp(grad_of_var_of_grads(grads))
    return inner_grad
defvjp(relax_grads_var, rebar_obj_vjp, relax_var_vjp, argnums=[0, 1])
