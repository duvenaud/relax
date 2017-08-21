import autograd.numpy as np
from autograd.scipy.special import expit, logit
from autograd import grad, primitive, value_and_grad, make_jvp

def heaviside(z):
    return z >= 0

def relaxed_heaviside(z, temperature):
    # sigma_lambda in REBAR paper.
    return expit(z / temperature)

def logistic_sample(logit_theta, noise_samples):
    # REBAR's z = g(theta, u)
    return logit_theta + logit(noise_samples)

def bernoulli_sample(logit_theta, noise_samples):
    return heaviside(logistic_sample(logit_theta, noise_samples))

def relaxed_bernoulli_sample(logit_theta, noise_samples, temperature):
    return relaxed_heaviside(logistic_sample(logit_theta, noise_samples), temperature)

def conditional_noise(logit_theta, samples, noise):
    # The output of this function can be fed into logistic_sample to get g tilde.
    uprime = expit(-logit_theta)
    return samples * (noise * (1 - uprime) + uprime) + (1 - samples) * noise * uprime

def bernoulli_logprob(logit_theta, targets):
    # Computes log Bernoulli(targets | theta), targets are 0 or 1.
    return -np.logaddexp(0, -logit_theta * (targets * 2 - 1))


############### REINFORCE ##################

def reinforce_grad(func_vals, params, noise, f):
    samples = bernoulli_sample(params, noise)
    grad_func_vals = grad(f)(params, samples)
    grad_logprobs = grad(bernoulli_logprob)(params, samples)
    return grad_func_vals + func_vals * grad_logprobs

@primitive
def reinforce(params, noise, f):
    # This function simply samples discrete variables and feeds them to f.
    samples = bernoulli_sample(params, noise)
    return f(params, samples)

def reinforce_vjp(g, func_vals, vs, gvs, params, noise, f):
    return g * reinforce_grad(func_vals, params, noise, f)
reinforce.defvjp(reinforce_vjp)


############### CONCRETE ###################

def concrete(params, temperature, noise, f):
    # Concrete sampling and evaluation is already differentiable.
    relaxed_samples = relaxed_bernoulli_sample(params, noise, temperature)
    return f(params, relaxed_samples)


############### REBAR ######################

def rebar_grad(f_vals, model_params, est_params, noise_u, noise_v, f):
    temperature, eta = est_params
    samples = bernoulli_sample(model_params, noise_u)

    def concrete_cond(model_params):
        # Captures the dependency of the conditional samples on model_params.
        cond_noise = conditional_noise(model_params, samples, noise_v)
        return concrete(model_params, temperature, cond_noise, f)

    grad_concrete = grad(concrete)(model_params, temperature, noise_u, f)     # d_f(z) / d_theta
    f_cond, grad_concrete_cond = value_and_grad(concrete_cond)(model_params)  # d_f(ztilde) / d_theta

    return reinforce_grad(f_vals - eta * f_cond, model_params, noise_u, f) \
        + eta * grad_concrete - eta * grad_concrete_cond


@primitive
def rebar(model_params, est_params, noise_u, noise_v, f):
    samples = bernoulli_sample(model_params, noise_u)
    return f(model_params, samples)

def rebar_vjp(g, f_vals, vs, gvs, model_params, est_params, noise_u, noise_v, f):
    return g * rebar_grad(f_vals, model_params, est_params, noise_u, noise_v, f)
rebar.defvjp(rebar_vjp, argnum=0)
rebar.defvjp_is_zero(argnums=(1,))



# This is an attempt to implement the single-sample estimator
# of the gradient of the variance of the gradients from the paper.  It doesn't work yet.
@primitive
def rebar_variance(est_params, model_params, noise_u, noise_v, f):
    rebar_grads = grad(rebar)(model_params, est_params, noise_u, noise_v, f)
    return np.var(rebar_grads, axis=0)

def rebar_variance_vjp(g, variance, vs, gvs, est_params, model_params, noise_u, noise_v, f):
    def rebar_est(est_params):
        f_vals = rebar(model_params, est_params, noise_u, noise_v, f)
        return rebar_grad(f_vals, model_params, est_params, noise_u, noise_v, f)
    rebar_hat = np.mean(rebar_est(est_params))
    return make_jvp(rebar_est)(est_params)(2 * g * rebar_hat)
rebar_variance.defvjp(rebar_variance_vjp)
