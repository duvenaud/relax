import autograd.numpy as np
from autograd.scipy.special import expit, logit
from autograd import grad, primitive, value_and_grad

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
    sym_targets = targets * 2 - 1  # targets must be 0 or 1
    return -np.logaddexp(0, -logit_theta * sym_targets)


############### REINFORCE ##################

@primitive
def reinforce(params, noise, f):
    # This function simply samples discrete variables and feeds them to f.
    samples = bernoulli_sample(params, noise)
    return f(params, samples)

def reinforce_vjp(g, func_vals, vs, gvs, params, noise, f):
    # We implement the gradient using the reinforce estimator.
    samples = bernoulli_sample(params, noise)
    grad_func_vals = grad(f)(params, samples)
    grad_logprobs  = grad(bernoulli_logprob)(params, samples)
    return g * (grad_func_vals + func_vals * grad_logprobs)
reinforce.defvjp(reinforce_vjp)


############### CONCRETE ###################

def concrete(params, temperature, noise, f):
    # Concrete sampling and evaluation is already differentiable.
    relaxed_samples = relaxed_bernoulli_sample(params, noise, temperature)
    return f(params, relaxed_samples)


############### REBAR ######################

@primitive
def rebar(params, temperature, noise, noise2, f):
    samples = bernoulli_sample(params, noise)
    return f(params, samples)

def rebar_vjp(g, f_vals, vs, gvs, params, temperature, noise, noise2, f):
    samples = bernoulli_sample(params, noise)

    def concrete_cond(params):
        # This closure captures the dependency of the conditional samples on params.
        cond_noise = conditional_noise(params, samples, noise2)
        cond_relaxed_samples = relaxed_bernoulli_sample(params, cond_noise, temperature)
        return f(params, cond_relaxed_samples)

    grad_func = grad(f)(params, samples)                                # d_f(b) / d_theta
    grad_logprobs = grad(bernoulli_logprob)(params, samples)            # d_log_p(b) / d_theta
    grad_concrete = grad(concrete)(params, temperature, noise, f)       # d_f(z) / d_theta
    f_cond, grad_concrete_cond = value_and_grad(concrete_cond)(params)  # d_f(ztilde) / d_theta

    return g * ((f_vals - f_cond) * grad_logprobs + grad_concrete - grad_concrete_cond + grad_func)

rebar.defvjp(rebar_vjp)
