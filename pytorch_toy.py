from __future__ import absolute_import
from __future__ import print_function

from itertools import product

import argparse
import numpy as np
import torch


class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=10):
        super(QFunc, self).__init__()
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.Tanh()
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, z):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        z = self.h1(z * 2. - 1.)
        z = self.nonlin(z)
        z = self.out(z)
        return z


def loss_func(b, t):
    return ((b - t) ** 2).mean(1)


def _parse_args(args):
    parser = argparse.ArgumentParser(
        description='Toy experiment from backpropagation throught the void, '
        'written in pytorch')
    parser.add_argument(
        '--estimator', choices=['reinforce', 'relax', 'rebar'],
        default='reinforce')
    parser.add_argument('--rand-seed', type=int, default=42)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--target', type=float, default=.499)
    parser.add_argument('--num-latents', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.01)
    return parser.parse_args(args)


def reinforce(f_b, b, logits, **kwargs):
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    d_logits = f_b.unsqueeze(1) * d_log_prob
    return d_logits


def _get_z_tilde(logits, b, v):
    theta = torch.sigmoid(logits)
    v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
    return z_tilde


def rebar(
        f_b, b, logits, z, v, eta, log_temp, target, loss_func=loss_func,
        **kwargs):
    z_tilde = _get_z_tilde(logits, b, v)
    temp = torch.exp(log_temp).unsqueeze(0)
    sig_z = torch.sigmoid(z / temp)
    sig_z_tilde = torch.sigmoid(z_tilde / temp)
    f_z = loss_func(sig_z, target)
    f_z_tilde = loss_func(sig_z_tilde, target)
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    d_f_z = torch.autograd.grad(
        [f_z], [logits], grad_outputs=torch.ones_like(f_z),
        create_graph=True, retain_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(
        [f_z_tilde], [logits], grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True, retain_graph=True)[0]
    diff = f_b.unsqueeze(1) - eta * f_z_tilde.unsqueeze(1)
    d_logits = diff * d_log_prob + eta * (d_f_z - d_f_z_tilde)
    var_loss = (d_logits ** 2).mean()
    var_loss.backward()
    return d_logits.detach()


def relax(f_b, b, logits, z, v, log_temp, q_func, **kwargs):
    z_tilde = _get_z_tilde(logits, b, v)
    temp = torch.exp(log_temp).unsqueeze(0)
    sig_z = torch.sigmoid(z / temp)
    sig_z_tilde = torch.sigmoid(z_tilde / temp)
    f_z = q_func(sig_z)[:, 0]
    f_z_tilde = q_func(sig_z_tilde)[:, 0]
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    d_f_z = torch.autograd.grad(
        [f_z], [logits], grad_outputs=torch.ones_like(f_z),
        create_graph=True, retain_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(
        [f_z_tilde], [logits], grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True, retain_graph=True)[0]
    diff = f_b.unsqueeze(1) - f_z_tilde.unsqueeze(1)
    d_logits = diff * d_log_prob + d_f_z - d_f_z_tilde
    var_loss = (d_logits.mean(0) ** 2).mean()
    var_loss.backward()
    return d_logits.detach()


def run_toy_example(args=None):
    args = _parse_args(args)
    print('Target is {}'.format(args.target))
    target = torch.Tensor(1, args.num_latents)
    target.fill_(args.target)
    logits = torch.zeros(args.num_latents, requires_grad=True)
    eta = torch.ones(args.num_latents, requires_grad=True)
    log_temp = torch.from_numpy(
        np.array([.5] * args.num_latents, dtype=np.float32))
    log_temp.requires_grad_(True)
    q_func = QFunc(args.num_latents)
    torch.manual_seed(args.rand_seed)
    if args.estimator == 'reinforce':
        estimator = reinforce
        tunable = []
    elif args.estimator == 'rebar':
        estimator = rebar
        tunable = [eta, log_temp]
    else:
        estimator = relax
        tunable = [log_temp] + list(q_func.parameters())
    logit_optim = torch.optim.Adam([logits], lr=args.lr)
    if tunable:
        tune_optim = torch.optim.Adam(tunable, lr=args.lr)
    else:
        tune_optim = None
    for i in range(args.iters):
        logit_optim.zero_grad()
        if tune_optim:
            tune_optim.zero_grad()
        u = torch.rand(args.batch_size, args.num_latents)
        v = torch.rand(args.batch_size, args.num_latents)
        z = logits + torch.log(u) - torch.log1p(-u)
        b = z.gt(0.).type_as(z)
        f_b = loss_func(b, target)
        d_logits = estimator(
            f_b=f_b, b=b, u=u, v=v, z=z, target=target, logits=logits,
            log_temp=log_temp, eta=eta, q_func=q_func,
        )
        logits.backward(d_logits.mean(0))  # mean of batch
        d_logits = d_logits.numpy()
        logit_optim.step()
        if tune_optim:
            tune_optim.step()
        thetas = torch.sigmoid(logits.detach()).numpy()
        loss = thetas * (1 - args.target) ** 2
        loss += (1 - thetas) * args.target ** 2
        loss = loss.mean()
        mean = d_logits.mean()
        std = d_logits.std()
        print(
            'Iter: {} Loss: {:.03f} Thetas: {} Mean: {:.03f} Std: {:.03f} '
            'Temp: {:.03f}'.format(
                i, loss, thetas, mean, std, torch.exp(log_temp).item())
        )


if __name__ == '__main__':
    run_toy_example()
