from __future__ import absolute_import
from __future__ import print_function

from pytorch_toy import *


def _parse_args(args):
    parser = argparse.ArgumentParser(
        description='sanity check for pytorch_toy gradient estimators')
    parser.add_argument('--num-mc-samples', type=int, default=10000)
    parser.add_argument('--latent-dim', type=int, default=3)
    parser.add_argument('--param-seed', type=int, default=0)
    parser.add_argument('--mc-seed', type=int, default=0)
    return parser.parse_args(args)


def test(args=None):
    args = _parse_args(args)
    torch.manual_seed(args.param_seed)
    theta = torch.rand(args.latent_dim).clamp(1e-5, 1. - 1e-5)
    logits = torch.log(theta) - torch.log1p(-theta)
    logits.requires_grad_(True)
    q_func = QFunc(args.latent_dim, 5)

    def f(samples):
        return torch.sum(
            (samples - torch.linspace(0.2, 0.9, args.latent_dim)) ** 2,
            1
        )

    def expected_f(logits):
        # population iterates over all possible configurations of D
        # bernoulli samples
        population = torch.stack(
            [
                torch.FloatTensor(b)
                for b in product([0.0, 1.0], repeat=args.latent_dim)
            ],
            dim=0,
        )
        return torch.sum(
            f(population) * torch.prod(
                torch.sigmoid(logits * (population * 2. - 1.)), dim=1))

    def monte_carlo_estimator(logits, estimator, **kwargs):
        torch.manual_seed(args.mc_seed)
        u = torch.rand(args.num_mc_samples, args.latent_dim)
        v = torch.rand(args.num_mc_samples, args.latent_dim)
        # add extra samples to (new) batch index
        logits = logits.unsqueeze(0).expand(
            args.num_mc_samples, args.latent_dim)
        z = logits.detach() + torch.log(u) - torch.log1p(-u)
        z.requires_grad_(True)
        b = z.gt(0.).type_as(z)
        f_b = f(b)
        d_logits = estimator(f_b=f_b, b=b, logits=logits, z=z, v=v, **kwargs)
        return d_logits.mean(0)

    print("Gradient estimators:")
    print("Exact            : {}".format(
        torch.autograd.grad([expected_f(logits)], [logits])[0].numpy()
    ))
    print("Reinforce        : {}".format(
        monte_carlo_estimator(logits, reinforce).numpy()))
    print("Rebar, temp = 1  : {}".format(
        monte_carlo_estimator(
            logits, rebar,
            eta=torch.ones(args.latent_dim) * 0.3,
            log_temp=torch.ones(args.latent_dim).log(),
            target=None,
            loss_func=lambda x, t: f(x)).numpy()))
    print("Rebar, temp = 10 : {}".format(
        monte_carlo_estimator(
            logits, rebar,
            eta=torch.ones(args.latent_dim) * 0.3,
            log_temp=(torch.ones(args.latent_dim) * 10.).log(),
            target=None,
            loss_func=lambda x, t: f(x)).numpy()))
    print("Rebar, eta = 0   : {}".format(
        monte_carlo_estimator(
            logits, rebar,
            eta=torch.zeros(args.latent_dim),
            log_temp=torch.ones(args.latent_dim).log(),
            target=None,
            loss_func=lambda x, t: f(x)).numpy()))
    print("Relax            : {}".format(
        monte_carlo_estimator(
            logits, relax,
            eta=torch.ones(args.latent_dim) * 0.3,
            log_temp=torch.ones(args.latent_dim).log(),
            q_func=q_func).numpy()))


if __name__ == '__main__':
    test()
