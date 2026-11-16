import torch

def eqm_sample(x, lambd, eps=None, gamma=None):
    """Equilibrium Matching corruption and gradient target.

    x_gamma = gamma * x + (1 - gamma) * eps
    target  = lambd * c(gamma) * (eps - x),  with c(gamma) = 1 - gamma
    """
    if eps is None:
        eps = torch.randn_like(x)

    dims = [1] * (len(x.shape) - 1)

    if gamma is None:
        gamma = torch.rand((x.shape[0],), device=x.device, dtype=x.dtype)
    gamma = gamma.reshape(gamma.shape[0], *dims)

    x_gamma = torch.lerp(eps, x, gamma)
    c = lambd * (1 - gamma)
    target = (eps - x) * c

    return x_gamma, target
