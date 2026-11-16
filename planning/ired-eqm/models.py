import einops
import torch.nn as nn
import torch


def swish(x):
    return x * torch.sigmoid(x)


class NLMConv1DBlockV2(nn.Module):
    def __init__(self, inp_dim, cond_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.inp_dim * 6 + self.cond_dim * 2, self.out_dim)

    def forward(self, x, cond):
        # x.shape == [B, T, N, inp_dim]
        # cond.shape == [B, T, N, N, cond_dim]

        B, T, N, _ = x.shape
        x1 = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:]], dim=1)
        x2 = torch.cat([x[:, :-1], torch.zeros_like(x[:, -1:])], dim=1)
        x = torch.cat([x1, x, x2], dim=-1)  # [B, T, N, inp_dim * 3]
        x1 = einops.repeat(x, 'b t n1 c -> b t n1 n2 c', n1=N, n2=N)
        x2 = einops.repeat(x, 'b t n2 c -> b t n1 n2 c', n1=N, n2=N)
        cond1 = einops.repeat(cond, 'b n1 n2 c -> b t n1 n2 c', t=T, n1=N, n2=N)
        cond2 = einops.repeat(cond, 'b n1 n2 c -> b t n2 n1 c', t=T, n1=N, n2=N)
        x = torch.cat([x1, x2, cond1, cond2], dim=-1)  # [B, T, N, N, inp_dim * 6 + cond_dim]
        x = self.linear(x)  # [B, T, N, N, out_dim]
        x = swish(x)
        return x.amax(dim=-2)  # [B, T, N, out_dim]


class GNNConv1DEBMV2(nn.Module):
    """IRED's GNNConv1DEBMV2 with the diffusion-time conditioning removed.

    The per-node, per-step outputs z in R^{T x N} define the energy
    E = 0.5 * ||z||^2, as in the reasoning models (IRED sums z instead).
    """

    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        inp_h = out_dim
        cond_h = inp_dim

        self.block1 = NLMConv1DBlockV2(inp_h, cond_h, 64)
        self.block2 = NLMConv1DBlockV2(64, cond_h, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, inp, out):
        # inp.shape == [B, N, N, inp_dim]
        # out.shape == [B, T, N, out_dim]

        x = self.block1(out, cond=inp)
        x = self.block2(x, cond=inp)
        z = self.fc(x).squeeze(-1)  # [B, T, N]
        energy = 0.5 * torch.sum(z ** 2, dim=[1, 2])  # [B]

        return energy[:, None]


class GNNConv1DV2EBMWrapper(nn.Module):
    """IRED's GNNConv1DV2DiffusionWrapper without the time input."""

    def __init__(self, ebm):
        super().__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

    def forward(self, inp, opt_out, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        energy = self.ebm(inp, opt_out)
        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=self.training)[0]
        if return_both:
            return energy, opt_grad
        else:
            return opt_grad

    def randn(self, batch_size, shape, inp, device):
        n = inp.shape[-2]
        return torch.randn((batch_size, 8, n, shape[0]), device=device)
