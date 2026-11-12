import torch.nn as nn
import torch

class EBM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.enable_grad()
    def forward(self, x, y=None):
        x = x.detach().requires_grad_(True)
        energy = self.model(x, y)

        grad = torch.autograd.grad(energy.sum(), x, create_graph=self.training)[0]

        return grad, energy