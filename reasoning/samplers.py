import torch
import math

class GD:
    def __init__(self, config):
        self.step_size = config.step_size
    
    def step(self, x, energy_func):
        energy = energy_func(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        x = x - self.step_size * grad

        return x
    
class NAG:
    def __init__(self, config):
        self.step_size = config.step_size
        self.mu = config.mu
        self.v = None

    def step(self, x, energy_func):
        if self.v is None:
            self.v = torch.zeros_like(x)
        
        x_look = (x + self.mu * self.v).detach().requires_grad_(True)
        energy_look = energy_func(x_look)
        grad_look = torch.autograd.grad(energy_look.sum(), x_look)[0]

        self.v = self.mu * self.v - self.step_size * grad_look
        x = x + self.v

        return x

class ULA:
    def __init__(self, config):
        self.step_size = config.step_size

    def step(self, x, energy_func):
        energy = energy_func(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]

        noise = torch.randn_like(x)
        x = x - self.step_size * grad + math.sqrt(2 * self.step_size) * noise
        
        return x

class MALA:
    def __init__(self, config):
        self.step_size = config.step_size

    def step(self, x, energy_func):
        energy = energy_func(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]

        noise = torch.randn_like(x)
        x_new = x - self.step_size * grad + math.sqrt(2 * self.step_size) * noise
        x_new = x_new.detach().requires_grad_(True)
        energy_new = energy_func(x_new)
        grad_new = torch.autograd.grad(energy_new.sum(), x_new)[0]

        forward = ((x_new - (x - self.step_size * grad))**2).sum(dim=list(range(1, x.ndim)))
        backward = ((x - (x_new - self.step_size * grad_new))**2).sum(dim=list(range(1, x.ndim)))

        log_alpha = - energy_new.sum(dim=-1) + energy.sum(dim=-1) \
                    - backward / (4 * self.step_size) + forward / (4 * self.step_size)
        
        accept = (torch.log(torch.rand_like(log_alpha)) < log_alpha).float()

        while accept.ndim < x.ndim:
            accept = accept.unsqueeze(-1)

        x = accept * x_new + (1 - accept) * x

        return x

class UHMC:
    def __init__(self, config):
        self.step_size = config.step_size
        self.num_leapfrog_steps = config.num_leapfrog_steps
        self.damping = config.damping
        self.persistent = config.persistent
        self.p = None

    def step(self, x, energy_func):
        energy = energy_func(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]

        x = x.detach().requires_grad_(True)

        if self.p is None or not self.persistent:
            self.p = torch.randn_like(x)
        else:
            self.p = self.damping * self.p + (1 - self.damping**2)**0.5 * torch.randn_like(x)

        self.p = self.p - 0.5 * self.step_size * grad

        for l in range(self.num_leapfrog_steps):
            x = (x + self.step_size * self.p).detach().requires_grad_(True)
            energy = energy_func(x)
            g = torch.autograd.grad(energy.sum(), x)[0]

            if l < self.num_leapfrog_steps - 1:
                self.p = self.p - self.step_size * g
            else:
                self.p = self.p - 0.5 * self.step_size * g


        return x
    
class HMC:
    def __init__(self, config):
        self.step_size = config.step_size
        self.num_leapfrog_steps = config.num_leapfrog_steps
        self.damping = config.damping
        self.persistent = config.persistent
        self.p = None

    def step(self, x, energy_func):
        energy = energy_func(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]

        if self.p is None or not self.persistent:
            self.p = torch.randn_like(x)
        else:
            self.p = self.damping * self.p + (1 - self.damping**2)**0.5 * torch.randn_like(x)

        H_init = energy.sum(dim=-1) + 0.5 * (self.p**2).sum(dim=list(range(1, self.p.ndim)))
        self.p = self.p - 0.5 * self.step_size * grad
        x_new = x.detach().requires_grad_(True)

        for l in range(self.num_leapfrog_steps):
            x_new = (x_new + self.step_size * self.p).detach().requires_grad_(True)
            energy_new = energy_func(x_new)
            g = torch.autograd.grad(energy_new.sum(), x_new)[0]

            if l < self.num_leapfrog_steps - 1:
                self.p = self.p - self.step_size * g
            else:
                self.p = self.p - 0.5 * self.step_size * g

        H_new = energy_new.sum(dim=-1) + 0.5 * (self.p**2).sum(dim=list(range(1, self.p.ndim)))

        log_alpha = - H_new + H_init
        accept = (torch.log(torch.rand_like(log_alpha)) < log_alpha).float()

        while accept.ndim < x.ndim:
            accept = accept.unsqueeze(-1)

        x = accept * x_new + (1 - accept) * x

        return x

class PEM:
    def __init__(self, config):
        self.config = config
        self.step_size = config.step_size
        self.noise_scale = config.noise_scale

    def step(self, x, energy_func):
        energy = energy_func(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]

        particles = x.shape[0]

        if energy.ndim > 1:
            total_energy = energy.sum(-1)
        else:
            total_energy = energy

        weights = torch.softmax(-total_energy, dim=0)

        indices = torch.multinomial(weights, num_samples=particles, replacement=True)
        x = x[indices]
        grad = grad[indices]

        if self.config.use_ula:
            noise = torch.randn_like(x)
            x = x - self.step_size * grad + math.sqrt(2 * self.step_size) * noise
        else:
            x = x + self.noise_scale * torch.randn_like(x)
            x = x - self.step_size * grad

        return x

sampler_dict = {
    'gd': lambda config: GD(config),
    'nag': lambda config: NAG(config),
    'ula': lambda config: ULA(config),
    'mala': lambda config: MALA(config),
    'uhmc': lambda config: UHMC(config),
    'hmc': lambda config: HMC(config),
    'pem': lambda config: PEM(config)
}
