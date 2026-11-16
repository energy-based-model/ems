"""EqM version of IRED's diffusion_lib/denoising_diffusion_pytorch_1d.py.

EquilibriumMatching1D replaces GaussianDiffusion1D: one time-free energy
landscape trained with Equilibrium Matching instead of ten annealed
diffusion landscapes. Trainer1D and the shortest-path metrics follow IRED.
"""

import math
import collections
from multiprocessing import cpu_count
from pathlib import Path

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
from tqdm.auto import tqdm

import os.path as osp
import time

from eqm import eqm_sample


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EquilibriumMatching1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        lambd = 8.0,
        supervise_energy_landscape = True,
        contrastive_scale = 0.5,
        sampler = 'gd',
        sampling_steps = 55,
        step_size = 0.05,
    ):
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )

        self.lambd = lambd
        self.supervise_energy_landscape = supervise_energy_landscape
        self.contrastive_scale = contrastive_scale

        assert sampler in {'gd', 'ula'}
        self.sampler = sampler
        self.sampling_steps = sampling_steps
        self.step_size = step_size

    @torch.no_grad()
    def sample(self, inp, batch_size):
        img = self.model.randn(batch_size, self.out_shape, inp, inp.device)

        with torch.enable_grad():
            for _ in range(self.sampling_steps):
                img = img.detach().requires_grad_(True)
                energy, grad = self.model(inp, img, return_both=True)
                img_new = img - self.step_size * grad

                if self.sampler == 'gd':
                    # As in IRED's opt_step: keep the old sample where the step raised the energy.
                    energy_new = self.model(inp, img_new.detach(), return_energy=True)
                    bad_step = (energy_new > energy)[:, 0]
                    img_new[bad_step] = img[bad_step]
                else:
                    img_new = img_new + math.sqrt(2 * self.step_size) * torch.randn_like(img_new)

                img = img_new.detach()

        return img

    def p_losses(self, inp, x_start):
        noise = torch.randn_like(x_start)
        gamma = torch.rand(x_start.shape[0], device=x_start.device)

        x, target = eqm_sample(x_start, self.lambd, noise, gamma)
        energy_real, model_out = self.model(inp, x, return_both=True)

        loss = F.mse_loss(model_out, target, reduction = 'none')

        # IRED's shortest-path weighting: up-weight the path entries.
        mask1 = (x_start > 0)
        mask2 = torch.logical_not(mask1)
        weight = mask1 * 10 + mask2 * 0.5
        loss = loss * weight

        loss_mse = loss.mean()

        if self.supervise_energy_landscape:
            # IRED's shortest-path negative: the ground-truth path with its
            # first node replaced by a random node.
            x_start_list = x_start.argmax(dim=2)
            classes = x_start.size(2)
            rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)

            x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
            x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
            xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2

            # Noise the negative with the same noise and gamma as the positive.
            xmin_noise, _ = eqm_sample(xmin_noise_rescale, self.lambd, noise, gamma)
            energy_fake = self.model(inp, xmin_noise, return_energy=True)

            # Compute noise contrastive energy loss
            energy_stack = torch.cat([energy_real, energy_fake], dim=-1)
            target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
            loss_energy = F.cross_entropy(-1 * energy_stack, target.long())

            loss = loss_mse + self.contrastive_scale * loss_energy
            return loss, (loss_mse, loss_energy)
        else:
            return loss_mse, (loss_mse, torch.tensor(-1.0))

    def forward(self, inp, target):
        return self.p_losses(inp, target)


class Trainer1D(object):
    def __init__(
        self,
        eqm_model: EquilibriumMatching1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        validation_batch_size = None,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        data_workers = None,
        results_folder = './results',
        validation_dataset = None,
        extra_validation_datasets = None,
        extra_validation_every_mul = 10,
        evaluate_first = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(split_batches = True)

        self.model = eqm_model

        self.save_and_sample_every = save_and_sample_every
        self.extra_validation_every_mul = extra_validation_every_mul

        self.batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size if validation_batch_size is not None else train_batch_size

        self.train_num_steps = train_num_steps

        self.data_workers = data_workers

        if self.data_workers is None:
            self.data_workers = cpu_count()

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        if validation_dataset is not None:
            dl = DataLoader(validation_dataset, batch_size = self.validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
            self.validation_dl = self.accelerator.prepare(dl)
        else:
            self.validation_dl = None

        self.extra_validation_dls = dict()
        for key, extra_dataset in (extra_validation_datasets or dict()).items():
            dl = DataLoader(extra_dataset, batch_size = self.validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
            self.extra_validation_dls[key] = self.accelerator.prepare(dl)

        # optimizer

        self.opt = Adam(eqm_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(eqm_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.evaluate_first = evaluate_first

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if osp.isfile(milestone):
            milestone_file = milestone
        else:
            milestone_file = str(self.results_folder / f'model-{milestone}.pt')
        data = torch.load(milestone_file, map_location = self.device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.evaluate_first:
            milestone = self.step // self.save_and_sample_every
            self.evaluate(device, milestone)
            self.evaluate_first = False  # later used as a signal for whether to run extra validation.

        end_time = time.time()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:

            while self.step < self.train_num_steps:

                inp, label = next(self.dl)
                inp, label = inp.float().to(device), label.float().to(device)

                data_time = time.time() - end_time; end_time = time.time()

                loss, (loss_mse, loss_energy) = self.model(inp, label)

                self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                pbar.set_description(f'loss: {loss.item():.4f} eqm: {loss_mse.item():.4f} cl: {loss_energy.item():.4f} data: {data_time:.2f} nn: {nn_time:.2f}')

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        self.save(milestone)
                        self.evaluate(device, milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def evaluate(self, device, milestone):
        print('Running Evaluation...', flush=True)
        self.ema.ema_model.eval()

        if self.validation_dl is not None:
            self._run_validation(self.validation_dl, device, milestone, prefix = 'Validation')

        if self.step % (self.save_and_sample_every * self.extra_validation_every_mul) == 0 or self.evaluate_first:
            for key, extra_dl in self.extra_validation_dls.items():
                self._run_validation(extra_dl, device, milestone, prefix = key)

    def _run_validation(self, dl, device, milestone, prefix='Validation'):
        # As in IRED, only the first 22 batches of each set are scored.
        meters = collections.defaultdict(AverageMeter)
        with torch.no_grad():
            for i, data in enumerate(tqdm(dl, total=len(dl), desc=f'running on the validation dataset (ID: {prefix})')):
                inp, label = map(lambda x: x.float().to(device), data)

                samples = self.ema.ema_model.sample(inp, batch_size=inp.size(0))

                summary = binary_classification_accuracy_4(samples, label)
                summary.update(shortest_path_1d_accuracy(samples, label, inp))
                for k, v in summary.items():
                    meters[k].update(v, n=inp.size(0))
                if i > 20:
                    break

            rows = [[k, v.avg] for k, v in meters.items()]
            print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
            print(tabulate(rows), flush=True)


as_float = lambda x: float(x.item())


@torch.no_grad()
def binary_classification_accuracy_4(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    numel = pred.numel()

    gt_0_pred_0 = ((label < 0.0) & (pred < 0.0)).sum() / numel
    gt_0_pred_1 = ((label < 0.0) & (pred >= 0.0)).sum() / numel
    gt_1_pred_0 = ((label > 0.0) & (pred < 0.0)).sum() / numel
    gt_1_pred_1 = ((label > 0.0) & (pred >= 0.0)).sum() / numel

    accuracy = gt_0_pred_0 + gt_1_pred_1
    balanced_accuracy = sum([
        gt_0_pred_0 / ((label < 0.0).float().sum() / numel),
        gt_1_pred_1 / ((label >= 0.0).float().sum() / numel),
    ]) / 2

    return {
        prefix + '/gt_0_pred_0': as_float(gt_0_pred_0),
        prefix + '/gt_0_pred_1': as_float(gt_0_pred_1),
        prefix + '/gt_1_pred_0': as_float(gt_1_pred_0),
        prefix + '/gt_1_pred_1': as_float(gt_1_pred_1),
        prefix + '/accuracy': as_float(accuracy),
        prefix + '/balance_accuracy': as_float(balanced_accuracy),
    }


def shortest_path_1d_accuracy(pred: torch.Tensor, label: torch.Tensor, inp: torch.Tensor, name: str = ''):
    """`first_action_accuracy_2` is the reported metric: the first predicted action
    follows a valid edge and decreases the shortest-path distance to the goal."""
    if name != '':
        name = '/' + name

    pred_argmax = pred[:, :, :, -1].argmax(-1)
    label_argmax = label[:, :, :, -1].argmax(-1)

    argmax_accuracy = (pred_argmax == label_argmax).float().mean()

    pred_argmax_first = pred_argmax[:, 0]
    label_argmax_first = label_argmax[:, 0]

    first_action_accuracy = (pred_argmax_first == label_argmax_first).float().mean()

    first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
    first_action_t = pred_argmax_first
    first_action_feasibility = (inp[
        torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
        first_action_s,
        first_action_t,
        0
    ] > 0).float().cpu()

    final_t = label_argmax[:, -1]
    first_action_accuracy_2 = first_action_distance_accuracy(inp[..., 0], first_action_s, final_t, first_action_t).float().cpu()
    first_action_accuracy_2 = first_action_accuracy_2 * first_action_feasibility

    return {
        'argmax_accuracy' + name: as_float(argmax_accuracy),
        'first_action_accuracy' + name: as_float(first_action_accuracy),
        'first_action_feasibility' + name: as_float(first_action_feasibility.mean()),
        'first_action_accuracy_2' + name: as_float(first_action_accuracy_2.mean()),
    }


def get_shortest_batch(edges: torch.Tensor) -> torch.Tensor:
    """ Return the length of shortest path between nodes. """
    b = edges.shape[0]
    n = edges.shape[1]

    # n + 1 indicates unreachable.
    shortest = torch.ones((b, n, n), dtype=torch.float32, device=edges.device) * (n + 1)
    shortest[torch.where(edges == 1)] = 1
    # Make sure that shortest[x, x] = 0
    shortest -= shortest * torch.eye(n).unsqueeze(0).to(shortest.device)

    # Floyd Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    shortest[:, i, j] = torch.min(shortest[:, i, j], shortest[:, i, k] + shortest[:, k, j])
    return shortest


def first_action_distance_accuracy(edge: torch.Tensor, s: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
    shortest = get_shortest_batch(edge.detach().cpu())
    b = edge.shape[0]
    # `shortest` is on the CPU, so index it with CPU tensors.
    b_arrange = torch.arange(b, dtype=torch.int64)
    return shortest[b_arrange, pred.cpu(), t.cpu()] < shortest[b_arrange, s.cpu(), t.cpu()]
