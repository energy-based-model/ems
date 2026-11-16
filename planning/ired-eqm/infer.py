"""Evaluate a trained checkpoint with more or fewer sampling steps.

Scores the same graphs as the validation runs in train.py (the first 22
batches of 64 from the 15-node validation split and the 25-node set) with
the checkpoint's EMA weights. By default there is one sample per graph;
--particles K keeps the lowest-energy of K independent samples.

    python infer.py --ckpt results/ds_shortest-path-15-1d/model_gnn-conv-1d-v2_eqm_lam1.0_bs512/model-90.pt \
        --sampling-steps 55 110 --particles 1 8

--sampler picks the search method. 'gd' is the GD with bad-step rejection
used during training; the others come from samplers.py (shared with the
reasoning code).
"""

import os

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import collections
from types import SimpleNamespace

import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tabulate import tabulate

from eqm_lib.equilibrium_matching_1d import AverageMeter, EquilibriumMatching1D, shortest_path_1d_accuracy
from models import GNNConv1DEBMV2, GNNConv1DV2EBMWrapper
from planning_dataset import PlanningDataset
from samplers import sampler_dict


# Settings for the samplers from samplers.py, as in the paper's sampler ablation.
SAMPLER_KWARGS = {
    'nag': {'mu': 0.9},
    'ula': {},
    'mala': {},
    'uhmc': {'num_leapfrog_steps': 4, 'damping': 0.9, 'persistent': False},
    'hmc': {'num_leapfrog_steps': 4, 'damping': 0.9, 'persistent': False},
}
# Model evaluations per step (each is one energy and gradient).
EVALS_PER_STEP = {'gd': 2, 'nag': 1, 'ula': 1, 'mala': 2, 'uhmc': 5, 'hmc': 5}

parser = argparse.ArgumentParser(description='Evaluate an EqM shortest-path checkpoint')
parser.add_argument('--ckpt', required=True, type=str, help='path to a model-*.pt checkpoint')
parser.add_argument('--sampler', default='gd', type=str, choices=['gd', 'nag', 'ula', 'mala', 'uhmc', 'hmc'])
parser.add_argument('--sampling-steps', default=[55], type=int, nargs='+',
                    help='one or more step counts; each is run on the same graphs and initial noise')
parser.add_argument('--step-size', default=0.05, type=float)
parser.add_argument('--particles', default=[1], type=int, nargs='+',
                    help='one or more candidate counts K; each keeps the lowest-energy of K chains per graph')
parser.add_argument('--batch_size', default=64, type=int, help='validation batch size, as in train.py')
parser.add_argument('--max-batches', default=22, type=int, help='batches per set, as in train.py and IRED')
parser.add_argument('--seed', default=0, type=int, help='seed for the initial noise')
parser.add_argument('--data-workers', default=0, type=int)


@torch.no_grad()
def draw(eqm, inp, b, sampler, steps, step_size):
    """One sample per graph with the chosen sampler."""
    if sampler == 'gd':
        eqm.sampling_steps = steps
        return eqm.sample(inp, batch_size=b)

    chain = sampler_dict[sampler](SimpleNamespace(step_size=step_size, **SAMPLER_KWARGS[sampler]))
    energy = lambda y: eqm.model(inp, y, return_energy=True)
    x = eqm.model.randn(b, eqm.out_shape, inp, inp.device)
    with torch.enable_grad():
        for _ in range(steps):
            x = chain.step(x.detach().requires_grad_(True), energy)
    return x.detach()


@torch.no_grad()
def run_set(eqm, dl, device, max_batches, particles, sampler, steps, step_size):
    """Score best-of-K for each K in `particles`.

    max(particles) independent chains are run per graph, and best-of-K keeps
    the lowest-energy sample among the first K chains, so every K reuses the
    same chains.
    """
    meters = {k: collections.defaultdict(AverageMeter) for k in particles}
    for i, (inp, label) in enumerate(dl):
        if i >= max_batches:
            break
        inp, label = inp.float().to(device), label.float().to(device)
        b = inp.size(0)

        samples, energies = [], []
        for _ in range(max(particles)):
            y = draw(eqm, inp, b, sampler, steps, step_size)
            samples.append(y)
            energies.append(eqm.model(inp, y, return_energy=True).view(-1))
        samples = torch.stack(samples)    # [K, B, T, N, 1]
        energies = torch.stack(energies)  # [K, B]

        for k in particles:
            best = energies[:k].argmin(dim=0)
            pred = samples[best, torch.arange(b, device=device)]
            for name, v in shortest_path_1d_accuracy(pred, label, inp).items():
                meters[k][name].update(v, n=b)
    n = meters[particles[0]]['first_action_accuracy_2'].count
    return {k: {name: v.avg for name, v in m.items()} for k, m in meters.items()}, n


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_sets = {
        '15-node': PlanningDataset('shortest-path-15-1d', split='validation', num_identifier=100000),
        '25-node': PlanningDataset('shortest-path-25-1d', split='train', num_identifier=10000),
    }
    dls = {name: DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.data_workers)
           for name, ds in eval_sets.items()}

    ds = eval_sets['15-node']
    model = GNNConv1DV2EBMWrapper(GNNConv1DEBMV2(inp_dim=ds.inp_dim, out_dim=ds.out_dim))
    eqm = EquilibriumMatching1D(model, sampler='gd', step_size=FLAGS.step_size).to(device)

    # Load the EMA weights the same way Trainer1D saves them.
    ema = EMA(eqm, beta=0.995, update_every=10)
    data = torch.load(FLAGS.ckpt, map_location=device, weights_only=False)
    ema.load_state_dict(data['ema'])
    eqm = ema.ema_model.to(device)
    eqm.eval()
    print(f'Loaded the EMA weights at update {data["step"]} from {FLAGS.ckpt}', flush=True)

    particles = sorted(set(FLAGS.particles))
    rows = []
    for steps in FLAGS.sampling_steps:
        evals = steps * EVALS_PER_STEP[FLAGS.sampler]
        results = {}
        for name, dl in dls.items():
            torch.manual_seed(FLAGS.seed)
            results[name], n = run_set(eqm, dl, device, FLAGS.max_batches, particles,
                                       FLAGS.sampler, steps, FLAGS.step_size)
            for k in particles:
                m = results[name][k]
                print(f'{FLAGS.sampler} steps={steps} evals={evals} particles={k}: {name} '
                      f'first_action_accuracy_2={m["first_action_accuracy_2"]:.4f} '
                      f'feasibility={m["first_action_feasibility"]:.4f} ({n} graphs)', flush=True)
        for k in particles:
            row = [FLAGS.sampler, steps, FLAGS.step_size, evals, k]
            for name in dls:
                row += [results[name][k]['first_action_accuracy_2'], results[name][k]['first_action_feasibility']]
            rows.append(row)

    headers = ['sampler', 'steps', 'step size', 'model evals', 'particles', '15 acc', '15 feas', '25 acc', '25 feas']
    print(tabulate(rows, headers=headers, floatfmt='.4f'), flush=True)
