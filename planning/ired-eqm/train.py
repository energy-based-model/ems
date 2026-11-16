import os
import os.path as osp

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from eqm_lib.equilibrium_matching_1d import EquilibriumMatching1D, Trainer1D
from models import GNNConv1DEBMV2, GNNConv1DV2EBMWrapper
from planning_dataset import PlanningDataset

import argparse


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))


parser = argparse.ArgumentParser(description='Train an EqM energy model for shortest-path planning')

parser.add_argument('--dataset', default='shortest-path-15-1d', type=str, choices=['shortest-path-15-1d'], help='dataset to train on')
parser.add_argument('--model', default='gnn-conv-1d-v2', type=str, choices=['gnn-conv-1d-v2'])
parser.add_argument('--load-milestone', type=str, default=None, help='load a model from a milestone')
parser.add_argument('--batch_size', default=512, type=int, help='size of batch of input to use')
parser.add_argument('--train-num-steps', default=100000, type=int, help='number of training updates')
parser.add_argument('--data-workers', type=int, default=None, help='number of workers to use for data loading')
parser.add_argument('--supervise-energy-landscape', type=str2bool, default=True, help='add the contrastive loss')
parser.add_argument('--lambd', default=1.0, type=float, help='EqM target scale (lambda)')
parser.add_argument('--contrastive-scale', default=0.5, type=float, help='contrastive weight (beta)')
parser.add_argument('--sampler', default='gd', type=str, choices=['gd', 'ula'], help='sampler for the validation runs')
parser.add_argument('--sampling-steps', default=55, type=int, help='sampler steps for the validation runs; '
                    '55 GD steps use 110 model evaluations, the same as IRED (10 landscapes x (1 + 2 x 5))')
parser.add_argument('--step-size', default=0.05, type=float, help='sampler step size for the validation runs')
parser.add_argument('--evaluate', action='store_true', default=False)


if __name__ == "__main__":
    FLAGS = parser.parse_args()

    extra_validation_every_mul = 10
    save_and_sample_every = 1000

    if FLAGS.dataset == 'shortest-path-15-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-25': PlanningDataset('shortest-path-25-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        validation_batch_size = 64
    else:
        assert False

    if FLAGS.model == 'gnn-conv-1d-v2':
        model = GNNConv1DEBMV2(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
        model = GNNConv1DV2EBMWrapper(model)
    else:
        assert False

    eqm = EquilibriumMatching1D(
        model,
        lambd = FLAGS.lambd,
        supervise_energy_landscape = FLAGS.supervise_energy_landscape,
        contrastive_scale = FLAGS.contrastive_scale,
        sampler = FLAGS.sampler,
        sampling_steps = FLAGS.sampling_steps,
        step_size = FLAGS.step_size,
    )

    result_dir = osp.join('results', f'ds_{FLAGS.dataset}', f'model_{FLAGS.model}_eqm_lam{FLAGS.lambd}_bs{FLAGS.batch_size}')
    if not FLAGS.supervise_energy_landscape:
        result_dir = result_dir + '_nocl'
    elif FLAGS.contrastive_scale != 0.5:
        result_dir = result_dir + f'_cl{FLAGS.contrastive_scale}'
    os.makedirs(result_dir, exist_ok=True)

    trainer = Trainer1D(
        eqm,
        dataset,
        train_batch_size = FLAGS.batch_size,
        validation_batch_size = validation_batch_size,
        train_lr = 1e-4,
        train_num_steps = FLAGS.train_num_steps,  # total training steps
        ema_decay = 0.995,                        # exponential moving average decay
        data_workers = FLAGS.data_workers,
        results_folder = result_dir,
        validation_dataset = validation_dataset,
        extra_validation_datasets = extra_validation_datasets,
        extra_validation_every_mul = extra_validation_every_mul,
        save_and_sample_every = save_and_sample_every,
        evaluate_first = FLAGS.evaluate,  # run one evaluation first
    )

    if FLAGS.load_milestone is not None:
        trainer.load(FLAGS.load_milestone)

    trainer.train()
