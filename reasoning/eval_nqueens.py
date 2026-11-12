import torch
import torch.nn.functional as F
from models import MLP
from datasets import NQueensDataset
import os
from configs import NQueensConfig, load_config
from samplers import sampler_dict
import numpy as np

def is_valid_queens(board, N):
    if not (board.sum(0) == 1).all():
        return False
    if not (board.sum(1) == 1).all():
        return False
    for d in range(-(N - 1), N):
        if board.diagonal(d).sum() > 1:
            return False
        if board.flip(1).diagonal(d).sum() > 1:
            return False
    return True

def greedy_decode(board, N):
    flat = board.view(-1)
    _, indices = torch.sort(flat, descending=True)
    indices = indices[:N]
    board = torch.zeros(N, N)

    for idx in indices:
        r = idx.item() // N
        c = idx.item() % N
        board[r, c] = 1
        valid = True
        for d in range(-(N - 1), N):
            if board.diagonal(d).sum() > 1 or board.flip(1).diagonal(d).sum() > 1:
                valid = False
                break
        if not valid or board.sum(0).max() > 1 or board.sum(1).max() > 1:
            board[r, c] = 0

    return board

if __name__ == '__main__':
    config = load_config(NQueensConfig)
    seed = config.seed
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.mps.manual_seed(seed)
    else:
        device = torch.device('cpu')

    dataset = NQueensDataset(config.train_path)
    N = dataset.N

    model = MLP(
        input_dim=N,
        embed_dim=config.embed_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio
    )
    model.to(device)
    model.eval()

    sampler = sampler_dict[config.sampler](config)

    num_params = sum(p.numel() for p in model.parameters())

    if os.path.exists(config.model_ckpt):
        print(f'Loading checkpoint: {config.model_ckpt}')
        state_dict = torch.load(config.model_ckpt, map_location=device)
        if config.use_ema_model:
            print('Using EMA model.')
            ema_state_dict = {k.replace('module.', ''): v for k, v in state_dict['ema_model'].items() if k != 'n_averaged'}
            model.load_state_dict(ema_state_dict, strict=True)
        else:
            print(f'Using normal model.')
            model.load_state_dict(state_dict['model'], strict=True)
    else:
        raise FileNotFoundError('Could not find/load model checkpoint.')

    print(f'Using device: {device}')
    print(f'Model name: {config.model_name}')
    print(f'Number of parameters: {num_params}')
    print(f'N: {N}')
    print(f'Number of particles: {config.num_particles}')
    print(f'Number of sampling steps: {config.sampling_steps}')
    print(f'Step size: {config.step_size}')
    print(f'Using sampler: {config.sampler}')

    num_samples = config.num_trials
    num_valid = 0
    all_sizes = []

    def energy_func(x):
        num_particles = x.shape[0]
        board = x.reshape(num_particles, N, N)
        all_slices = []

        for r in range(N):
            all_slices.append(board[:, r, :])
        for c in range(N):
            all_slices.append(board[:, :, c])
        for d in range(-(N - 1), N):
            diag = torch.stack([board[:, i, i + d] for i in range(N) if 0 <= i + d < N], dim=1)
            all_slices.append(F.pad(diag, (0, N - diag.shape[1])))
        for d in range(-(N - 1), N):
            diag = torch.stack([board[:, i, (N - 1) - i - d] for i in range(N) if 0 <= (N - 1) - i - d < N], dim=1)
            all_slices.append(F.pad(diag, (0, N - diag.shape[1])))

        all_slices = torch.stack(all_slices, dim=1)
        flat = all_slices.reshape(-1, N)
        energy = model(flat)
        energy = energy.reshape(num_particles, -1)
        return energy / 4.0

    for sample in range(num_samples):
        if hasattr(sampler, 'v'):
            sampler.v = None
        if hasattr(sampler, 'p'):
            sampler.p = None

        x = torch.randn(config.num_particles, N * N, device=device, requires_grad=True)

        for step in range(config.sampling_steps):
            x = sampler.step(x, energy_func)
            x = x.detach().requires_grad_(True)

        with torch.no_grad():
            energy = energy_func(x)
            total_energy = energy.sum(dim=-1)
            best_idx = total_energy.argmin()

        prediction = x[best_idx].detach().reshape(N, N)

        board = greedy_decode(prediction, N)

        valid = is_valid_queens(board, N)
        num_valid += int(valid)
        all_sizes.append(board.sum().item())

        if (sample + 1) % 10 == 0:
            print(f'Sample {sample + 1}: Valid so far: {num_valid}/{sample + 1}')

    print(f'\nCorrect Instances: {num_valid}/{num_samples}')
    print(f'Size: {np.mean(all_sizes):.2f}, {np.std(all_sizes):.2f}')
