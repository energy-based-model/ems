import torch
import torch.nn.functional as F
from models import MLP
from configs import ColoringConfig, load_config
from datasets import parse_graph_file
import os
import glob
import numpy as np
from samplers import sampler_dict

DISTRIBUTIONS = {
    'Erdos Renyi (Small)': 'erdos_renyi_20_40_*',
    'Erdos Renyi (Large)': 'erdos_renyi_80_100_*',
    'Holme Kim (Small)': 'powerlaw_cluster_graph_20_40_*',
    'Holme Kim (Large)': 'powerlaw_cluster_graph_80_100_*',
    'Regular Expander (Small)': 'random_regular_expander_graph_20_40_*',
    'Regular Expander (Large)': 'random_regular_expander_graph_80_100_*',
    'Paley': 'paley_graph_*',
    'Complete': 'complete_graph_*',
}

def solve_graph(config, model, sampler, num_nodes, edges, chrom_number, device):
    if hasattr(sampler, 'v'):
        sampler.v = None
    if hasattr(sampler, 'p'):
        sampler.p = None

    u, v = edges[:, 0], edges[:, 1]
    
    def energy_func(x):
        left_node = F.pad(x[:, u], (0, config.num_colors - chrom_number))
        right_node = F.pad(x[:, v], (0, config.num_colors - chrom_number))
        pairs = torch.cat((left_node, right_node), dim=-1)
        energy = model(pairs)
        return energy
    
    x = torch.randn(config.num_particles, num_nodes, chrom_number, device=device, requires_grad=True)

    for i in range(config.sampling_steps):
        x = sampler.step(x, energy_func)
        x = x.detach().requires_grad_(True)
        x = torch.clamp(x, -1, 1)

    with torch.no_grad():
        energy = energy_func(x)
        total_energy = energy.sum(dim=-1)
        best_idx = total_energy.argmin()
        
    colors = x[best_idx].detach().argmax(dim=-1)
    conflicts = (colors[u] == colors[v]).sum().item()

    return conflicts

if __name__ == '__main__':
    config = load_config(ColoringConfig)
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

    model = MLP(
        input_dim=config.input_dim,
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
    print(f'Number of particles: {config.num_particles}')
    print(f'Number of sampling steps: {config.sampling_steps}')
    print(f'Step size: {config.step_size}')
    print(f'Using sampler: {config.sampler}')

    results = {}

    for distribution, graph in DISTRIBUTIONS.items():
        files = sorted(glob.glob(os.path.join(config.graphs_dir, graph)))

        if not files:
            continue
        
        conflicts_list = []

        for file in files:
            num_nodes, edges, chrom_number = parse_graph_file(file)
            edges = edges.to(device)

            conflicts = solve_graph(
                config,
                model,
                sampler,
                num_nodes,
                edges,
                chrom_number,
                device
            )

            conflicts_list.append(conflicts)

        mean = np.mean(conflicts_list)
        std = np.std(conflicts_list)
        results[distribution] = (mean, std)

    print('\n' + '=' * 50)
    print(f'{"Distribution":<30} | {"Conflicts":<20}')
    print('-' * 50)
    for distribution, (mean, std) in results.items():
        print(f'{distribution:<30} | {mean:.2f} ± {std:.2f}')
    print('=' * 50)
