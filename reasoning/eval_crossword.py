import torch
from models import MLP
from configs import CrosswordConfig, load_config
from datasets import CrosswordEvalDataset
from torch.utils.data import DataLoader
import os
from samplers import sampler_dict

if __name__ == '__main__':
    config = load_config(CrosswordConfig)
    seed = config.seed
    torch.manual_seed(seed)

    pin_memory = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(seed)
        pin_memory = True
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
    
    dataset = CrosswordEvalDataset(config.test_path, config.instances_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )

    sampler = sampler_dict[config.sampler](config)

    print(f'Using device: {device}')
    print(f'Model name: {config.model_name}')
    print(f'Number of parameters: {num_params}')
    print(f'Number of particles: {config.num_particles}')
    print(f'Number of sampling steps: {config.sampling_steps}')
    print(f'Step size: {config.step_size}')
    print(f'Using sampler: {config.sampler}')

    total_letters = 0
    total_words = 0
    num_puzzles = 0

    def energy_func(x):
        puzzle = x.reshape(config.num_particles, 5, 5, 26)
        h_words = puzzle.flatten(2)
        v_words = puzzle.permute(0, 2, 1, 3).flatten(2)
        h_input = torch.cat((h_words, h_emb.unsqueeze(0).expand(config.num_particles, -1, -1)), dim=-1)
        v_input = torch.cat((v_words, v_emb.unsqueeze(0).expand(config.num_particles, -1, -1)), dim=-1)
        all_input = torch.cat((h_input, v_input), dim=1)
        energy = model(all_input) / 2.0
        return energy

    for target, h_emb, v_emb in dataloader:
        h_emb = h_emb.squeeze(0).to(device)
        v_emb = v_emb.squeeze(0).to(device)
        target = target.squeeze(0).to(device)

        x = torch.randn(config.num_particles, 25, 26, device=device, requires_grad=True)

        for i in range(config.sampling_steps):
            x = sampler.step(x, energy_func)

            x = x.detach().requires_grad_(True)
            

        x = torch.clamp(x, -1, 1)

        with torch.no_grad():
            energy = energy_func(x)
            total_energy = energy.sum(dim=-1)
            best_idx = total_energy.argmin()

        prediction = x[best_idx].detach().argmax(dim=-1)
        target = target.argmax(dim=-1)

        matching_letters = (prediction == target).sum().item()

        prediction_puzzle = prediction.reshape(5, 5)
        target_puzzle = target.reshape(5, 5)

        matching_words = 0

        for i in range(5):
            if torch.equal(prediction_puzzle[i], target_puzzle[i]):
                matching_words += 1
            if torch.equal(prediction_puzzle[:, i], target_puzzle[:, i]):
                matching_words += 1

        total_letters += matching_letters
        total_words += matching_words
        num_puzzles += 1

    print(f'Letter Success Rate: {total_letters / (num_puzzles * 25) * 100:.1f}')
    print(f'Word Success Rate: {total_words / (num_puzzles * 10) * 100:.1f}')
