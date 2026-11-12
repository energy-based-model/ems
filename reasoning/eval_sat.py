import torch
from models import MLP
from configs import SATConfig, load_config
from datasets import SATEvalDataset
from torch.utils.data import DataLoader
from samplers import sampler_dict
import os

if __name__ == "__main__":
    config = load_config(SATConfig)
    seed = config.seed
    torch.manual_seed(seed)

    pin_memory = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
        pin_memory = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(seed)
    else:
        device = torch.device("cpu")

    model = MLP(
        input_dim=config.input_dim,
        embed_dim=config.embed_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
    )
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    if os.path.exists(config.model_ckpt):
        print(f"Loading checkpoint: {config.model_ckpt}")
        state_dict = torch.load(config.model_ckpt, map_location=device)
        if config.use_ema_model:
            print("Using EMA model.")
            ema_state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict["ema_model"].items()
                if k != "n_averaged"
            }
            model.load_state_dict(ema_state_dict, strict=True)
        else:
            print("Using normal model.")
            model.load_state_dict(state_dict["model"], strict=True)
    else:
        raise FileNotFoundError("Could not find/load model checkpoint.")

    print(f"Using device: {device}")
    print(f"Model name: {config.model_name}")
    print(f"Number of parameters: {num_params}")
    print(f"Number of particles: {config.num_particles}")
    print(f"Number of sampling steps: {config.sampling_steps}")
    print(f"Step size: {config.step_size}")
    print(f"Using sampler: {config.sampler}")

    dataset = SATEvalDataset(config.test_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    sampler = sampler_dict[config.sampler](config)

    total_correct = 0
    total_satisfied = 0
    total_clauses = 0
    num_instances = 0

    for indices, negation, num_variables in dataloader:
        indices = indices.squeeze(0).to(device)
        negation = negation.squeeze(0).to(device)
        num_variables = num_variables.item()

        if hasattr(sampler, "v"):
            sampler.v = None
        if hasattr(sampler, "p"):
            sampler.p = None

        P = config.num_particles

        def energy_func(x):
            clause_vars = x[:, indices]
            clause_input = torch.cat(
                [clause_vars, negation.unsqueeze(0).expand(P, -1, -1)], dim=-1
            )
            energy = model(clause_input)
            return energy

        x = torch.randn(P, num_variables, device=device, requires_grad=True)

        for step in range(config.sampling_steps):
            x = sampler.step(x, energy_func)
            x = x.detach().requires_grad_(True)

        x = torch.clamp(x, 0, 1)
        with torch.no_grad():
            energy = energy_func(x)
            total_energy = energy.sum(dim=-1)
            best_idx = total_energy.argmin()

        assignment = torch.round(x[best_idx].detach().clamp(0, 1)).long()

        num_clauses = indices.shape[0]
        clause_vals = assignment[indices]
        negation_bool = negation.bool()
        literal_satisfied = (~negation_bool & (clause_vals == 1)) | (
            negation_bool & (clause_vals == 0)
        )
        clause_satisfied = literal_satisfied.any(dim=-1)

        satisfied = clause_satisfied.sum().item()
        is_correct = satisfied == num_clauses

        total_correct += int(is_correct)
        total_satisfied += satisfied
        total_clauses += num_clauses
        num_instances += 1

        if num_instances % 10 == 0:
            print(
                f"Instance {num_instances}: Correct so far: {total_correct}/{num_instances} | Clauses: {total_satisfied}/{total_clauses}"
            )

    print(f"\nCorrect Instances: {total_correct}/{num_instances}")
    print(
        f"Satisfied Clauses: {total_satisfied}/{total_clauses} ({total_satisfied / total_clauses * 100:.2f}%)"
    )
