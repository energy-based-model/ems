import torch
import torch.nn as nn
from models import MLP, EBM
from datasets import SATDataset
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import os
from configs import SATConfig, load_config
from eqm import eqm_sample

def run_epoch(config, model, ema_model, ebm, dataloader, mse_criterion, cl_criterion, device, optimizer=None):
    is_train = optimizer is not None
    ebm.train() if is_train else ebm.eval()

    total_loss = 0.0
    count = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for solution, negation, negative in dataloader:
            solution = solution.to(device)
            negation = negation.to(device)

            eps = torch.randn_like(solution, device=device)
            gamma = torch.rand(solution.shape[0], device=device)

            x_gamma, target = eqm_sample(solution, config.lambd, eps, gamma)
            prediction, positive_energy = ebm(x_gamma, negation)

            loss = mse_criterion(prediction, target)

            if config.use_cl:
                negative = negative.to(device)
                
                x_gamma_negative, _ = eqm_sample(negative, config.lambd, eps, gamma)
                negative_energy = model(x_gamma_negative, negation)

                logits = torch.cat([-positive_energy.unsqueeze(1), -negative_energy.unsqueeze(1)], dim=1)
                labels = torch.zeros(negative.shape[0], dtype=torch.long, device=device)

                loss_cl = cl_criterion(logits, labels)
                loss = loss + config.cl_scale * loss_cl

            if is_train:
                loss.backward()
                optimizer.step()
                ema_model.update_parameters(model)
                optimizer.zero_grad()

            total_loss += loss.item()
            count += 1

    return total_loss / count

if __name__ == '__main__':
    config = load_config(SATConfig)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    seed = config.seed
    torch.manual_seed(seed)

    pin_memory = False
    fused = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        fused = True
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.mps.manual_seed(seed)
    else:
        device = torch.device('cpu')

    train_dataset = SATDataset(config.train_path)
    valid_dataset = SATDataset(config.valid_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )

    model = MLP(
        input_dim=config.input_dim,
        embed_dim=config.embed_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio
    )
    model.to(device)
    model.train()

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay))
    ema_model.to(device)
    ema_model.eval()

    ebm = EBM(model)
    ebm.to(device)
    ebm.train()

    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=fused, weight_decay=config.weight_decay)
    mse_criterion = nn.MSELoss()
    cl_criterion = nn.CrossEntropyLoss()

    epochs = config.epochs
    start_epoch = 0
    best_val_loss = float('inf')

    if config.resume_training:
        print('Resuming training.')
        if os.path.exists(config.model_ckpt):
            state_dict = torch.load(config.model_ckpt, map_location=device)
            model.load_state_dict(state_dict['model'], strict=True)
            ema_model.load_state_dict(state_dict['ema_model'], strict=True)
            optimizer.load_state_dict(state_dict['optimizer'])
            start_epoch = state_dict['epoch']
            best_val_loss = state_dict['best_val_loss']
        else:
            raise FileNotFoundError('Resume training is on but could not find/load model.')
    else:
        print('Starting training from scratch.')

    print(f'Using device: {device}')
    print(f'Model name: {config.model_name}')
    print(f'Number of parameters: {num_params}')
    print(f'Weight decay: {config.weight_decay}')

    print(f'Starting training from epoch: {start_epoch}')
    for epoch in range(start_epoch, epochs):
        train_loss = run_epoch(
            config,
            model,
            ema_model,
            ebm,
            train_loader,
            mse_criterion,
            cl_criterion,
            device,
            optimizer
        )            
        
        if epoch % 1 == 0:
            val_loss = run_epoch(
                config,
                model,
                ema_model,
                ebm,
                valid_loader,
                mse_criterion,
                cl_criterion,
                device
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f'Saving model.')
                ckpt = {
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss
                }

                torch.save(ckpt, config.model_ckpt)

        print(f'Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Best Val Loss: {best_val_loss}')

    print('Done training!')
