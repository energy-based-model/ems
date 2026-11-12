from dataclasses import dataclass, field

# 3-SAT (Appendix B.3). Defaults reproduce the main three-seed comparison
# (seeds 0, 13, 42).

@dataclass
class SATConfig:
    seed: int = 0
    task: str = 'sat'

    # Model
    embed_dim: int = 128
    num_layers: int = 3
    mlp_ratio: float = 2.0
    input_dim: int = 6              # 3 assignment values + 3 negation indicators
    output_dim: int = 1
    ckpt_dir: str = './checkpoints'
    model_name: str = ''            # defaults to sat-mlp-seed{seed}

    # Training
    resume_training: bool = False
    use_cl: bool = True
    cl_scale: float = 0.5           # contrastive weight (beta)
    batch_size: int = 1024
    num_workers: int = 1
    epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lambd: float = 0.007            # EqM target scale (lambda); ablations use 0.005
    ema_decay: float = 0.9999
    train_path: str = './data/sat/train'
    valid_path: str = './data/sat/valid'

    # Eval
    use_ema_model: bool = False
    test_path: str = './data/sat/test/uf20-91'
    num_particles: int = 1024       # candidates
    sampler: str = 'gd'             # gd | nag | ula | mala | uhmc | hmc | pem
    sampling_steps: int = 100
    step_size: float = 1e-1
    # NAG
    mu: float = 0.3
    # UHMC / HMC
    persistent: bool = True
    num_leapfrog_steps: int = 3
    damping: float = 0.9
    # PEM
    noise_scale: float = 0.1
    use_ula: bool = False

    # Derived
    model_ckpt: str = field(init=False)

    def __post_init__(self):
        if not self.model_name:
            self.model_name = f'sat-mlp-seed{self.seed}'
        self.model_ckpt = f'{self.ckpt_dir}/{self.model_name}.pt'
