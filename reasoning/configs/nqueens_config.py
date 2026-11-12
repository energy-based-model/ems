from dataclasses import dataclass, field

# N-Queens (Appendix B.2). Defaults reproduce the main 8-Queens configuration.

@dataclass
class NQueensConfig:
    seed: int = 0
    task: str = 'nqueens'
    n_queen: int = 8

    # Model
    embed_dim: int = 128
    num_layers: int = 3
    mlp_ratio: float = 2.0
    output_dim: int = 1
    ckpt_dir: str = './checkpoints'
    model_name: str = ''            # defaults to nqueens{n_queen}-mlp

    # Training
    resume_training: bool = False
    use_cl: bool = True
    cl_scale: float = 0.5           # contrastive weight (beta)
    batch_size: int = 2048
    num_samples: int = 20000        # rows/columns sampled with replacement per epoch
    num_workers: int = 8
    epochs: int = 2040
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lambd: float = 16.0             # EqM target scale (lambda)
    ema_decay: float = 0.9999

    # Eval
    use_ema_model: bool = False
    num_trials: int = 100
    num_particles: int = 128        # candidates
    sampler: str = 'gd'             # gd | nag | ula | mala | uhmc | hmc | pem
    sampling_steps: int = 20
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
    train_path: str = field(init=False)
    valid_path: str = field(init=False)
    test_path: str = field(init=False)
    model_ckpt: str = field(init=False)

    def __post_init__(self):
        self.train_path = f'./data/nqueens/{self.n_queen}_queens_train.txt'
        self.valid_path = f'./data/nqueens/{self.n_queen}_queens_valid.txt'
        self.test_path = f'./data/nqueens/{self.n_queen}_queens_test.txt'
        if not self.model_name:
            self.model_name = f'nqueens{self.n_queen}-mlp'
        self.model_ckpt = f'{self.ckpt_dir}/{self.model_name}.pt'
