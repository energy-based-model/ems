from dataclasses import dataclass, field

# Crosswords (Appendix B.4). Defaults reproduce the main configuration.
# Crossword training uses EqM only (no contrastive term, beta = 0).

@dataclass
class CrosswordConfig:
    seed: int = 0
    task: str = 'crossword'

    # Model
    embed_dim: int = 1024
    num_layers: int = 3
    mlp_ratio: float = 1.0
    clue_embedding: int = 1536
    vocab_size: int = 26
    ckpt_dir: str = './checkpoints'
    model_name: str = 'crossword-mlp'

    # Training
    resume_training: bool = False
    batch_size: int = 2048
    num_workers: int = 1
    epochs: int = 1500
    learning_rate: float = 1e-4
    weight_decay: float = 0.8
    lambd: float = 8.0              # EqM target scale (lambda)
    ema_decay: float = 0.9999
    train_path: str = './data/crossword/train.pkl'
    valid_path: str = './data/crossword/valid.pkl'

    # Eval
    use_ema_model: bool = False
    test_path: str = './data/crossword/test.pkl'
    instances_path: str = './data/crossword/mini0505.json'
    num_particles: int = 1          # candidates
    sampler: str = 'gd'             # gd | nag | ula | mala | uhmc | hmc | pem
    sampling_steps: int = 500
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
    input_dim: int = field(init=False)
    output_dim: int = field(init=False)
    model_ckpt: str = field(init=False)

    def __post_init__(self):
        self.input_dim = self.clue_embedding + self.vocab_size * 5
        self.output_dim = self.input_dim
        self.model_ckpt = f'{self.ckpt_dir}/{self.model_name}.pt'
