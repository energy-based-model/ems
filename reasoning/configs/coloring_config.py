from dataclasses import dataclass, field

# Graph coloring (Appendix B.1). Defaults reproduce the main configuration.

@dataclass
class ColoringConfig:
    seed: int = 0
    task: str = 'coloring'

    # Model
    embed_dim: int = 256
    num_layers: int = 3
    mlp_ratio: float = 2.0
    num_colors: int = 14
    output_dim: int = 1
    ckpt_dir: str = './checkpoints'
    model_name: str = 'coloring-mlp'

    # Training
    resume_training: bool = False
    use_cl: bool = True
    cl_scale: float = 1.0           # contrastive weight (beta)
    batch_size: int = 1024
    num_samples: int = 50000        # edges sampled with replacement per epoch
    num_workers: int = 8
    epochs: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lambd: float = 4.0              # EqM target scale (lambda)
    ema_decay: float = 0.9999

    # Eval
    use_ema_model: bool = True
    graphs_dir: str = './data/coloring'
    num_particles: int = 128        # candidates
    sampler: str = 'ula'            # gd | nag | ula | mala | uhmc | hmc | pem
    sampling_steps: int = 500
    step_size: float = 1e-2
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
    model_ckpt: str = field(init=False)

    def __post_init__(self):
        self.input_dim = 2 * self.num_colors
        self.model_ckpt = f'{self.ckpt_dir}/{self.model_name}.pt'
