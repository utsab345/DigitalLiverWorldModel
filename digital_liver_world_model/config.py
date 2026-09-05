from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    n_patients: int = 300
    n_steps: int = 120
    context_length: int = 24
    forecast_horizon: int = 12
    batch_size: int = 64
    train_frac: float = 0.8

    # Model
    latent_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 4

    # SSL / JEPA
    target_momentum: float = 0.996
    predictor_hidden: int = 128

    # Training
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 50
    grad_clip: float = 1.0

    # Loss weights
    jepa_weight: float = 0.1
    recon_weight: float = 1.0
    monotonic_weight: float = 1.0
    std_target: float = 1.0
    variance_weight: float = 1.0
    cov_weight: float = 0.1

    # Data augmentation
    stride: int = 1
    pin_memory: bool = False
    num_workers: int = 0

    # Generation
    seed: int = 42

    # Paths
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    figure_dir: str = "outputs/figures"
    report_dir: str = "outputs/reports"

    # World-model type: "gru" or "neural_ode"
    world_model: str = "gru"

    # Enforce constraints by construction
    enforce_monotonic: bool = True

    # Early stopping
    patience: int = 30
