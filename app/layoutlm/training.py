"""
LayoutLMv3 Training Configuration

Centralizes the configuration used for fine-tuning
the document classification model.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for LayoutLMv3 fine-tuning."""

    # Model
    model_name: str = (
        "microsoft/layoutlmv3-base"
    )

    num_labels: int = 5

    # Dataset
    train_dir: str = (
        "data/processed_clean/train"
    )

    validation_dir: str = (
        "data/processed_clean/validation"
    )

    test_dir: str = (
        "data/processed_clean/test"
    )

    # Training
    batch_size: int = 2

    gradient_accumulation_steps: int = 4

    epochs: int = 5

    learning_rate: float = 2e-5

    weight_decay: float = 0.01

    # DataLoader
    num_workers: int = 0

    pin_memory: bool = True

    # Checkpoints
    checkpoint_dir: str = (
        "checkpoints"
    )

    best_model_path: str = (
        "checkpoints/best_model"
    )

    # Reproducibility
    seed: int = 42


def get_training_config():
    """Return the default training configuration."""

    return TrainingConfig()