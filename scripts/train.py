"""
Train LayoutLMv3 document classifier.
"""

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.training import get_training_config
from app.layoutlm.trainer import DocumentTrainer


def main():

    print("=" * 60)
    print("DOCINTEL LAYOUTLMV3 TRAINING")
    print("=" * 60)

    config = get_training_config()

    print("\nTraining configuration:")
    print(f"Model: {config.model_name}")
    print(f"Classes: {config.num_labels}")
    print(f"Batch size: {config.batch_size}")
    print(
        f"Gradient accumulation: "
        f"{config.gradient_accumulation_steps}"
    )
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")

    trainer = DocumentTrainer(config)

    trainer.train()


if __name__ == "__main__":
    main()