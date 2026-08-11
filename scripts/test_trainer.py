"""
Smoke test for the LayoutLMv3 training engine.

Runs exactly one training batch to verify:
    - model loading
    - DataLoader
    - mixed precision
    - forward pass
    - loss
    - backward pass
    - optimizer step
"""

import sys
from pathlib import Path

import torch

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.training import get_training_config
from app.layoutlm.trainer import DocumentTrainer


def main():

    print("=" * 60)
    print("TRAINER SMOKE TEST")
    print("=" * 60)

    config = get_training_config()

    # Keep the real configuration, but only run one batch.
    trainer = DocumentTrainer(config)

    print("\nLoading one training batch...")

    batch = next(iter(trainer.train_loader))

    batch = trainer._move_batch_to_device(batch)

    print(
        f"input_ids:      {tuple(batch['input_ids'].shape)}"
    )
    print(
        f"attention_mask: {tuple(batch['attention_mask'].shape)}"
    )
    print(
        f"bbox:           {tuple(batch['bbox'].shape)}"
    )
    print(
        f"pixel_values:   {tuple(batch['pixel_values'].shape)}"
    )
    print(
        f"labels:         {tuple(batch['labels'].shape)}"
    )

    trainer.model.train()

    trainer.optimizer.zero_grad(
        set_to_none=True
    )

    print("\nRunning forward pass...")

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=trainer.use_amp,
    ):

        outputs = trainer.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )

    print(
        f"Logits shape: {tuple(outputs.logits.shape)}"
    )

    print(
        f"Loss: {outputs.loss.item():.6f}"
    )

    print("\nRunning backward pass...")

    trainer.scaler.scale(
        outputs.loss
    ).backward()

    print("Backward pass complete.")

    print("\nRunning optimizer step...")

    trainer.scaler.step(
        trainer.optimizer
    )

    trainer.scaler.update()

    trainer.optimizer.zero_grad(
        set_to_none=True
    )

    print("Optimizer step complete.")

    print("\n" + "=" * 60)
    print("TRAINER SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()