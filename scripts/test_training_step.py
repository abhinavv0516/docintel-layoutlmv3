"""
Test one LayoutLMv3 training step.

Verifies:
    DataLoader
    -> model
    -> logits
    -> loss
    -> backward pass
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForSequenceClassification

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.cached_dataset import (
    CachedDocumentDataset,
)
from app.layoutlm.labels import NUM_CLASSES


MODEL_NAME = "microsoft/layoutlmv3-base"

BATCH_SIZE = 2


def main():

    print("=" * 60)
    print("LAYOUTLMV3 TRAINING STEP TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    print("\nLoading dataset...")

    dataset = CachedDocumentDataset(
        "data/processed/train"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    print("\nLoading LayoutLMv3 classifier...")

    model = (
        LayoutLMv3ForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
        )
    )

    model.to(device)

    model.train()

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print("\nLoading batch...")

    batch = next(iter(dataloader))

    # Move tensors to GPU
    batch = {
        key: value.to(
            device,
            non_blocking=True,
        )
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }

    print("Batch loaded.")

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    print("\nRunning forward pass...")

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        bbox=batch["bbox"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"],
    )

    print("Forward pass complete.")

    # --------------------------------------------------
    # Inspect output
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL OUTPUT")
    print("=" * 60)

    print(
        f"Logits shape: {outputs.logits.shape}"
    )

    print(
        f"Expected shape: "
        f"({BATCH_SIZE}, {NUM_CLASSES})"
    )

    print(
        f"\nLoss: {outputs.loss.item():.6f}"
    )

    print(
        "\nLogits:"
    )

    print(
        outputs.logits
    )

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    print("\nRunning backward pass...")

    outputs.loss.backward()

    print("Backward pass complete.")

    # --------------------------------------------------
    # Verify gradients
    # --------------------------------------------------

    gradient_found = False

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            gradient_found = True

            print(
                f"\nGradient found: {name}"
            )

            print(
                f"Gradient shape: "
                f"{parameter.grad.shape}"
            )

            print(
                f"Gradient mean: "
                f"{parameter.grad.mean().item():.8f}"
            )

            break

    if not gradient_found:
        raise RuntimeError(
            "No gradients were produced."
        )

    print("\n" + "=" * 60)
    print("TRAINING STEP TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()