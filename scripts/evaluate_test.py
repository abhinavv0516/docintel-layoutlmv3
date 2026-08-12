"""
Evaluate the best LayoutLMv3 model on the test dataset.
"""

import json
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
from app.layoutlm.labels import DOCUMENT_CLASSES

MODEL_PATH = "checkpoints/grayscale/best_model"
TEST_DIR = "data/processed_gray/test"
BATCH_SIZE = 4


def main():

    print("=" * 60)
    print("LAYOUTLMV3 TEST EVALUATION")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading best model...")

    model = (
        LayoutLMv3ForSequenceClassification
        .from_pretrained(
            MODEL_PATH
        )
    )

    model.to(device)
    model.eval()

    print("Model loaded.")

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------

    print("\nLoading test dataset...")

    dataset = CachedDocumentDataset(
        TEST_DIR
    )

    print(
        f"Test samples: {len(dataset)}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    num_classes = len(
        DOCUMENT_CLASSES
    )

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    total_loss = 0.0
    total = 0
    correct = 0

    print("\nRunning test evaluation...")

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                bbox=batch["bbox"],
                pixel_values=batch[
                    "pixel_values"
                ],
                labels=batch["labels"],
            )

            predictions = (
                outputs.logits.argmax(
                    dim=-1
                )
            )

            labels = batch["labels"]

            batch_size = labels.size(0)

            total_loss += (
                outputs.loss.item()
                * batch_size
            )

            total += batch_size

            correct += (
                predictions == labels
            ).sum().item()

            # Build confusion matrix
            for true_label, predicted_label in zip(
                labels.cpu(),
                predictions.cpu(),
            ):

                confusion_matrix[
                    true_label,
                    predicted_label,
                ] += 1

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    test_loss = total_loss / total
    accuracy = correct / total

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Test loss:     {test_loss:.4f}"
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
        f" ({accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PER-CLASS RESULTS")
    print("=" * 60)

    metrics = {}

    for class_index, class_name in enumerate(
        DOCUMENT_CLASSES
    ):

        true_positive = (
            confusion_matrix[
                class_index,
                class_index,
            ].item()
        )

        actual = (
            confusion_matrix[
                class_index
            ].sum().item()
        )

        predicted = (
            confusion_matrix[
                :,
                class_index,
            ].sum().item()
        )

        precision = (
            true_positive / predicted
            if predicted > 0
            else 0.0
        )

        recall = (
            true_positive / actual
            if actual > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }

        print(
            f"\n{class_name}"
        )

        print(
            f"  Precision: {precision:.4f}"
        )

        print(
            f"  Recall:    {recall:.4f}"
        )

        print(
            f"  F1:        {f1:.4f}"
        )

        print(
            f"  Support:   {actual}"
        )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        "\nRows = actual"
    )

    print(
        "Columns = predicted\n"
    )

    print(
        f"{'':18}",
        end="",
    )

    for class_name in DOCUMENT_CLASSES:

        print(
            f"{class_name[:12]:>14}",
            end="",
        )

    print()

    for index, class_name in enumerate(
        DOCUMENT_CLASSES
    ):

        print(
            f"{class_name:18}",
            end="",
        )

        for predicted_index in range(
            num_classes
        ):

            print(
                f"{confusion_matrix[index, predicted_index].item():>14}",
                end="",
            )

        print()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "per_class": metrics,
        "confusion_matrix": (
            confusion_matrix.tolist()
        ),
    }

    output_path = Path(
        "checkpoints/test_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\nResults saved to:")
    print(output_path)

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()