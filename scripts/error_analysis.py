"""
Analyze incorrect LayoutLMv3 predictions.

Uses the best grayscale checkpoint and the clean
grayscale test dataset.
"""

import json
import sys
from collections import Counter
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


# --------------------------------------------------
# Grayscale experiment paths
# --------------------------------------------------

MODEL_PATH = (
    "checkpoints/grayscale/best_model"
)

TEST_DIR = (
    "data/processed_gray/test"
)

BATCH_SIZE = 4


def main():

    print("=" * 60)
    print("LAYOUTLMV3 GRAYSCALE ERROR ANALYSIS")
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

    print("\nLoading grayscale best model...")

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
    # Load grayscale test dataset
    # --------------------------------------------------

    print(
        "\nLoading grayscale test dataset..."
    )

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
    # Run inference
    # --------------------------------------------------

    errors = []

    total = 0
    correct = 0

    print("\nRunning inference...")

    with torch.no_grad():

        for batch in dataloader:

            image_paths = batch.get(
                "image_path"
            )

            model_batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
                if key != "image_path"
            }

            outputs = model(
                input_ids=model_batch[
                    "input_ids"
                ],
                attention_mask=model_batch[
                    "attention_mask"
                ],
                bbox=model_batch[
                    "bbox"
                ],
                pixel_values=model_batch[
                    "pixel_values"
                ],
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            predictions = (
                probabilities.argmax(
                    dim=-1
                )
            )

            confidences = (
                probabilities.max(
                    dim=-1
                ).values
            )

            labels = model_batch[
                "labels"
            ]

            for index in range(
                labels.size(0)
            ):

                true_label = (
                    labels[index].item()
                )

                predicted_label = (
                    predictions[index].item()
                )

                confidence = (
                    confidences[index].item()
                )

                total += 1

                if (
                    true_label
                    == predicted_label
                ):

                    correct += 1
                    continue

                if image_paths is not None:

                    image_path = (
                        image_paths[index]
                    )

                else:

                    image_path = (
                        dataset.samples[
                            total - 1
                        ][0]
                    )

                errors.append(
                    {
                        "image_path": str(
                            image_path
                        ),
                        "actual": DOCUMENT_CLASSES[
                            true_label
                        ],
                        "predicted": DOCUMENT_CLASSES[
                            predicted_label
                        ],
                        "confidence": round(
                            confidence,
                            6,
                        ),
                    }
                )

    # --------------------------------------------------
    # Overall summary
    # --------------------------------------------------

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print(
        "\n" + "=" * 60
    )
    print("ERROR SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal test samples: {total}"
    )

    print(
        f"Correct:            {correct}"
    )

    print(
        f"Incorrect:          {len(errors)}"
    )

    print(
        f"Accuracy:            "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------
    # Group errors by confusion pair
    # --------------------------------------------------

    confusion_counts = Counter(
        (
            error["actual"],
            error["predicted"],
        )
        for error in errors
    )

    print(
        "\n" + "=" * 60
    )
    print("ERROR GROUPS")
    print("=" * 60)

    for (
        actual,
        predicted,
    ), count in confusion_counts.most_common():

        print(
            f"{actual:18} -> "
            f"{predicted:18} : "
            f"{count}"
        )

    # --------------------------------------------------
    # Lowest-confidence errors
    # --------------------------------------------------

    lowest_confidence = sorted(
        errors,
        key=lambda item: item[
            "confidence"
        ],
    )

    print(
        "\n" + "=" * 60
    )
    print("LOWEST-CONFIDENCE ERRORS")
    print("=" * 60)

    for error in lowest_confidence[:10]:

        print(
            f"\nActual:     "
            f"{error['actual']}"
        )

        print(
            f"Predicted:  "
            f"{error['predicted']}"
        )

        print(
            f"Confidence: "
            f"{error['confidence']:.4f}"
        )

        print(
            f"Image:      "
            f"{error['image_path']}"
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = {
        "experiment": "grayscale",
        "model_path": MODEL_PATH,
        "test_dir": TEST_DIR,
        "test_samples": total,
        "correct": correct,
        "incorrect": len(errors),
        "accuracy": accuracy,
        "error_groups": [
            {
                "actual": actual,
                "predicted": predicted,
                "count": count,
            }
            for (
                actual,
                predicted,
            ), count in confusion_counts.most_common()
        ],
        "errors": errors,
    }

    output_path = Path(
        "checkpoints/grayscale/error_analysis.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "GRAYSCALE ERROR ANALYSIS COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()