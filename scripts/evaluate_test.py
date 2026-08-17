"""
Evaluate the best LayoutLMv3 model on the test dataset.

The evaluation includes:

- Overall test loss
- Overall test accuracy
- Per-class precision
- Per-class recall
- Per-class F1 score
- Confusion matrix
- Misclassified document analysis
- Prediction confidence for errors
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForSequenceClassification


# --------------------------------------------------
# Project path
# --------------------------------------------------

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)


# --------------------------------------------------
# Application imports
# --------------------------------------------------

from app.layoutlm.cached_dataset import (
    CachedDocumentDataset,
)

from app.layoutlm.labels import (
    DOCUMENT_CLASSES,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = (
    "checkpoints/grayscale/best_model"
)

TEST_DIR = (
    "data/processed_gray/test"
)

BATCH_SIZE = 4

RESULTS_OUTPUT = Path(
    "checkpoints/test_results.json"
)

ERRORS_OUTPUT = Path(
    "checkpoints/test_errors.json"
)


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("LAYOUTLMV3 TEST EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print(
        "\nLoading best model..."
    )

    model = (
        LayoutLMv3ForSequenceClassification
        .from_pretrained(
            MODEL_PATH
        )
    )

    model.to(device)
    model.eval()

    print(
        "Model loaded."
    )

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------

    print(
        "\nLoading test dataset..."
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
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    # --------------------------------------------------
    # Evaluation variables
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

    # Store all misclassified documents
    errors = []

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    print(
        "\nRunning test evaluation..."
    )

    with torch.no_grad():

        for batch in dataloader:

            # ------------------------------------------
            # Move tensors to device
            #
            # image_path remains a string/list and
            # therefore stays on CPU.
            # ------------------------------------------

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }

            # ------------------------------------------
            # Forward pass
            # ------------------------------------------

            outputs = model(
                input_ids=batch[
                    "input_ids"
                ],
                attention_mask=batch[
                    "attention_mask"
                ],
                bbox=batch[
                    "bbox"
                ],
                pixel_values=batch[
                    "pixel_values"
                ],
                labels=batch[
                    "labels"
                ],
            )

            # ------------------------------------------
            # Predictions
            # ------------------------------------------

            predictions = (
                outputs.logits.argmax(
                    dim=-1
                )
            )

            labels = batch[
                "labels"
            ]

            batch_size = labels.size(
                0
            )

            # ------------------------------------------
            # Probabilities
            # ------------------------------------------

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            # ------------------------------------------
            # Record misclassified documents
            # ------------------------------------------

            for sample_index in range(
                batch_size
            ):

                true_label = labels[
                    sample_index
                ].item()

                predicted_label = (
                    predictions[
                        sample_index
                    ].item()
                )

                if (
                    true_label
                    != predicted_label
                ):

                    confidence = float(
                        probabilities[
                            sample_index,
                            predicted_label,
                        ].item()
                    )

                    errors.append(
                        {
                            "image_path": batch[
                                "image_path"
                            ][sample_index],

                            "actual": (
                                DOCUMENT_CLASSES[
                                    true_label
                                ]
                            ),

                            "predicted": (
                                DOCUMENT_CLASSES[
                                    predicted_label
                                ]
                            ),

                            "confidence": (
                                confidence
                            ),
                        }
                    )

            # ------------------------------------------
            # Loss
            # ------------------------------------------

            total_loss += (
                outputs.loss.item()
                * batch_size
            )

            # ------------------------------------------
            # Total samples
            # ------------------------------------------

            total += batch_size

            # ------------------------------------------
            # Correct predictions
            # ------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()

            # ------------------------------------------
            # Confusion matrix
            # ------------------------------------------

            for (
                true_label,
                predicted_label,
            ) in zip(
                labels.cpu(),
                predictions.cpu(),
            ):

                confusion_matrix[
                    true_label,
                    predicted_label,
                ] += 1

    # ==================================================
    # OVERALL METRICS
    # ==================================================

    test_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Test loss:     {test_loss:.4f}"
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
        f" ({accuracy * 100:.2f}%)"
    )

    print(
        f"Correct:       {correct}/{total}"
    )

    print(
        f"Incorrect:     {len(errors)}/{total}"
    )

    # ==================================================
    # PER-CLASS METRICS
    # ==================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PER-CLASS RESULTS"
    )

    print(
        "=" * 60
    )

    metrics = {}

    for (
        class_index,
        class_name,
    ) in enumerate(
        DOCUMENT_CLASSES
    ):

        # ----------------------------------------------
        # True positives
        # ----------------------------------------------

        true_positive = (
            confusion_matrix[
                class_index,
                class_index,
            ].item()
        )

        # ----------------------------------------------
        # Actual samples
        # ----------------------------------------------

        actual = (
            confusion_matrix[
                class_index
            ].sum().item()
        )

        # ----------------------------------------------
        # Predicted samples
        # ----------------------------------------------

        predicted = (
            confusion_matrix[
                :,
                class_index,
            ].sum().item()
        )

        # ----------------------------------------------
        # Precision
        # ----------------------------------------------

        precision = (
            true_positive / predicted
            if predicted > 0
            else 0.0
        )

        # ----------------------------------------------
        # Recall
        # ----------------------------------------------

        recall = (
            true_positive / actual
            if actual > 0
            else 0.0
        )

        # ----------------------------------------------
        # F1
        # ----------------------------------------------

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if (
                precision + recall
                > 0
            )
            else 0.0
        )

        metrics[
            class_name
        ] = {
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

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 60
    )

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

    for class_name in (
        DOCUMENT_CLASSES
    ):

        print(
            f"{class_name[:12]:>14}",
            end="",
        )

    print()

    for (
        index,
        class_name,
    ) in enumerate(
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

    # ==================================================
    # ERROR SUMMARY
    # ==================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "ERROR ANALYSIS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nMisclassified documents: "
        f"{len(errors)}"
    )

    for index, error in enumerate(
        errors,
        start=1,
    ):

        print(
            f"\n{index}. "
            f"{error['image_path']}"
        )

        print(
            f"   Actual:     "
            f"{error['actual']}"
        )

        print(
            f"   Predicted:  "
            f"{error['predicted']}"
        )

        print(
            f"   Confidence: "
            f"{error['confidence']:.4f}"
        )

    # ==================================================
    # SAVE COMPLETE RESULTS
    # ==================================================

    results = {
        "test_loss": test_loss,

        "test_accuracy": accuracy,

        "total_samples": total,

        "correct": correct,

        "incorrect": len(errors),

        "per_class": metrics,

        "confusion_matrix": (
            confusion_matrix.tolist()
        ),

        "errors": errors,
    }

    RESULTS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # ==================================================
    # SAVE ERROR LIST SEPARATELY
    # ==================================================

    with open(
        ERRORS_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            errors,
            file,
            indent=4,
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_OUTPUT
    )

    print(
        "\nError analysis saved to:"
    )

    print(
        ERRORS_OUTPUT
    )

    # ==================================================
    # COMPLETE
    # ==================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST EVALUATION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()