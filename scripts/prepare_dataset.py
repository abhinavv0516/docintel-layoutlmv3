"""
Prepare validation and test datasets from RVL-CDIP.
"""

import sys
from pathlib import Path

from datasets import load_dataset

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.labels import DOCUMENT_CLASSES


DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

VALIDATION_PER_CLASS = 100
TEST_PER_CLASS = 100

VALIDATION_ROOT = Path("data/validation")
TEST_ROOT = Path("data/test")


TARGET_LABELS = {
    1: "form",
    4: "advertisement",
    10: "budget",
    11: "invoice",
    14: "resume",
}


def save_image(image, output_path):
    """
    Save a PIL image to disk.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)


def main():

    print("=" * 60)
    print("RVL-CDIP VALIDATION / TEST PREPARATION")
    print("=" * 60)

    print("\nLoading dataset test split...")

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True,
    )

    print("Dataset loaded.")

    validation_counts = {
        class_name: 0
        for class_name in DOCUMENT_CLASSES
    }

    test_counts = {
        class_name: 0
        for class_name in DOCUMENT_CLASSES
    }

    print("\nCollecting validation and test images...\n")

    for sample in dataset:

        label_id = sample["label"]

        if label_id not in TARGET_LABELS:
            continue

        class_name = TARGET_LABELS[label_id]

        # First 100 → validation
        if validation_counts[class_name] < VALIDATION_PER_CLASS:

            image_number = validation_counts[class_name]

            output_path = (
                VALIDATION_ROOT
                / class_name
                / f"{class_name}_{image_number:04d}.png"
            )

            save_image(
                sample["image"],
                output_path,
            )

            validation_counts[class_name] += 1

            print(
                f"validation/{class_name}: "
                f"{validation_counts[class_name]}/{VALIDATION_PER_CLASS}"
            )

            continue

        # Next 100 → test
        if test_counts[class_name] < TEST_PER_CLASS:

            image_number = test_counts[class_name]

            output_path = (
                TEST_ROOT
                / class_name
                / f"{class_name}_{image_number:04d}.png"
            )

            save_image(
                sample["image"],
                output_path,
            )

            test_counts[class_name] += 1

            print(
                f"test/{class_name}: "
                f"{test_counts[class_name]}/{TEST_PER_CLASS}"
            )

        # Stop when everything is complete
        if (
            all(
                count == VALIDATION_PER_CLASS
                for count in validation_counts.values()
            )
            and
            all(
                count == TEST_PER_CLASS
                for count in test_counts.values()
            )
        ):
            print("\nAll validation and test classes complete.")
            break

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for class_name, count in validation_counts.items():
        print(f"{class_name}: {count}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for class_name, count in test_counts.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()