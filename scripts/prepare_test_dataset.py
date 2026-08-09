"""
Prepare the test split from RVL-CDIP.
"""

import sys
from pathlib import Path

from datasets import load_dataset

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.labels import DOCUMENT_CLASSES


DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

TEST_PER_CLASS = 62

TEST_ROOT = Path("data/test")


TARGET_LABELS = {
    1: "form",
    4: "advertisement",
    10: "budget",
    11: "invoice",
    14: "resume",
}


def save_image(image, output_path):
    """Save a PIL image to disk."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)


def main():

    print("=" * 60)
    print("RVL-CDIP TEST DATASET PREPARATION")
    print("=" * 60)

    print("\nLoading test split...")

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True,
    )

    print("Dataset loaded.")

    counts = {
        class_name: 0
        for class_name in DOCUMENT_CLASSES
    }

    print("\nCollecting test images...\n")

    for sample in dataset:

        label_id = sample["label"]

        if label_id not in TARGET_LABELS:
            continue

        class_name = TARGET_LABELS[label_id]

        if counts[class_name] >= TEST_PER_CLASS:
            continue

        image_number = counts[class_name]

        output_path = (
            TEST_ROOT
            / class_name
            / f"{class_name}_{image_number:04d}.png"
        )

        save_image(
            sample["image"],
            output_path,
        )

        counts[class_name] += 1

        print(
            f"{class_name}: "
            f"{counts[class_name]}/{TEST_PER_CLASS}"
        )

        if all(
            count == TEST_PER_CLASS
            for count in counts.values()
        ):
            print("\nAll test classes complete.")
            break

    print("\n" + "=" * 60)
    print("TEST DATASET SUMMARY")
    print("=" * 60)

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()