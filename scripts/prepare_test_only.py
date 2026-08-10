"""
Prepare only the test split from RVL-CDIP.

Does NOT modify train or validation data.
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


def main():

    print("=" * 60)
    print("RVL-CDIP TEST DATASET PREPARATION")
    print("=" * 60)

    print("\nThis script will NOT modify train or validation.")

    for class_name in DOCUMENT_CLASSES:

        class_dir = TEST_ROOT / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Remove existing test images only
        for file in class_dir.glob("*.png"):
            file.unlink()

    print("\nLoading source test split...")

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

        image = sample["image"]

        image_number = counts[class_name]

        output_path = (
            TEST_ROOT
            / class_name
            / f"{class_name}_{image_number:04d}.png"
        )

        image.save(output_path)

        counts[class_name] += 1

        print(
            f"test/{class_name}: "
            f"{counts[class_name]}/{TEST_PER_CLASS}"
        )

        if all(
            count == TEST_PER_CLASS
            for count in counts.values()
        ):
            break

    print("\n" + "=" * 60)
    print("TEST DATASET SUMMARY")
    print("=" * 60)

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")

    print("=" * 60)


if __name__ == "__main__":
    main()