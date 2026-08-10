"""
Rebuild the document classification dataset from RVL-CDIP.

Source:
    hf-tuner/rvl-cdip-document-classification

Output:
    data/train       -> 400 images per class
    data/validation  -> 100 images per class
    data/test        -> 62 images per class
"""

import sys
from pathlib import Path

from datasets import load_dataset

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.labels import DOCUMENT_CLASSES


DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

TRAIN_PER_CLASS = 400
VALIDATION_PER_CLASS = 100
TEST_PER_CLASS = 62

TRAIN_ROOT = Path("data/train")
VALIDATION_ROOT = Path("data/validation")
TEST_ROOT = Path("data/test")

TARGET_LABELS = {
    1: "form",
    4: "advertisement",
    10: "budget",
    11: "invoice",
    14: "resume",
}


def prepare_directories():
    """Create clean class directories."""

    for root in [
        TRAIN_ROOT,
        VALIDATION_ROOT,
        TEST_ROOT,
    ]:
        for class_name in DOCUMENT_CLASSES:

            class_dir = root / class_name

            class_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Remove old PNG files
            for file in class_dir.glob("*.png"):
                file.unlink()


def save_image(image, output_path):
    """Save PIL image."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)


def collect_train_and_validation():
    """Build train and validation from source train split."""

    print("=" * 60)
    print("LOADING SOURCE TRAIN SPLIT")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    counts = {
        class_name: 0
        for class_name in DOCUMENT_CLASSES
    }

    completed = set()

    print("\nCollecting train + validation images...\n")

    for sample in dataset:

        label_id = sample["label"]

        if label_id not in TARGET_LABELS:
            continue

        class_name = TARGET_LABELS[label_id]

        current_count = counts[class_name]

        if current_count >= (
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS
        ):
            continue

        image = sample["image"]

        if current_count < TRAIN_PER_CLASS:

            output_path = (
                TRAIN_ROOT
                / class_name
                / f"{class_name}_{current_count:04d}.png"
            )

            save_image(
                image,
                output_path,
            )

        else:

            validation_index = (
                current_count
                - TRAIN_PER_CLASS
            )

            output_path = (
                VALIDATION_ROOT
                / class_name
                / f"{class_name}_{validation_index:04d}.png"
            )

            save_image(
                image,
                output_path,
            )

        counts[class_name] += 1

        if current_count < TRAIN_PER_CLASS:

            print(
                f"train/{class_name}: "
                f"{min(counts[class_name], TRAIN_PER_CLASS)}"
                f"/{TRAIN_PER_CLASS}"
            )

        else:

            print(
                f"validation/{class_name}: "
                f"{counts[class_name] - TRAIN_PER_CLASS}"
                f"/{VALIDATION_PER_CLASS}"
            )

        if counts[class_name] == (
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS
        ):
            completed.add(class_name)

        if len(completed) == len(DOCUMENT_CLASSES):
            break

    print("\nTrain/validation collection complete.")


def collect_test():
    """Build test split from source test split."""

    print("\n" + "=" * 60)
    print("LOADING SOURCE TEST SPLIT")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True,
    )

    counts = {
        class_name: 0
        for class_name in DOCUMENT_CLASSES
    }

    completed = set()

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

        save_image(
            image,
            output_path,
        )

        counts[class_name] += 1

        print(
            f"test/{class_name}: "
            f"{counts[class_name]}/{TEST_PER_CLASS}"
        )

        if counts[class_name] == TEST_PER_CLASS:
            completed.add(class_name)

        if len(completed) == len(DOCUMENT_CLASSES):
            break

    print("\nTest collection complete.")


def print_summary():
    """Print final dataset counts."""

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    for split_name, root in [
        ("TRAIN", TRAIN_ROOT),
        ("VALIDATION", VALIDATION_ROOT),
        ("TEST", TEST_ROOT),
    ]:

        print(f"\n{split_name}")

        total = 0

        for class_name in DOCUMENT_CLASSES:

            count = len(
                list(
                    (root / class_name).glob("*.png")
                )
            )

            total += count

            print(
                f"{class_name:15}: {count}"
            )

        print(
            f"{'TOTAL':15}: {total}"
        )


def main():

    print("=" * 60)
    print("RVL-CDIP DATASET REBUILD")
    print("=" * 60)

    print(
        "\nWARNING: Existing PNG files inside "
        "data/train, data/validation and data/test "
        "class folders will be removed."
    )

    response = input(
        "\nType REBUILD to continue: "
    )

    if response != "REBUILD":
        print("\nCancelled.")
        return

    print("\nPreparing directories...")
    prepare_directories()

    collect_train_and_validation()
    collect_test()

    print_summary()

    print("\n" + "=" * 60)
    print("DATASET REBUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()