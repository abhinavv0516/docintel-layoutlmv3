"""
Build a leakage-free, balanced RVL-CDIP dataset.

Source:
    RVL-CDIP train + test

Global SHA-256 deduplication is performed first.

Final split:
    350 train / class
     73 validation / class
     73 test / class

Total:
    1750 train
     365 validation
     365 test
    2480 images
"""

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = (
    "hf-tuner/rvl-cdip-document-classification"
)

OUTPUT_ROOT = Path("data/clean")

SEED = 42

TRAIN_PER_CLASS = 350
VALIDATION_PER_CLASS = 73
TEST_PER_CLASS = 73

TARGET_LABELS = {
    1: "form",
    4: "advertisement",
    10: "budget",
    11: "invoice",
    14: "resume",
}


def image_hash(image):
    """Return SHA-256 hash of image pixels."""

    image = image.convert("RGB")

    return hashlib.sha256(
        image.tobytes()
    ).hexdigest()


def save_image(image, path):
    """Save image as PNG."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.convert("RGB").save(path)


def main():

    print("=" * 60)
    print("BUILDING CLEAN RVL-CDIP DATASET")
    print("=" * 60)

    required_per_class = (
        TRAIN_PER_CLASS
        + VALIDATION_PER_CLASS
        + TEST_PER_CLASS
    )

    print(
        f"\nRequired unique images/class: "
        f"{required_per_class}"
    )

    print(
        f"Train:      {TRAIN_PER_CLASS}"
    )

    print(
        f"Validation: {VALIDATION_PER_CLASS}"
    )

    print(
        f"Test:       {TEST_PER_CLASS}"
    )

    print(
        f"Seed:       {SEED}"
    )

    # --------------------------------------------------
    # Collect globally unique samples
    # --------------------------------------------------

    unique_samples = defaultdict(list)

    seen_hashes = set()

    duplicate_count = 0

    for split_name in ["train", "test"]:

        print("\n" + "=" * 60)
        print(
            f"READING SOURCE {split_name.upper()}"
        )
        print("=" * 60)

        dataset = load_dataset(
            DATASET_NAME,
            split=split_name,
            streaming=True,
        )

        for sample in dataset:

            label_id = sample["label"]

            if label_id not in TARGET_LABELS:
                continue

            class_name = TARGET_LABELS[
                label_id
            ]

            image = sample["image"]

            digest = image_hash(image)

            if digest in seen_hashes:

                duplicate_count += 1
                continue

            seen_hashes.add(digest)

            unique_samples[
                class_name
            ].append(
                (
                    digest,
                    image,
                )
            )

        print(
            f"Unique samples collected so far: "
            f"{len(seen_hashes)}"
        )

    # --------------------------------------------------
    # Verify availability
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("UNIQUE SOURCE POOL")
    print("=" * 60)

    for class_name in TARGET_LABELS.values():

        count = len(
            unique_samples[class_name]
        )

        print(
            f"{class_name:18}: {count}"
        )

        if count < required_per_class:

            raise RuntimeError(
                f"Not enough unique images for "
                f"{class_name}: "
                f"{count} available, "
                f"{required_per_class} required."
            )

    # --------------------------------------------------
    # Clear previous clean dataset
    # --------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous incomplete "
            "clean dataset..."
        )

        import shutil

        shutil.rmtree(
            OUTPUT_ROOT
        )

    # --------------------------------------------------
    # Deterministic balanced splitting
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CREATING BALANCED SPLITS")
    print("=" * 60)

    rng = random.Random(SEED)

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    final_hashes = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for class_name in TARGET_LABELS.values():

        samples = unique_samples[
            class_name
        ]

        # Deterministic shuffle
        rng.shuffle(samples)

        required = (
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS
            + TEST_PER_CLASS
        )

        selected = samples[:required]

        train_samples = selected[
            :TRAIN_PER_CLASS
        ]

        validation_samples = selected[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS
        ]

        test_samples = selected[
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS:
        ]

        # --------------------------------------------------
        # Save train
        # --------------------------------------------------

        for index, (digest, image) in enumerate(
            train_samples
        ):

            path = (
                OUTPUT_ROOT
                / "train"
                / class_name
                / f"{class_name}_{index:04d}.png"
            )

            save_image(
                image,
                path,
            )

            final_hashes[
                "train"
            ].add(digest)

            split_counts[
                "train"
            ] += 1

        # --------------------------------------------------
        # Save validation
        # --------------------------------------------------

        for index, (digest, image) in enumerate(
            validation_samples
        ):

            path = (
                OUTPUT_ROOT
                / "validation"
                / class_name
                / f"{class_name}_{index:04d}.png"
            )

            save_image(
                image,
                path,
            )

            final_hashes[
                "validation"
            ].add(digest)

            split_counts[
                "validation"
            ] += 1

        # --------------------------------------------------
        # Save test
        # --------------------------------------------------

        for index, (digest, image) in enumerate(
            test_samples
        ):

            path = (
                OUTPUT_ROOT
                / "test"
                / class_name
                / f"{class_name}_{index:04d}.png"
            )

            save_image(
                image,
                path,
            )

            final_hashes[
                "test"
            ].add(digest)

            split_counts[
                "test"
            ] += 1

    # --------------------------------------------------
    # Final verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(
        f"\nTrain:      "
        f"{split_counts['train']}"
    )

    print(
        f"Validation: "
        f"{split_counts['validation']}"
    )

    print(
        f"Test:       "
        f"{split_counts['test']}"
    )

    expected_train = (
        TRAIN_PER_CLASS
        * len(TARGET_LABELS)
    )

    expected_validation = (
        VALIDATION_PER_CLASS
        * len(TARGET_LABELS)
    )

    expected_test = (
        TEST_PER_CLASS
        * len(TARGET_LABELS)
    )

    assert (
        split_counts["train"]
        == expected_train
    )

    assert (
        split_counts["validation"]
        == expected_validation
    )

    assert (
        split_counts["test"]
        == expected_test
    )

    # --------------------------------------------------
    # Cross-split duplicate verification
    # --------------------------------------------------

    train_validation = (
        final_hashes["train"]
        & final_hashes["validation"]
    )

    train_test = (
        final_hashes["train"]
        & final_hashes["test"]
    )

    validation_test = (
        final_hashes["validation"]
        & final_hashes["test"]
    )

    print("\n" + "=" * 60)
    print("LEAKAGE VERIFICATION")
    print("=" * 60)

    print(
        f"\nTrain ↔ Validation: "
        f"{len(train_validation)}"
    )

    print(
        f"Train ↔ Test:       "
        f"{len(train_test)}"
    )

    print(
        f"Validation ↔ Test:  "
        f"{len(validation_test)}"
    )

    if (
        train_validation
        or train_test
        or validation_test
    ):

        raise RuntimeError(
            "Cross-split duplicates detected!"
        )

    print(
        "\nNO CROSS-SPLIT DUPLICATES FOUND."
    )

    print(
        "\nDuplicate source images removed: "
        f"{duplicate_count}"
    )

    print(
        "\nClean dataset location:"
    )

    print(
        OUTPUT_ROOT.resolve()
    )

    print("\n" + "=" * 60)
    print("CLEAN DATASET BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()