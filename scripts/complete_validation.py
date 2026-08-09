"""
Complete the validation split using images from training data.
"""

import sys
import shutil
from pathlib import Path

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.labels import DOCUMENT_CLASSES


TRAIN_ROOT = Path("data/train")
VALIDATION_ROOT = Path("data/validation")

TARGET_VALIDATION_COUNT = 100


def main():

    print("=" * 60)
    print("COMPLETING VALIDATION DATASET")
    print("=" * 60)

    for class_name in DOCUMENT_CLASSES:

        train_dir = TRAIN_ROOT / class_name
        validation_dir = VALIDATION_ROOT / class_name

        validation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        validation_files = list(
            validation_dir.glob("*.png")
        )

        current_count = len(validation_files)

        needed = (
            TARGET_VALIDATION_COUNT
            - current_count
        )

        print(
            f"\n{class_name}: "
            f"{current_count}/"
            f"{TARGET_VALIDATION_COUNT}"
        )

        if needed <= 0:
            print("Already complete.")
            continue

        train_files = sorted(
            train_dir.glob("*.png")
        )

        validation_names = {
            file.name
            for file in validation_files
        }

        available_files = [
            file
            for file in train_files
            if file.name not in validation_names
        ]

        selected_files = available_files[:needed]

        for source in selected_files:

            destination = (
                validation_dir / source.name
            )

            shutil.move(
                source,
                destination,
            )

        print(
            f"Added {len(selected_files)} images."
        )

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()