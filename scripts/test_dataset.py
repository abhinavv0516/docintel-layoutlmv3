"""
Test document dataset configuration.
"""

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.dataset import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    get_class_directory,
)
from app.layoutlm.labels import DOCUMENT_CLASSES


def main():

    print("=" * 60)
    print("DOCUMENT DATASET CONFIGURATION")
    print("=" * 60)

    print("\nClasses:")

    for index, class_name in enumerate(DOCUMENT_CLASSES):
        print(f"{index}: {class_name}")

    print("\nDataset directories:")

    print(f"Train:      {TRAIN_DIR}")
    print(f"Validation: {VALIDATION_DIR}")
    print(f"Test:       {TEST_DIR}")

    print("\nClass directories:")

    for class_name in DOCUMENT_CLASSES:

        print(
            f"{class_name}: "
            f"{get_class_directory('train', class_name)}"
        )


if __name__ == "__main__":
    main()