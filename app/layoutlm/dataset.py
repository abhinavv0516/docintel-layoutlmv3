"""
Dataset configuration for document classification.
"""

from pathlib import Path

from app.layoutlm.labels import DOCUMENT_CLASSES


DATASET_ROOT = Path("data")


TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"


def get_class_directory(
    split: str,
    class_name: str,
) -> Path:
    """
    Return the directory for a specific class and split.
    """

    if class_name not in DOCUMENT_CLASSES:
        raise ValueError(
            f"Unknown document class: {class_name}"
        )

    split_directories = {
        "train": TRAIN_DIR,
        "validation": VALIDATION_DIR,
        "test": TEST_DIR,
    }

    if split not in split_directories:
        raise ValueError(
            f"Unknown split: {split}"
        )

    return split_directories[split] / class_name