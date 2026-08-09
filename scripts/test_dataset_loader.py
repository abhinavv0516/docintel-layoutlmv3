"""
Test loading a document classification sample.
"""

import sys
from pathlib import Path

import cv2

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.labels import DOCUMENT_CLASSES


DATASET_ROOT = Path("data/train")


def main():

    print("=" * 60)
    print("DOCUMENT DATASET LOADER TEST")
    print("=" * 60)

    # Find the first image from the first class
    class_name = DOCUMENT_CLASSES[0]

    class_dir = DATASET_ROOT / class_name

    image_files = list(
        class_dir.glob("*.png")
    )

    if not image_files:
        raise FileNotFoundError(
            f"No images found in {class_dir}"
        )

    image_path = image_files[0]

    print("\nImage:")
    print(image_path)

    # Load image
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            "Failed to load image."
        )

    # Determine class index
    class_index = DOCUMENT_CLASSES.index(
        class_name
    )

    print("\nClass information:")
    print(f"Class name : {class_name}")
    print(f"Class index: {class_index}")

    print("\nImage information:")
    print(f"Shape      : {image.shape}")
    print(f"Height     : {image.shape[0]}")
    print(f"Width      : {image.shape[1]}")
    print(f"Channels   : {image.shape[2]}")

    print("\n" + "=" * 60)
    print("DATASET SAMPLE LOADED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()