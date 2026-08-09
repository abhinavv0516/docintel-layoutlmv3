"""
Test OCR processing for a dataset document.
"""

import sys
from pathlib import Path

import cv2

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.labels import DOCUMENT_CLASSES
from app.ocr.engine import OCREngine


DATASET_ROOT = Path("data/train")


def main():

    print("=" * 60)
    print("DOCUMENT DATASET OCR TEST")
    print("=" * 60)

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

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            "Failed to load image."
        )

    height, width = image.shape[:2]

    print("\nImage information:")
    print(f"Width : {width}")
    print(f"Height: {height}")

    # Create OCR engine
    engine = OCREngine()

    # Extract OCR metadata
    data = engine.extract_data(
        str(image_path)
    )

    words = []
    boxes = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if text == "":
            continue

        left = data["left"][i]
        top = data["top"][i]
        box_width = data["width"][i]
        box_height = data["height"][i]

        words.append(text)

        boxes.append(
            [
                left,
                top,
                left + box_width,
                top + box_height,
            ]
        )

    print("\nOCR results:")
    print(f"Total words: {len(words)}")

    print("\nFirst 10 words:")
    print(words[:10])

    print("\nFirst 10 bounding boxes:")

    for box in boxes[:10]:
        print(box)

    print("\n" + "=" * 60)
    print("OCR DATASET SAMPLE SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()