"""
Test LayoutLMv3 processor with no OCR words.
"""

import sys
from pathlib import Path

import cv2

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.processor import DocumentProcessor


IMAGE_PATH = (
    "data/test/advertisement/"
    "advertisement_0052.png"
)


def main():

    print("=" * 60)
    print("EMPTY OCR FALLBACK TEST")
    print("=" * 60)

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(
            f"Could not load image: {IMAGE_PATH}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    print(
        f"Image shape: {image.shape}"
    )

    processor = (
        DocumentProcessor()
        .get_processor()
    )

    words = []
    boxes = []

    print("\nRunning processor with:")
    print(f"Words: {words}")
    print(f"Boxes: {boxes}")

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    print("\n" + "=" * 60)
    print("PROCESSOR OUTPUT")
    print("=" * 60)

    for key, value in encoding.items():

        print(
            f"{key:20} "
            f"shape={tuple(value.shape)}"
        )

    print("\n" + "=" * 60)
    print("EMPTY OCR FALLBACK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()