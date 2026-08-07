"""
Test LayoutLMv3 Processor.
"""

import sys
from pathlib import Path

import cv2

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.processor import DocumentProcessor
from app.ocr.engine import OCREngine


IMAGE_PATH = "data/uploads/Screenshot 2026-07-22 112154.png"


def main():

    print("Starting processor test...\n")

    processor = DocumentProcessor().get_processor()

    engine = OCREngine()

    data = engine.extract_data(IMAGE_PATH)

    image = cv2.imread(IMAGE_PATH)

    words = []
    boxes = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if text == "":
            continue

        words.append(text)

        boxes.append(
            [
                data["left"][i],
                data["top"][i],
                data["left"][i] + data["width"][i],
                data["top"][i] + data["height"][i],
            ]
        )

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
    )

    print("=" * 60)
    print("ENCODING KEYS")
    print("=" * 60)
    print(encoding.keys())

    print()

    print("=" * 60)
    print("INPUT SHAPES")
    print("=" * 60)

    for key, value in encoding.items():
        print(f"{key}: {value.shape}")


if __name__ == "__main__":
    main()