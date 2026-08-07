"""
Test LayoutLMv3 forward pass.
"""

import sys
from pathlib import Path

import cv2
import torch

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.model import DocumentModel
from app.layoutlm.processor import DocumentProcessor
from app.ocr.engine import OCREngine

IMAGE_PATH = "data/uploads/Screenshot 2026-07-22 112154.png"


def normalize_box(box, width, height):
    """
    Normalize bounding box coordinates to the range [0, 1000].
    """

    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def main():

    print("Loading processor...")
    processor = DocumentProcessor().get_processor()

    print("Loading model...")
    model = DocumentModel().get_model()

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Move model to GPU
    model.to(device)

    # Evaluation mode
    model.eval()

    engine = OCREngine()

    data = engine.extract_data(IMAGE_PATH)

    image = cv2.imread(IMAGE_PATH)

    image_height, image_width = image.shape[:2]

    words = []
    boxes = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if text == "":
            continue

        words.append(text)

        box = [
            data["left"][i],
            data["top"][i],
            data["left"][i] + data["width"][i],
            data["top"][i] + data["height"][i],
        ]

        boxes.append(
            normalize_box(
                box,
                image_width,
                image_height,
            )
        )

    print("\nFirst 5 normalized boxes:")

    for box in boxes[:5]:
        print(box)

    print("\nMaximum coordinate:")
    print(max(max(box) for box in boxes))

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
    )

    # Move tensors to GPU
    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    print("\nRunning inference...")

    with torch.no_grad():
        outputs = model(**encoding)

    print("\n" + "=" * 60)
    print("MODEL OUTPUT")
    print("=" * 60)

    print(f"Device: {outputs.last_hidden_state.device}")
    print(f"Shape : {outputs.last_hidden_state.shape}")

    print()

    print("First token embedding (first 10 values):")

    print(outputs.last_hidden_state[0, 0, :10])


if __name__ == "__main__":
    main()