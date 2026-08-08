"""
Test the LayoutLMv3 document classification head.
"""

import sys
from pathlib import Path

import cv2
import torch

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.classifier import DocumentClassifier
from app.layoutlm.model import DocumentModel
from app.layoutlm.processor import DocumentProcessor
from app.ocr.engine import OCREngine


IMAGE_PATH = "data/uploads/Screenshot 2026-07-22 112154.png"


def normalize_box(box, width, height):
    """
    Normalize bounding box coordinates to [0, 1000].
    """

    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    processor = DocumentProcessor().get_processor()

    model = DocumentModel().get_model()

    classifier = DocumentClassifier(
        hidden_size=768,
        num_classes=5,
    )

    model.to(device)
    classifier.to(device)

    model.eval()
    classifier.eval()

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

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
    )

    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**encoding)

        document_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        logits = classifier(document_embedding)

    print("=" * 60)
    print("CLASSIFICATION OUTPUT")
    print("=" * 60)

    print(f"Embedding shape: {document_embedding.shape}")
    print(f"Logits shape: {logits.shape}")

    print("\nLogits:")
    print(logits)

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    print("\nProbabilities:")
    print(probabilities)

    predicted_class = torch.argmax(
        probabilities,
        dim=-1,
    )

    print("\nPredicted class index:")
    print(predicted_class.item())


if __name__ == "__main__":
    main()