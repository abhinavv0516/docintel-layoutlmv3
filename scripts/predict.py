"""
Command-line document prediction.

The DocumentPredictor loads the trained model once
and performs inference on a single document.
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.append(
    str(PROJECT_ROOT)
)

from app.inference.predictor import (
    DocumentPredictor,
)


def main():

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python scripts\\predict.py "
            "<image_path>"
        )

        print(
            "\nExample:"
        )

        print(
            "python scripts\\predict.py "
            "data\\clean\\test\\invoice\\invoice_0034.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 60)
    print(
        "LAYOUTLMV3 DOCUMENT PREDICTION"
    )
    print("=" * 60)

    print(
        f"\nInput: {image_path}"
    )

    # Model + processor + OCR are loaded once.
    predictor = DocumentPredictor()

    print(
        "\nRunning prediction..."
    )

    result = predictor.predict(
        image_path
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"\nDocument type: "
        f"{result['document_type']}"
    )

    print(
        f"Confidence:    "
        f"{result['confidence']:.4f}"
        f" ({result['confidence'] * 100:.2f}%)"
    )

    print(
        f"OCR words:     "
        f"{result['ocr_words']}"
    )

    print(
        "\nClass probabilities:"
    )

    for (
        class_name,
        probability,
    ) in sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"  {class_name:16} "
            f"{probability:.4f}"
            f" ({probability * 100:.2f}%)"
        )

    print("\n" + "=" * 60)
    print(
        "PREDICTION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()