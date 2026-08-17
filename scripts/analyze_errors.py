"""
Analyze OCR quality for misclassified test documents.

Reads checkpoints/test_errors.json, runs Tesseract on each
misclassified image, and reports OCR word count and confidence.
"""

import json
from pathlib import Path

import cv2
import pytesseract


ERRORS_FILE = Path(
    "checkpoints/test_errors.json"
)

OUTPUT_FILE = Path(
    "checkpoints/error_ocr_analysis.json"
)


def analyze_image(image_path):
    """Run Tesseract and calculate OCR statistics."""

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return {
            "ocr_words": 0,
            "ocr_average_confidence": 0.0,
        }

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []

    for text, confidence in zip(
        data["text"],
        data["conf"],
    ):

        text = text.strip()

        try:
            confidence = float(
                confidence
            )
        except (TypeError, ValueError):
            continue

        if not text:
            continue

        # Tesseract uses -1 for invalid/empty confidence.
        if confidence < 0:
            continue

        words.append(text)
        confidences.append(
            confidence
        )

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    return {
        "ocr_words": len(words),
        "ocr_average_confidence": (
            average_confidence
        ),
    }


def main():

    print("=" * 70)
    print("OCR ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # Load errors
    # --------------------------------------------------

    with open(
        ERRORS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        errors = json.load(file)

    print(
        f"\nMisclassified documents: {len(errors)}"
    )

    results = []

    # --------------------------------------------------
    # Analyze every error
    # --------------------------------------------------

    for index, error in enumerate(
        errors,
        start=1,
    ):

        image_path = Path(
            error["image_path"]
        )

        print(
            f"\n[{index}/{len(errors)}] "
            f"{image_path.name}"
        )

        if not image_path.exists():

            print(
                "  WARNING: image not found"
            )

            ocr_result = {
                "ocr_words": 0,
                "ocr_average_confidence": 0.0,
            }

        else:

            ocr_result = analyze_image(
                image_path
            )

        result = {
            "image_path": str(
                image_path
            ),
            "actual": error[
                "actual"
            ],
            "predicted": error[
                "predicted"
            ],
            "model_confidence": error[
                "confidence"
            ],
            "ocr_words": ocr_result[
                "ocr_words"
            ],
            "ocr_average_confidence": (
                ocr_result[
                    "ocr_average_confidence"
                ]
            ),
        }

        results.append(result)

        print(
            f"  Actual: {result['actual']}"
        )

        print(
            f"  Predicted: {result['predicted']}"
        )

        print(
            f"  Model confidence: "
            f"{result['model_confidence'] * 100:.2f}%"
        )

        print(
            f"  OCR words: "
            f"{result['ocr_words']}"
        )

        print(
            f"  OCR average confidence: "
            f"{result['ocr_average_confidence']:.2f}%"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    word_counts = [
        result["ocr_words"]
        for result in results
    ]

    ocr_confidences = [
        result[
            "ocr_average_confidence"
        ]
        for result in results
        if result["ocr_words"] > 0
    ]

    average_words = (
        sum(word_counts)
        / len(word_counts)
        if word_counts
        else 0.0
    )

    average_ocr_confidence = (
        sum(ocr_confidences)
        / len(ocr_confidences)
        if ocr_confidences
        else 0.0
    )

    zero_ocr = sum(
        1
        for count in word_counts
        if count == 0
    )

    low_ocr = sum(
        1
        for count in word_counts
        if count < 10
    )

    low_confidence_ocr = sum(
        1
        for result in results
        if (
            result["ocr_words"] > 0
            and result[
                "ocr_average_confidence"
            ] < 60
        )
    )

    summary = {
        "total_errors": len(results),
        "average_ocr_words": average_words,
        "average_ocr_confidence": (
            average_ocr_confidence
        ),
        "zero_ocr_documents": zero_ocr,
        "under_10_ocr_words": low_ocr,
        "low_ocr_confidence_documents": (
            low_confidence_ocr
        ),
    }

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------

    output = {
        "summary": summary,
        "errors": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("OCR SUMMARY")
    print("=" * 70)

    print(
        f"Total errors:              "
        f"{len(results)}"
    )

    print(
        f"Average OCR words:         "
        f"{average_words:.2f}"
    )

    print(
        f"Average OCR confidence:    "
        f"{average_ocr_confidence:.2f}%"
    )

    print(
        f"Zero OCR words:             "
        f"{zero_ocr}"
    )

    print(
        f"Under 10 OCR words:         "
        f"{low_ocr}"
    )

    print(
        f"Low OCR confidence (<60%): "
        f"{low_confidence_ocr}"
    )

    print("\nResults saved to:")

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)
    print("OCR ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()