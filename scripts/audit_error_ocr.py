"""
Audit OCR quality on LayoutLMv3 classification errors.

Compares OCR statistics for the 39 incorrectly classified
clean test documents.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


ERROR_FILE = Path(
    "checkpoints/error_analysis.json"
)

OUTPUT_FILE = Path(
    "checkpoints/error_ocr_audit.json"
)


def analyze_image(image_path):
    """Run Tesseract and collect OCR statistics."""

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
    )

    words = []

    confidences = []

    boxes = 0

    for index, text in enumerate(
        data["text"]
    ):

        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(
                data["conf"][index]
            )
        except (
            ValueError,
            TypeError,
        ):
            confidence = -1

        words.append(text)

        if confidence >= 0:
            confidences.append(
                confidence
            )

        try:
            w = int(
                data["width"][index]
            )
            h = int(
                data["height"][index]
            )

            if w > 0 and h > 0:
                boxes += 1

        except (
            ValueError,
            TypeError,
        ):
            pass

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    return {
        "width": width,
        "height": height,
        "word_count": len(words),
        "box_count": boxes,
        "average_ocr_confidence": round(
            average_confidence,
            2,
        ),
        "ocr_text_preview": " ".join(
            words[:50]
        ),
    }


def main():

    print("=" * 60)
    print("OCR ERROR AUDIT")
    print("=" * 60)

    with open(
        ERROR_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    errors = results["errors"]

    print(
        f"\nIncorrect documents: "
        f"{len(errors)}"
    )

    audited = []

    print(
        "\nRunning Tesseract OCR on errors..."
    )

    for index, error in enumerate(
        errors,
        start=1,
    ):

        image_path = Path(
            error["image_path"]
        )

        try:

            ocr = analyze_image(
                image_path
            )

            record = {
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
                **ocr,
            }

            audited.append(record)

            print(
                f"{index:02d}/{len(errors)} "
                f"{error['actual']:15} -> "
                f"{error['predicted']:15} | "
                f"words={ocr['word_count']:4d} | "
                f"OCR conf="
                f"{ocr['average_ocr_confidence']:.1f}"
            )

        except Exception as exc:

            print(
                f"\nERROR: "
                f"{image_path}"
            )

            print(
                f"Reason: {exc}"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("OCR SUMMARY")
    print("=" * 60)

    if audited:

        word_counts = [
            item["word_count"]
            for item in audited
        ]

        ocr_confidences = [
            item[
                "average_ocr_confidence"
            ]
            for item in audited
        ]

        print(
            f"\nAverage OCR words: "
            f"{sum(word_counts) / len(word_counts):.1f}"
        )

        print(
            f"Minimum OCR words: "
            f"{min(word_counts)}"
        )

        print(
            f"Maximum OCR words: "
            f"{max(word_counts)}"
        )

        print(
            f"\nAverage OCR confidence: "
            f"{sum(ocr_confidences) / len(ocr_confidences):.1f}"
        )

        print(
            f"Minimum OCR confidence: "
            f"{min(ocr_confidences):.1f}"
        )

    # --------------------------------------------------
    # Group by confusion pair
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("OCR BY ERROR GROUP")
    print("=" * 60)

    groups = {}

    for item in audited:

        key = (
            item["actual"],
            item["predicted"],
        )

        groups.setdefault(
            key,
            [],
        ).append(item)

    for (
        actual,
        predicted,
    ), items in sorted(
        groups.items(),
        key=lambda pair: -len(
            pair[1]
        ),
    ):

        avg_words = (
            sum(
                item["word_count"]
                for item in items
            )
            / len(items)
        )

        avg_confidence = (
            sum(
                item[
                    "average_ocr_confidence"
                ]
                for item in items
            )
            / len(items)
        )

        print(
            f"\n{actual} -> {predicted}"
        )

        print(
            f"Count: {len(items)}"
        )

        print(
            f"Average words: "
            f"{avg_words:.1f}"
        )

        print(
            f"Average OCR confidence: "
            f"{avg_confidence:.1f}"
        )

    # --------------------------------------------------
    # Worst OCR documents
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("WORST OCR ERRORS")
    print("=" * 60)

    worst = sorted(
        audited,
        key=lambda item: (
            item["word_count"],
            item[
                "average_ocr_confidence"
            ],
        ),
    )

    for item in worst[:10]:

        print(
            f"\n{item['actual']} -> "
            f"{item['predicted']}"
        )

        print(
            f"Words: "
            f"{item['word_count']}"
        )

        print(
            f"OCR confidence: "
            f"{item['average_ocr_confidence']:.1f}"
        )

        print(
            f"Model confidence: "
            f"{item['model_confidence']:.4f}"
        )

        print(
            f"Image: "
            f"{item['image_path']}"
        )

        print(
            f"Text: "
            f"{item['ocr_text_preview'][:150]}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = {
        "total_errors": len(
            errors
        ),
        "audited_documents": len(
            audited
        ),
        "documents": audited,
    }

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

    print("\n" + "=" * 60)

    print(
        "OCR AUDIT COMPLETE"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()