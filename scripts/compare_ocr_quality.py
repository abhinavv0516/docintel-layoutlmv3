"""
Compare OCR quality between correctly and incorrectly
classified clean test documents.

This determines whether poor OCR is associated with
classification errors.
"""

import json
import random
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


ERROR_FILE = Path(
    "checkpoints/error_analysis.json"
)

TEST_DIR = Path(
    "data/clean/test"
)

SEED = 42

SAMPLES_PER_CLASS = 39


def analyze_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read: {image_path}"
        )

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
    )

    words = []
    confidences = []

    for index, text in enumerate(
        data["text"]
    ):

        text = text.strip()

        if not text:
            continue

        words.append(text)

        try:
            confidence = float(
                data["conf"][index]
            )

            if confidence >= 0:
                confidences.append(
                    confidence
                )

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
        "word_count": len(words),
        "ocr_confidence": average_confidence,
    }


def main():

    print("=" * 60)
    print("OCR QUALITY CONTROL COMPARISON")
    print("=" * 60)

    with open(
        ERROR_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        error_results = json.load(file)

    errors = error_results[
        "errors"
    ]

    # --------------------------------------------------
    # Build error paths
    # --------------------------------------------------

    error_paths = {
        str(
            Path(
                item["image_path"]
            )
        ).lower()
        for item in errors
    }

    # --------------------------------------------------
    # Collect test images by class
    # --------------------------------------------------

    classes = [
        "invoice",
        "resume",
        "form",
        "budget",
        "advertisement",
    ]

    random.seed(SEED)

    correct_candidates = {
        class_name: []
        for class_name in classes
    }

    for class_name in classes:

        class_dir = (
            TEST_DIR
            / class_name
        )

        for image_path in class_dir.glob(
            "*.png"
        ):

            if (
                str(image_path).lower()
                in error_paths
            ):
                continue

            correct_candidates[
                class_name
            ].append(image_path)

    # --------------------------------------------------
    # Sample control documents
    # --------------------------------------------------

    control_samples = {}

    for class_name in classes:

        candidates = (
            correct_candidates[
                class_name
            ]
        )

        random.shuffle(
            candidates
        )

        control_samples[
            class_name
        ] = candidates[
            :SAMPLES_PER_CLASS
        ]

    # --------------------------------------------------
    # Audit errors
    # --------------------------------------------------

    print(
        "\nAuditing incorrect documents..."
    )

    error_records = []

    for item in errors:

        path = Path(
            item["image_path"]
        )

        stats = analyze_image(
            path
        )

        error_records.append(
            {
                "class": item[
                    "actual"
                ],
                **stats,
            }
        )

    # --------------------------------------------------
    # Audit controls
    # --------------------------------------------------

    print(
        "Auditing correctly classified "
        "control documents..."
    )

    control_records = []

    for class_name in classes:

        for path in control_samples[
            class_name
        ]:

            stats = analyze_image(
                path
            )

            control_records.append(
                {
                    "class": class_name,
                    **stats,
                }
            )

    # --------------------------------------------------
    # Overall comparison
    # --------------------------------------------------

    def average(
        records,
        key,
    ):

        values = [
            item[key]
            for item in records
        ]

        return (
            sum(values)
            / len(values)
            if values
            else 0.0
        )

    print("\n" + "=" * 60)
    print("OVERALL COMPARISON")
    print("=" * 60)

    print(
        f"\nIncorrect documents:"
    )

    print(
        f"  Samples: "
        f"{len(error_records)}"
    )

    print(
        f"  Avg words: "
        f"{average(error_records, 'word_count'):.1f}"
    )

    print(
        f"  Avg OCR confidence: "
        f"{average(error_records, 'ocr_confidence'):.1f}"
    )

    print(
        f"\nCorrect control documents:"
    )

    print(
        f"  Samples: "
        f"{len(control_records)}"
    )

    print(
        f"  Avg words: "
        f"{average(control_records, 'word_count'):.1f}"
    )

    print(
        f"  Avg OCR confidence: "
        f"{average(control_records, 'ocr_confidence'):.1f}"
    )

    # --------------------------------------------------
    # Per-class comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PER-CLASS COMPARISON")
    print("=" * 60)

    for class_name in classes:

        class_errors = [
            item
            for item in error_records
            if item["class"]
            == class_name
        ]

        class_controls = [
            item
            for item in control_records
            if item["class"]
            == class_name
        ]

        print(
            f"\n{class_name}"
        )

        print(
            f"  Errors: "
            f"{len(class_errors)}"
        )

        if class_errors:

            print(
                f"  Error avg words: "
                f"{average(class_errors, 'word_count'):.1f}"
            )

            print(
                f"  Error OCR conf: "
                f"{average(class_errors, 'ocr_confidence'):.1f}"
            )

        print(
            f"  Controls: "
            f"{len(class_controls)}"
        )

        print(
            f"  Control avg words: "
            f"{average(class_controls, 'word_count'):.1f}"
        )

        print(
            f"  Control OCR conf: "
            f"{average(class_controls, 'ocr_confidence'):.1f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = {
        "incorrect": error_records,
        "correct_controls": control_records,
    }

    output_path = Path(
        "checkpoints/ocr_quality_comparison.json"
    )

    with open(
        output_path,
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
        "COMPARISON COMPLETE"
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()