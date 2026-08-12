"""
Compare OCR quality across preprocessing strategies.

Strategies:
A. Current: grayscale + adaptive threshold
B. Grayscale only
C. Original image
"""

import json
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


ERROR_FILE = Path(
    "checkpoints/error_analysis.json"
)

OUTPUT_FILE = Path(
    "checkpoints/ocr_preprocessing_comparison.json"
)


TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = (
    TESSERACT_PATH
)


def load_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read: {path}"
        )

    return image


def preprocess_current(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def preprocess_grayscale(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def preprocess_original(image):

    return image


def run_ocr(image):

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
        "ocr_confidence": round(
            average_confidence,
            2,
        ),
    }


def main():

    print("=" * 60)
    print("OCR PREPROCESSING COMPARISON")
    print("=" * 60)

    with open(
        ERROR_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    errors = results["errors"]

    print(
        f"\nTesting {len(errors)} "
        "misclassified documents."
    )

    strategies = {
        "current": preprocess_current,
        "grayscale": preprocess_grayscale,
        "original": preprocess_original,
    }

    results = []

    for index, error in enumerate(
        errors,
        start=1,
    ):

        path = Path(
            error["image_path"]
        )

        image = load_image(path)

        record = {
            "image_path": str(path),
            "actual": error["actual"],
            "predicted": error["predicted"],
        }

        print(
            f"\n[{index}/{len(errors)}] "
            f"{error['actual']} -> "
            f"{error['predicted']}"
        )

        for name, processor in (
            strategies.items()
        ):

            processed = processor(
                image
            )

            stats = run_ocr(
                processed
            )

            record[name] = stats

            print(
                f"  {name:10} "
                f"words={stats['word_count']:4d} "
                f"conf="
                f"{stats['ocr_confidence']:5.1f}"
            )

        results.append(record)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    for strategy in strategies:

        word_counts = [
            item[strategy][
                "word_count"
            ]
            for item in results
        ]

        confidences = [
            item[strategy][
                "ocr_confidence"
            ]
            for item in results
        ]

        print(
            f"\n{strategy.upper()}"
        )

        print(
            f"Average words: "
            f"{sum(word_counts) / len(word_counts):.1f}"
        )

        print(
            f"Average OCR confidence: "
            f"{sum(confidences) / len(confidences):.1f}"
        )

        print(
            f"Documents with 0 words: "
            f"{sum(x == 0 for x in word_counts)}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()