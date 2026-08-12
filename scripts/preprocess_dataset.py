"""
Preprocess and cache LayoutLMv3 inputs.

Pipeline:

    Image
      ↓
    Tesseract OCR
      ↓
    Bounding boxes
      ↓
    LayoutLMv3 Processor
      ↓
    Cached tensors

The script is resumable and records failed documents.
"""

import sys
from pathlib import Path

import torch

# Add project root to Python path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.dataset import DocumentDataset
from app.ocr.engine import OCREngine


DATASETS = {
    "train": "data/clean/train",
    "validation": "data/clean/validation",
    "test": "data/clean/test",
}

# OCR experiment:
# grayscale only, instead of adaptive thresholding

OCR_PREPROCESSING_MODE = "oriented_grayscale"

OUTPUT_ROOT = Path(
    "data/processed_oriented"
)

FAILURE_ROOT = (
    OUTPUT_ROOT / "failures"
)


def preprocess_split(
    split_name,
    source_dir,
):
    """
    Preprocess one dataset split.
    """

    print("\n" + "=" * 60)
    print(
        f"PREPROCESSING {split_name.upper()} DATASET"
    )
    print("=" * 60)

    dataset = DocumentDataset(
        source_dir,
        ocr_engine=OCREngine(
            preprocessing_mode=OCR_PREPROCESSING_MODE
        ),
    )

    output_dir = (
        OUTPUT_ROOT / split_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAILURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_log = (
        FAILURE_ROOT
        / f"{split_name}_failures.txt"
    )

    total = len(dataset)

    print(
        f"\nDocuments: {total}"
    )

    processed = 0
    skipped = 0
    failed = 0

    for index in range(total):

        image_path, label = (
            dataset.samples[index]
        )

        output_path = (
            output_dir
            / f"{index:06d}.pt"
        )

        # Resume support
        if output_path.exists():

            skipped += 1
            continue

        try:

            sample = dataset[index]

            torch.save(
                {
                    "input_ids": sample[
                        "input_ids"
                    ],
                    "attention_mask": sample[
                        "attention_mask"
                    ],
                    "bbox": sample[
                        "bbox"
                    ],
                    "pixel_values": sample[
                        "pixel_values"
                    ],
                    "labels": sample[
                        "labels"
                    ],
                    "image_path": str(
                        image_path
                    ),
                },
                output_path,
            )

            processed += 1

            print(
                f"{split_name}: "
                f"{index + 1}/{total}"
            )

        except Exception as error:

            failed += 1

            error_message = (
                f"INDEX: {index}\n"
                f"IMAGE: {image_path}\n"
                f"LABEL: {label}\n"
                f"ERROR: {repr(error)}\n"
                f"{'-' * 60}\n"
            )

            with open(
                failure_log,
                "a",
                encoding="utf-8",
            ) as file:

                file.write(
                    error_message
                )

            print(
                f"\nERROR processing "
                f"{image_path}"
            )

            print(
                f"Reason: {error}"
            )

    print(
        "\n" + "-" * 60
    )

    print(
        f"{split_name} complete"
    )

    print(
        f"Processed: {processed}"
    )

    print(
        f"Skipped:   {skipped}"
    )

    print(
        f"Failed:    {failed}"
    )

    print(
        f"Failure log: {failure_log}"
    )

    print(
        "-" * 60
    )

    return failed


def main():

    print("=" * 60)
    print("DOCUMENT DATASET PREPROCESSING")
    print("=" * 60)

    print(
        "\nOCR preprocessing mode:"
    )

    print(
        OCR_PREPROCESSING_MODE
    )

    print(
        "\nCached data:"
    )

    print(
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        "\nExisting .pt files will be skipped."
    )

    print(
        "Only missing documents will be processed."
    )

    total_failed = 0

    for (
        split_name,
        source_dir,
    ) in DATASETS.items():

        failed = preprocess_split(
            split_name,
            source_dir,
        )

        total_failed += failed

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    if total_failed == 0:

        print(
            "\nAll documents processed successfully."
        )

    else:

        print(
            f"\nTotal failed documents: "
            f"{total_failed}"
        )

        print(
            "\nFailure logs:"
        )

        print(
            FAILURE_ROOT.resolve()
        )


if __name__ == "__main__":
    main()