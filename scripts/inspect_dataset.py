"""
Inspect a sample of images from each document class.
"""

import random
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT = Path("data/train")
OUTPUT_PATH = Path("data/output/dataset_inspection.png")

CLASSES = [
    "invoice",
    "resume",
    "form",
    "budget",
    "advertisement",
]

SAMPLES_PER_CLASS = 5

IMAGE_WIDTH = 250
IMAGE_HEIGHT = 330


def create_thumbnail(image_path):
    """Load and resize an image."""

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    return cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    )


def main():

    print("=" * 60)
    print("DATASET VISUAL INSPECTION")
    print("=" * 60)

    random.seed(42)

    rows = []

    for class_name in CLASSES:

        class_dir = DATASET_ROOT / class_name

        files = list(class_dir.glob("*.png"))

        if not files:
            print(f"WARNING: No images found for {class_name}")
            continue

        selected = random.sample(
            files,
            min(SAMPLES_PER_CLASS, len(files)),
        )

        print(
            f"{class_name}: "
            f"{len(files)} total, "
            f"inspecting {len(selected)}"
        )

        class_images = []

        for image_path in selected:

            image = create_thumbnail(image_path)

            if image is None:
                continue

            # Label bar
            cv2.rectangle(
                image,
                (0, 0),
                (IMAGE_WIDTH, 35),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                class_name,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            class_images.append(image)

        rows.append(class_images)

    if not rows:
        raise RuntimeError("No images found.")

    # Make every row exactly 5 images wide
    normalized_rows = []

    for row in rows:

        while len(row) < SAMPLES_PER_CLASS:

            blank = np.ones(
                (
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    3,
                ),
                dtype=np.uint8,
            ) * 255

            row.append(blank)

        normalized_rows.append(row)

    contact_sheet = cv2.vconcat(
        [
            cv2.hconcat(row)
            for row in normalized_rows
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT_PATH),
        contact_sheet,
    )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()