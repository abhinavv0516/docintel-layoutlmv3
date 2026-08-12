"""
Generate a visual contact sheet of grayscale model errors.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ERROR_FILE = Path(
    "checkpoints/grayscale/error_analysis.json"
)

OUTPUT_FILE = Path(
    "checkpoints/grayscale/error_contact_sheet.jpg"
)

THUMBNAIL_SIZE = (300, 300)
COLUMNS = 3


def get_font(size):
    try:
        return ImageFont.truetype(
            "arial.ttf",
            size,
        )
    except OSError:
        return ImageFont.load_default()


def main():

    print("=" * 60)
    print("GRAYSCALE ERROR CONTACT SHEET")
    print("=" * 60)

    with open(
        ERROR_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    errors = results["errors"]

    print(
        f"\nErrors found: {len(errors)}"
    )

    rows = math.ceil(
        len(errors) / COLUMNS
    )

    cell_width = 320
    cell_height = 370

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * cell_width,
            rows * cell_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    title_font = get_font(18)
    info_font = get_font(15)

    for index, error in enumerate(errors):

        image_path = Path(
            error["image_path"]
        )

        if not image_path.exists():
            print(
                f"Missing image: {image_path}"
            )
            continue

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                THUMBNAIL_SIZE
            )

        except Exception as exc:

            print(
                f"Failed to load "
                f"{image_path}: {exc}"
            )

            continue

        column = index % COLUMNS
        row = index // COLUMNS

        x = (
            column * cell_width
            + (cell_width - image.width) // 2
        )

        y = (
            row * cell_height
            + 55
        )

        draw.text(
            (
                column * cell_width + 10,
                row * cell_height + 10,
            ),
            f"{index + 1}. "
            f"{error['actual']} → "
            f"{error['predicted']}",
            fill="black",
            font=title_font,
        )

        draw.text(
            (
                column * cell_width + 10,
                row * cell_height + 34,
            ),
            f"Confidence: "
            f"{error['confidence']:.3f}",
            fill="black",
            font=info_font,
        )

        sheet.paste(
            image,
            (x, y),
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        OUTPUT_FILE,
        quality=95,
    )

    print(
        "\nContact sheet saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\n" + "=" * 60
    )


if __name__ == "__main__":
    main()