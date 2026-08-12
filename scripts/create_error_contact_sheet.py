"""
Create a visual contact sheet of LayoutLMv3 test errors.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ERROR_FILE = Path(
    "checkpoints/error_analysis.json"
)

OUTPUT_FILE = Path(
    "checkpoints/error_contact_sheet.jpg"
)

THUMBNAIL_SIZE = (220, 300)
COLUMNS = 4


def main():

    print("=" * 60)
    print("CREATING ERROR CONTACT SHEET")
    print("=" * 60)

    with open(
        ERROR_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    errors = results["errors"]

    if not errors:
        print("\nNo errors found.")
        return

    rows = math.ceil(
        len(errors) / COLUMNS
    )

    label_height = 70

    sheet_width = (
        COLUMNS * THUMBNAIL_SIZE[0]
    )

    sheet_height = (
        rows
        * (
            THUMBNAIL_SIZE[1]
            + label_height
        )
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height,
            ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            16,
        )
    except OSError:
        font = ImageFont.load_default()

    for index, error in enumerate(errors):

        image_path = Path(
            error["image_path"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                THUMBNAIL_SIZE
            )

        except Exception as exc:

            print(
                f"Could not open "
                f"{image_path}: {exc}"
            )

            continue

        column = (
            index % COLUMNS
        )

        row = (
            index // COLUMNS
        )

        x = (
            column
            * THUMBNAIL_SIZE[0]
        )

        y = (
            row
            * (
                THUMBNAIL_SIZE[1]
                + label_height
            )
        )

        image_x = (
            x
            + (
                THUMBNAIL_SIZE[0]
                - image.width
            )
            // 2
        )

        image_y = y

        sheet.paste(
            image,
            (
                image_x,
                image_y,
            ),
        )

        label_y = (
            y
            + THUMBNAIL_SIZE[1]
            + 5
        )

        text = (
            f"{error['actual']} → "
            f"{error['predicted']}\n"
            f"conf: "
            f"{error['confidence']:.2f}\n"
            f"{image_path.name}"
        )

        draw.multiline_text(
            (
                x + 5,
                label_y,
            ),
            text,
            fill="black",
            font=font,
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        OUTPUT_FILE,
        quality=90,
    )

    print(
        f"\nSaved contact sheet:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
