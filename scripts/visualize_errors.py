"""
Create a contact sheet of all misclassified test documents.

Reads checkpoints/test_errors.json and creates a visual grid
showing actual class, predicted class, and confidence.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ERRORS_FILE = Path(
    "checkpoints/test_errors.json"
)

OUTPUT_FILE = Path(
    "checkpoints/test_errors_contact_sheet.jpg"
)

THUMBNAIL_SIZE = (
    300,
    220,
)

COLUMNS = 3

LABEL_HEIGHT = 80


def load_errors():
    """Load misclassified document information."""

    with open(
        ERRORS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_font(size):
    """Load a suitable font."""

    try:
        return ImageFont.truetype(
            "arial.ttf",
            size,
        )

    except OSError:

        return ImageFont.load_default()


def main():

    print("=" * 60)
    print("VISUAL TEST ERROR ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------
    # Load error list
    # --------------------------------------------------

    errors = load_errors()

    print(
        f"\nErrors found: {len(errors)}"
    )

    # --------------------------------------------------
    # Calculate sheet dimensions
    # --------------------------------------------------

    rows = (
        len(errors) + COLUMNS - 1
    ) // COLUMNS

    cell_width = THUMBNAIL_SIZE[0]

    cell_height = (
        THUMBNAIL_SIZE[1]
        + LABEL_HEIGHT
    )

    sheet_width = (
        COLUMNS * cell_width
    )

    sheet_height = (
        rows * cell_height
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

    title_font = load_font(16)
    info_font = load_font(13)

    # --------------------------------------------------
    # Process each error
    # --------------------------------------------------

    for index, error in enumerate(
        errors
    ):

        image_path = Path(
            error["image_path"]
        )

        if not image_path.exists():

            print(
                f"WARNING: Missing image: "
                f"{image_path}"
            )

            continue

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                THUMBNAIL_SIZE
            )

        except Exception as exception:

            print(
                f"WARNING: Could not open "
                f"{image_path}: {exception}"
            )

            continue

        # ----------------------------------------------
        # Cell position
        # ----------------------------------------------

        column = (
            index % COLUMNS
        )

        row = (
            index // COLUMNS
        )

        x = (
            column * cell_width
        )

        y = (
            row * cell_height
        )

        # ----------------------------------------------
        # Center image
        # ----------------------------------------------

        image_x = (
            x
            + (
                cell_width
                - image.width
            )
            // 2
        )

        image_y = (
            y
            + (
                THUMBNAIL_SIZE[1]
                - image.height
            )
            // 2
        )

        sheet.paste(
            image,
            (
                image_x,
                image_y,
            ),
        )

        # ----------------------------------------------
        # Error information
        # ----------------------------------------------

        actual = error[
            "actual"
        ]

        predicted = error[
            "predicted"
        ]

        confidence = (
            error["confidence"]
            * 100
        )

        filename = (
            image_path.name
        )

        info_y = (
            y
            + THUMBNAIL_SIZE[1]
            + 5
        )

        draw.text(
            (
                x + 5,
                info_y,
            ),
            f"{index + 1}. {filename}",
            fill="black",
            font=title_font,
        )

        draw.text(
            (
                x + 5,
                info_y + 20,
            ),
            (
                f"Actual: {actual}  "
                f"→  Predicted: {predicted}"
            ),
            fill="black",
            font=info_font,
        )

        draw.text(
            (
                x + 5,
                info_y + 40,
            ),
            f"Confidence: {confidence:.2f}%",
            fill="black",
            font=info_font,
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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
        "\nVisual error analysis complete."
    )


if __name__ == "__main__":
    main()