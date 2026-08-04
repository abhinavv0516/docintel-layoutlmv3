"""
Test the complete OCR pipeline.
"""

import sys
from pathlib import Path

import cv2

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ocr.engine import OCREngine
from app.ocr.preprocessing import ImagePreprocessor
from app.ocr.visualization import OCRVisualizer


IMAGE_PATH = "data/uploads/Screenshot 2026-07-22 112154.png"
OUTPUT_PATH = "data/output/final_ocr_visualization.png"


def main():
    def main():

    preprocessor = ImagePreprocessor()
    engine = OCREngine()
    visualizer = OCRVisualizer()

    # Create output directory if it doesn't exist
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = preprocessor.load_image(IMAGE_PATH)

    data = engine.extract_data(IMAGE_PATH)

    annotated_image = visualizer.draw_boxes(
        image,
        data,
    )

    cv2.imwrite(
        OUTPUT_PATH,
        annotated_image,
    )

    cv2.imshow(
        "OCR Visualization",
        annotated_image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    preprocessor = ImagePreprocessor()
    engine = OCREngine()
    visualizer = OCRVisualizer()

    # Load original image
    image = preprocessor.load_image(IMAGE_PATH)

    # Extract OCR metadata
    data = engine.extract_data(IMAGE_PATH)

    # Draw OCR boxes
    annotated_image = visualizer.draw_boxes(
        image,
        data,
    )

    # Save output
    cv2.imwrite(
        OUTPUT_PATH,
        annotated_image,
    )

    # Show output
    cv2.imshow(
        "OCR Visualization",
        annotated_image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()