"""
Image preprocessing utilities for OCR.
"""

import cv2


class ImagePreprocessor:
    """
    Performs image preprocessing before OCR.
    """

    def load_image(self, image_path: str):
        """
        Load an image from disk.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        return image

    def to_grayscale(self, image):
        """
        Convert image to grayscale.
        """

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )