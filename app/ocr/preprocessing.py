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

    def adaptive_threshold(self, gray_image):
        """
        Apply adaptive thresholding to improve OCR accuracy.
        """

        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    def preprocess_for_ocr(self, image_path: str):
        """
        Complete preprocessing pipeline for OCR.
        """

        image = self.load_image(image_path)

        gray = self.to_grayscale(image)

        processed = self.adaptive_threshold(gray)

        return processed