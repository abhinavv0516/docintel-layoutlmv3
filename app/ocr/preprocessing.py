"""
Image preprocessing utilities for OCR.

Supports:
    adaptive
    grayscale
    original
    oriented_grayscale
"""

import cv2
import pytesseract


class ImagePreprocessor:
    """
    Performs image preprocessing before OCR
    and optionally corrects document orientation.
    """

    def __init__(self, mode="adaptive"):
        """
        Initialize preprocessing strategy.

        Modes:
            adaptive
                Grayscale + adaptive threshold.

            grayscale
                Grayscale only.

            original
                Original image.

            oriented_grayscale
                Detect orientation, rotate the image,
                then convert to grayscale.
        """

        valid_modes = {
            "adaptive",
            "grayscale",
            "original",
            "oriented_grayscale",
        }

        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported preprocessing mode: {mode}. "
                f"Expected one of {valid_modes}."
            )

        self.mode = mode

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
        Apply adaptive thresholding.
        """

        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    def detect_orientation(self, image):
        """
        Detect document orientation using Tesseract OSD.

        Returns:
            Rotation angle in degrees.

        Supported rotations:
            0
            90
            180
            270

        If orientation detection fails, return 0.
        """

        try:

            osd = pytesseract.image_to_osd(
                image,
                config="--psm 0",
            )

            rotation = 0

            for line in osd.splitlines():

                if line.startswith(
                    "Rotate:"
                ):

                    rotation = int(
                        line.split(":")[1].strip()
                    )

                    break

            if rotation not in {
                0,
                90,
                180,
                270,
            }:

                return 0

            return rotation

        except Exception:

            # Some sparse or damaged documents
            # do not contain enough information
            # for Tesseract OSD.
            return 0

    def rotate_image(
        self,
        image,
        rotation,
    ):
        """
        Rotate image according to Tesseract OSD.
        """

        if rotation == 90:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

        if rotation == 180:

            return cv2.rotate(
                image,
                cv2.ROTATE_180,
            )

        if rotation == 270:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        return image

    def prepare_image(self, image_path: str):
        """
        Load and optionally orientation-correct
        the original image.

        Returns:
            prepared BGR image
            rotation angle
        """

        image = self.load_image(
            image_path
        )

        rotation = 0

        if self.mode == "oriented_grayscale":

            rotation = self.detect_orientation(
                image
            )

            image = self.rotate_image(
                image,
                rotation
            )

        return image, rotation

    def preprocess_for_ocr(
        self,
        image_path: str,
    ):
        """
        Complete preprocessing pipeline
        for OCR.
        """

        image, _ = self.prepare_image(
            image_path
        )

        if self.mode == "original":
            return image

        gray = self.to_grayscale(
            image
        )

        if self.mode in {
            "grayscale",
            "oriented_grayscale",
        }:

            return gray

        return self.adaptive_threshold(
            gray
        )