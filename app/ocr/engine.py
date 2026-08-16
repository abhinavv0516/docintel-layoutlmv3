"""
OCR Engine

Responsible for extracting text and OCR metadata
using the Tesseract OCR engine.
"""

import cv2
import pytesseract
from pytesseract import Output

from app.ocr.preprocessing import ImagePreprocessor


# Path to the Tesseract executable

import shutil

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

class OCREngine:
    """
    Wrapper around Tesseract OCR.
    """

    def __init__(
        self,
        preprocessing_mode="adaptive",
    ):

        self.preprocessor = ImagePreprocessor(
            mode=preprocessing_mode
        )

    def prepare_image(
        self,
        image_path: str,
    ):
        """
        Prepare the document image.

        Returns:
            prepared BGR image
            rotation angle
        """

        return self.preprocessor.prepare_image(
            image_path
        )

    def extract_text(
        self,
        image_path: str,
    ) -> str:
        """
        Extract plain text from an image.
        """

        image = (
            self.preprocessor
            .preprocess_for_ocr(
                image_path
            )
        )

        text = pytesseract.image_to_string(
            image
        )

        return text

    def extract_data(
        self,
        image_path: str,
    ) -> dict:
        """
        Extract OCR metadata.
        """

        image = (
            self.preprocessor
            .preprocess_for_ocr(
                image_path
            )
        )

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
        )

        return data