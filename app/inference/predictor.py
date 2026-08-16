"""
Document inference service.

Loads the trained grayscale LayoutLMv3 model once
and provides reusable single-document prediction.
"""

from pathlib import Path

import cv2
import torch
from transformers import LayoutLMv3ForSequenceClassification

from app.layoutlm.labels import DOCUMENT_CLASSES
from app.layoutlm.processor import DocumentProcessor
from app.ocr.engine import OCREngine


class DocumentPredictor:
    """
    Production inference wrapper for the trained
    LayoutLMv3 document classifier.

    Components are loaded once during initialization
    and reused for every prediction.
    """

    MODEL_PATH = (
        Path("checkpoints")
        / "grayscale"
        / "best_model"
    )

    OCR_PREPROCESSING_MODE = "grayscale"

    MAX_LENGTH = 512

    def __init__(self):
        """
        Load model, processor, and OCR engine once.
        """

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading DocumentPredictor on "
            f"{self.device}..."
        )

        # ----------------------------------------------
        # Load trained model
        # ----------------------------------------------

        self.model = (
            LayoutLMv3ForSequenceClassification
            .from_pretrained(
                str(self.MODEL_PATH)
            )
        )

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------------------------
        # Load processor
        # ----------------------------------------------

        self.processor = (
            DocumentProcessor()
            .get_processor()
        )

        # ----------------------------------------------
        # Load OCR engine
        # ----------------------------------------------

        self.ocr_engine = OCREngine(
            preprocessing_mode=(
                self.OCR_PREPROCESSING_MODE
            )
        )

        print(
            "DocumentPredictor ready."
        )

    # --------------------------------------------------
    # Bounding box normalization
    # --------------------------------------------------

    @staticmethod
    def normalize_box(
        box,
        width,
        height,
    ):
        """
        Normalize OCR coordinates to the
        LayoutLMv3 [0, 1000] coordinate range.
        """

        x1, y1, x2, y2 = box

        x1 = max(
            0,
            min(
                1000,
                int(
                    1000 * x1 / width
                ),
            ),
        )

        y1 = max(
            0,
            min(
                1000,
                int(
                    1000 * y1 / height
                ),
            ),
        )

        x2 = max(
            0,
            min(
                1000,
                int(
                    1000 * x2 / width
                ),
            ),
        )

        y2 = max(
            0,
            min(
                1000,
                int(
                    1000 * y2 / height
                ),
            ),
        )

        return [
            x1,
            y1,
            x2,
            y2,
        ]

    # --------------------------------------------------
    # OCR
    # --------------------------------------------------

    def _extract_ocr(
        self,
        image_path,
        width,
        height,
    ):
        """
        Extract OCR words and normalized bounding boxes.
        """

        ocr_data = (
            self.ocr_engine.extract_data(
                str(image_path)
            )
        )

        words = []
        boxes = []

        for index in range(
            len(ocr_data["text"])
        ):

            text = (
                ocr_data["text"][index]
                .strip()
            )

            if not text:
                continue

            left = ocr_data["left"][index]
            top = ocr_data["top"][index]

            box_width = (
                ocr_data["width"][index]
            )

            box_height = (
                ocr_data["height"][index]
            )

            box = [
                left,
                top,
                left + box_width,
                top + box_height,
            ]

            normalized_box = (
                self.normalize_box(
                    box,
                    width,
                    height,
                )
            )

            words.append(text)
            boxes.append(
                normalized_box
            )

        return words, boxes

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(
        self,
        image_path,
    ):
        """
        Classify one document.

        Returns:
            dict containing predicted class,
            confidence, OCR word count, and
            probabilities for all classes.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Document not found: "
                f"{image_path}"
            )

        # ----------------------------------------------
        # Load image
        # ----------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Failed to load image: "
                f"{image_path}"
            )

        height, width = image.shape[:2]

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------
        # OCR
        # ----------------------------------------------

        words, boxes = (
            self._extract_ocr(
                image_path,
                width,
                height,
            )
        )

        # ----------------------------------------------
        # LayoutLMv3 processor
        # ----------------------------------------------

        encoding = self.processor(
            image_rgb,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        )

        encoding = {
            key: value.to(
                self.device
            )
            for key, value in encoding.items()
        }

        # ----------------------------------------------
        # Model inference
        # ----------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                input_ids=encoding[
                    "input_ids"
                ],
                attention_mask=encoding[
                    "attention_mask"
                ],
                bbox=encoding[
                    "bbox"
                ],
                pixel_values=encoding[
                    "pixel_values"
                ],
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predicted_index = (
            probabilities
            .argmax(
                dim=-1
            )
            .item()
        )

        confidence = (
            probabilities[
                0,
                predicted_index,
            ]
            .item()
        )

        predicted_class = (
            DOCUMENT_CLASSES[
                predicted_index
            ]
        )

        # ----------------------------------------------
        # Class probabilities
        # ----------------------------------------------

        class_probabilities = {}

        for (
            index,
            class_name,
        ) in enumerate(
            DOCUMENT_CLASSES
        ):

            class_probabilities[
                class_name
            ] = (
                probabilities[
                    0,
                    index,
                ]
                .item()
            )

        return {
            "document_type": predicted_class,
            "confidence": confidence,
            "ocr_words": len(words),
            "probabilities": (
                class_probabilities
            ),
        }