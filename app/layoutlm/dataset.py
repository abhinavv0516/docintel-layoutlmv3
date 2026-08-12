"""
Document Dataset

Loads document images, performs OCR, normalizes
bounding boxes, and prepares inputs for LayoutLMv3.
"""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from app.layoutlm.labels import DOCUMENT_CLASSES
from app.layoutlm.processor import DocumentProcessor
from app.ocr.engine import OCREngine


class DocumentDataset(Dataset):
    """
    PyTorch dataset for document classification.

    Pipeline:

        Image
          ↓
        Orientation correction
          ↓
        Tesseract OCR
          ↓
        Words + bounding boxes
          ↓
        Normalize boxes
          ↓
        LayoutLMv3 Processor
          ↓
        Model-ready tensors
    """

    def __init__(
        self,
        root_dir,
        processor=None,
        ocr_engine=None,
        max_length=512,
    ):
        self.root_dir = Path(root_dir)
        self.max_length = max_length

        self.processor = (
            processor
            if processor is not None
            else DocumentProcessor().get_processor()
        )

        self.ocr_engine = (
            ocr_engine
            if ocr_engine is not None
            else OCREngine()
        )

        self.samples = []

        for class_index, class_name in enumerate(
            DOCUMENT_CLASSES
        ):
            class_dir = (
                self.root_dir / class_name
            )

            if not class_dir.exists():
                continue

            image_files = sorted(
                class_dir.glob("*.png")
            )

            for image_path in image_files:
                self.samples.append(
                    (
                        image_path,
                        class_index,
                    )
                )

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.root_dir}"
            )

    def __len__(self):
        """Return number of documents."""

        return len(self.samples)

    @staticmethod
    def normalize_box(
        box,
        width,
        height,
    ):
        """
        Normalize bounding box coordinates
        to the LayoutLMv3 range [0, 1000].
        """

        x1, y1, x2, y2 = box

        x1 = max(
            0,
            min(
                1000,
                int(1000 * x1 / width),
            ),
        )

        y1 = max(
            0,
            min(
                1000,
                int(1000 * y1 / height),
            ),
        )

        x2 = max(
            0,
            min(
                1000,
                int(1000 * x2 / width),
            ),
        )

        y2 = max(
            0,
            min(
                1000,
                int(1000 * y2 / height),
            ),
        )

        return [
            x1,
            y1,
            x2,
            y2,
        ]

    def __getitem__(self, index):
        """
        Load and process one document.
        """

        image_path, label = self.samples[index]

        # --------------------------------------------------
        # 1. Load and prepare image
        # --------------------------------------------------

        image, rotation = (
            self.ocr_engine.prepare_image(
                str(image_path)
            )
        )

        if image is None:
            raise RuntimeError(
                f"Failed to load image: "
                f"{image_path}"
            )

        height, width = image.shape[:2]

        # LayoutLMv3 expects RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------
        # 2. OCR
        # --------------------------------------------------

        ocr_data = (
            self.ocr_engine.extract_data(
                str(image_path)
            )
        )

        words = []
        boxes = []

        for i in range(
            len(ocr_data["text"])
        ):
            text = (
                ocr_data["text"][i]
                .strip()
            )

            if not text:
                continue

            left = ocr_data["left"][i]
            top = ocr_data["top"][i]

            box_width = (
                ocr_data["width"][i]
            )

            box_height = (
                ocr_data["height"][i]
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

        # --------------------------------------------------
        # 3. OCR fallback
        # --------------------------------------------------

        if not words:
            print(
                f"Warning: OCR returned no words for: "
                f"{image_path}. "
                f"Using visual-only fallback."
            )

        # --------------------------------------------------
        # 4. LayoutLMv3 processing
        # --------------------------------------------------

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        encoding = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        # --------------------------------------------------
        # 5. Classification label
        # --------------------------------------------------

        encoding["labels"] = torch.tensor(
            label,
            dtype=torch.long,
        )

        return encoding