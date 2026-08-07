"""
LayoutLMv3 Processor

Responsible for preparing document images
and OCR information for the LayoutLMv3 model.
"""

from transformers import LayoutLMv3Processor


class DocumentProcessor:
    """
    Wrapper around Hugging Face LayoutLMv3 Processor.
    """

    def __init__(self):

        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
)

    def get_processor(self):
        """
        Return the underlying Hugging Face processor.
        """

        return self.processor