"""
LayoutLMv3 Model

Responsible for loading the pretrained
LayoutLMv3 transformer model.
"""

from transformers import LayoutLMv3Model


class DocumentModel:
    """
    Wrapper around Hugging Face LayoutLMv3 model.
    """

    def __init__(self):

        self.model = LayoutLMv3Model.from_pretrained(
            "microsoft/layoutlmv3-base"
        )

    def get_model(self):
        """
        Return the underlying model.
        """

        return self.model