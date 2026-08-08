"""
Document classification head for LayoutLMv3.
"""

import torch
import torch.nn as nn


class DocumentClassifier(nn.Module):
    """
    Classification head built on top of LayoutLMv3
    document embeddings.
    """

    def __init__(self, hidden_size=768, num_classes=5):
        super().__init__()

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, document_embedding):
        """
        Convert the document embedding into class logits.
        """

        return self.classifier(document_embedding)