"""
Cached Document Dataset

Loads preprocessed LayoutLMv3 tensors from disk.
No OCR or image processing happens here.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedDocumentDataset(Dataset):
    """
    Dataset for loading cached LayoutLMv3 inputs.
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.samples = sorted(
            self.root_dir.glob("*.pt")
        )

        if not self.samples:
            raise RuntimeError(
                f"No cached samples found in "
                f"{self.root_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]

        sample = torch.load(
            sample_path,
            map_location="cpu",
            weights_only=True,
        )

        return sample