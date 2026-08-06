"""
Test the LayoutLMv3 model.
"""

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.model import DocumentModel


def main():

    model = DocumentModel()

    print("=" * 50)
    print("LAYOUTLMv3 MODEL LOADED")
    print("=" * 50)

    print(type(model.get_model()))


if __name__ == "__main__":
    main()