"""
Test the LayoutLMv3 processor.
"""

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.processor import DocumentProcessor


def main():

    processor = DocumentProcessor()

    print("=" * 50)
    print("LAYOUTLMv3 PROCESSOR LOADED")
    print("=" * 50)

    print(type(processor.get_processor()))


if __name__ == "__main__":
    main()