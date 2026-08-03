"""
Test the reusable OCR Engine.
"""
import sys
from pathlib import Path

# Add project root to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ocr.engine import OCREngine
from app.ocr.engine import OCREngine

IMAGE_PATH = "data/uploads/Screenshot 2026-07-22 112154.png"


def main():
    engine = OCREngine()

    text = engine.extract_text(IMAGE_PATH)

    print("=" * 60)
    print("OCR OUTPUT")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()