import cv2
import pytesseract

# Tell pytesseract where the OCR engine is
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image = cv2.imread(
    "data/uploads/Screenshot 2026-07-22 112154.png"
)

print("Tesseract Version:")
print(pytesseract.get_tesseract_version())

print("\nRunning OCR...\n")

text = pytesseract.image_to_string(image)

print("=" * 50)
print("OCR OUTPUT")
print("=" * 50)

print(text)