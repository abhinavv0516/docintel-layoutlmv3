import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image = cv2.imread(
    "data/uploads/Screenshot 2026-07-22 112154.png"
)

data = pytesseract.image_to_data(
    image,
    output_type=Output.DICT
)

print("\nDetected Words:\n")

for i in range(len(data["text"])):

    word = data["text"][i].strip()

    if word != "":

        print(
            f"{word:20}"
            f" Confidence: {data['conf'][i]:>4}"
            f" Position: ({data['left'][i]}, {data['top'][i]})"
        )