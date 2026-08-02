"""
Visualize OCR detections by drawing bounding boxes and labels
on the detected words.
"""

import cv2
import pytesseract
from pytesseract import Output

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load the input image
image = cv2.imread(
    "data/uploads/Screenshot 2026-07-22 112154.png"
)

# Run OCR and get detailed information
data = pytesseract.image_to_data(
    image,
    output_type=Output.DICT,
)

# Iterate through every OCR detection
for i in range(len(data["text"])):

    # Remove extra whitespace
    word = data["text"][i].strip()

    # Skip empty detections
    if word == "":
        continue

    # OCR confidence score
    confidence = float(data["conf"][i])

    # Ignore weak OCR detections
    if confidence < 50:
        continue

    # Bounding box coordinates
    x = data["left"][i]
    y = data["top"][i]
    w = data["width"][i]
    h = data["height"][i]

    # Calculate bottom-right corner
    x2 = x + w
    y2 = y + h

    # Draw bounding box
    cv2.rectangle(
        image,
        (x, y),
        (x2, y2),
        (0, 255, 0),  # Green (BGR)
        2,
    )

    # Draw detected word above the box
    cv2.putText(
        image,
        word,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )

# Save annotated image
cv2.imwrite(
    "data/output/ocr_boxes.png",
    image,
)

# Display annotated image
cv2.imshow("OCR Bounding Boxes", image)

cv2.waitKey(0)
cv2.destroyAllWindows()