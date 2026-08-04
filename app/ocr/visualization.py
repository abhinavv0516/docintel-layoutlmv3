"""
OCR visualization utilities.
"""

import cv2


class OCRVisualizer:
    """
    Draw OCR detections on images.
    """

    def draw_boxes(self, image, ocr_data):
        """
        Draw bounding boxes and labels on the image.
        """

        for i in range(len(ocr_data["text"])):

            word = ocr_data["text"][i].strip()

            if word == "":
                continue

            confidence = float(ocr_data["conf"][i])

            if confidence < 50:
                continue

            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            x2 = x + w
            y2 = y + h

            # Draw rectangle
            cv2.rectangle(
                image,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Draw label
            cv2.putText(
                image,
                word,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        return image