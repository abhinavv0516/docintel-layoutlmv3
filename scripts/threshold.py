import cv2

image = cv2.imread("data/uploads/Screenshot 2026-07-22 112154.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY,
)

cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Threshold", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()