import cv2

image = cv2.imread("data/uploads/Screenshot 2026-07-22 112154.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Original Shape :", image.shape)
print("Gray Shape     :", gray.shape)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()