import cv2

image = cv2.imread("data/uploads/Screenshot 2026-07-22 112154.png")

print("Type:", type(image))
print("Shape:", image.shape)
print("Data Type:", image.dtype)