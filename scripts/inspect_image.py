import cv2

image = cv2.imread("data/uploads/Screenshot 2026-07-22 112154.png")

print("Shape :", image.shape)
print("Type  :", type(image))
print("DType :", image.dtype)

print("\nFirst Pixel (BGR):")
print(image[0, 0])

print("\nCenter Pixel (BGR):")
h, w, _ = image.shape
print(image[h // 2, w // 2])