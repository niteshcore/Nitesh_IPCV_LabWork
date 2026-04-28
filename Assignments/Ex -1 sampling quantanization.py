import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Sample Images/image1.jpg", 0)

if img is None:
    print("Error loading image")
    exit()

sampled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
sampled_display = cv2.resize(sampled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

levels = 8
step = 256 // levels
quantized = (img // step) * step

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sampled_display, cmap='gray')
plt.title("Sampled")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(quantized, cmap='gray')
plt.title("Quantized")
plt.axis('off')

plt.tight_layout()
plt.show()