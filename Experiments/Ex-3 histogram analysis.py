import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Sample Images/image2.jpg", 0)

if img is None:
    print("Image not loaded")
    exit()

min_val, max_val = np.min(img), np.max(img)
cs_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8) if max_val > min_val else img.copy()
he_img = cv2.equalizeHist(img)

plt.figure(figsize=(10,8))

plt.subplot(3,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(3,2,2)
plt.hist(img.ravel(), 256, [0,256])
plt.title("Histogram")

plt.subplot(3,2,3)
plt.imshow(cs_img, cmap='gray')
plt.title("Contrast Stretching")
plt.axis('off')

plt.subplot(3,2,4)
plt.hist(cs_img.ravel(), 256, [0,256])
plt.title("Histogram")

plt.subplot(3,2,5)
plt.imshow(he_img, cmap='gray')
plt.title("Hist Equalized")
plt.axis('off')

plt.subplot(3,2,6)
plt.hist(he_img.ravel(), 256, [0,256])
plt.title("Histogram")

plt.tight_layout()
plt.show()
