import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('lenam.bmp', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('hrec.BMP', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('objectm.bmp', cv2.IMREAD_GRAYSCALE)

# Transformation T1 
T1_image1 = 255 - image1
T1_image2 = 255 - image2
T1_image3 = 255 - image3

# Transformation T2 
T2_image1 = np.where(image1 < 128, 0, 255)
T2_image2 = np.where(image2 < 128, 0, 255)
T2_image3 = np.where(image3 < 128, 0, 255)

# résultats
plt.figure(figsize=(18, 12))

# Image 1
plt.subplot(3, 3, 1), plt.imshow(image1, cmap='gray'), plt.title('Image Originale 1')
plt.subplot(3, 3, 2), plt.imshow(T1_image1, cmap='gray'), plt.title('Transformation T1')
plt.subplot(3, 3, 3), plt.imshow(T2_image1, cmap='gray'), plt.title('Transformation T2')

# Image 2
plt.subplot(3, 3, 4), plt.imshow(image2, cmap='gray'), plt.title('Image Originale 2')
plt.subplot(3, 3, 5), plt.imshow(T1_image2, cmap='gray'), plt.title('Transformation T1')
plt.subplot(3, 3, 6), plt.imshow(T2_image2, cmap='gray'), plt.title('Transformation T2')

# Image 3
plt.subplot(3, 3, 7), plt.imshow(image3, cmap='gray'), plt.title('Image Originale 3')
plt.subplot(3, 3, 8), plt.imshow(T1_image3, cmap='gray'), plt.title('Transformation T1')
plt.subplot(3, 3, 9), plt.imshow(T2_image3, cmap='gray'), plt.title('Transformation T2')

plt.tight_layout()
plt.show()
