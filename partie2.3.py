import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) * (255 / (max_val - min_val))
    return normalized.astype(np.uint8)

image1 = cv2.imread('lenam.bmp', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('hrec.BMP', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('objectm.bmp', cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None or image3 is None:
    print("Erreur : Impossible de charger une ou plusieurs images.")
    exit()

norm_image1 = normalize_image(image1)
norm_image2 = normalize_image(image2)
norm_image3 = normalize_image(image3)

def show_image_histogram(original, transformed, title):
    plt.figure(figsize=(12, 4))

    # Image originale
    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f"{title} - Originale")

    # Histogramme de l'image originale
    plt.subplot(1, 4, 2)
    plt.hist(original.ravel(), bins=256, range=(0, 255), color='black')
    plt.title("Histogramme Original")

    # Image transformée
    plt.subplot(1, 4, 3)
    plt.imshow(transformed, cmap='gray')
    plt.title(f"{title} - Transformée")

    # Histogramme de l'image transformée
    plt.subplot(1, 4, 4)
    plt.hist(transformed.ravel(), bins=256, range=(0, 255), color='black')
    plt.title("Histogramme Transformé")

# Affichage des images et histogrammes
show_image_histogram(image1, norm_image1, "lenam.bmp")
show_image_histogram(image2, norm_image2, "hrec.BMP")
show_image_histogram(image3, norm_image3, "objectm.bmp")
plt.show()