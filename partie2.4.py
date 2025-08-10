import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


image_filenames = ["lenam.bmp", "objectm.bmp", "hherc.BMP"]

for img_name in image_filenames:

    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Erreur : Impossible de charger {img_name}")
        continue

    img_eq_cv2 = cv2.equalizeHist(img)

    img_eq_clahe = exposure.equalize_adapthist(img, clip_limit=0.03)  # Normalisation entre [0,1]
    img_eq_clahe = (img_eq_clahe * 255).astype(np.uint8)  # Conversion en uint8

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # Image originale + Histogramme
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f"{img_name} - Originale")
    axes[0, 1].hist(img.ravel(), bins=256, histtype='step', color='black')
    axes[0, 1].set_title("Histogramme Original")

    # Égalisation OpenCV + Histogramme
    axes[1, 0].imshow(img_eq_cv2, cmap='gray')
    axes[1, 0].set_title(f"{img_name} - Égalisation OpenCV")
    axes[1, 1].hist(img_eq_cv2.ravel(), bins=256, histtype='step', color='black')
    axes[1, 1].set_title("Histogramme OpenCV")

    # Égalisation Adaptative + Histogramme
    axes[2, 0].imshow(img_eq_clahe, cmap='gray')
    axes[2, 0].set_title(f"{img_name} - Égalisation CLAHE")
    axes[2, 1].hist(img_eq_clahe.ravel(), bins=256, histtype='step', color='black')
    axes[2, 1].set_title("Histogramme CLAHE")

    for i in range(3):
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
