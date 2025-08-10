import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('lenam.bmp', cv2.IMREAD_GRAYSCALE)

image2 = cv2.imread('hrec.BMP', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('objectm.bmp', cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None or image3 is None:
    print("Erreur : Impossible de charger une ou plusieurs images.")
else:

    def plot_histogram(image, title):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.title(title)
        plt.xlabel("Valeurs de pixel (0-255)")
        plt.ylabel("Fréquence")
        plt.xlim([0, 255])
        plt.grid()

    plt.figure(figsize=(15, 5))

    # Image 1
    plt.subplot(1, 3, 1)
    plot_histogram(image1, "Histogramme - lenam.bmp")

    # Image 2
    plt.subplot(1, 3, 2)
    plot_histogram(image2, "Histogramme - hherc.BMP")

    # Image 3
    plt.subplot(1, 3, 3)
    plot_histogram(image3, "Histogramme - objectm.bmp")

    plt.tight_layout()
    plt.show()
