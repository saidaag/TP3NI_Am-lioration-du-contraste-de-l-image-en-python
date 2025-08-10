import numpy as np
import matplotlib.pyplot as plt


image = np.array([
    [0, 140, 51, 191, 140, 51],
    [0, 51, 191, 140, 140, 51],
    [51, 140, 20, 20, 140, 0],
    [51, 140, 20, 20, 20, 140],
    [0, 140, 191, 0, 20, 51],
    [0, 10, 51, 10, 140, 51]
], dtype=np.uint8)


def compute_histogram(image):
    hist = np.zeros(256, dtype=int)  
    for row in image:
        for pixel in row:
            hist[pixel] += 1 
    return hist

#Calcule
histogram = compute_histogram(image)

# Affichage 
plt.figure(figsize=(8, 5))
plt.bar(range(256), histogram, color='gray', edgecolor='black')
plt.title("Histogramme de l'image (Calcul manuel)")
plt.xlabel("Valeurs de pixel (0-255)")
plt.ylabel("Fréquence")
plt.xlim([0, 255])  
plt.show()
