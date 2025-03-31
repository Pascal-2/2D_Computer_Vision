from types import new_class

import matplotlib
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# AUFGABE 1
# Bild laden und anzeigen.
image_path = "Images\\BlauGruenRot.jpg"
image1 = io.imread(image_path)

#plt.imshow(image)
#plt.axis('off')
#plt.show()

# Datentyp des Bildes untersuchen
print(image1)
print(image1.shape)
print(type(image1))
# Ergebnis: 600 Zeilen mal 600 Spalten, jeder Pixel mit 3 Werten (RGB).

# Farbkanäle einzeln anzeigen
kanalRot = np.array([np.array([np.array([y[0],0,0]) for y in x]) for x in image1])
plt.imshow(kanalRot)
plt.axis('off')
plt.show()

kanalGruen = np.array([np.array([np.array([0,y[1],0]) for y in x]) for x in image1])
plt.imshow(kanalGruen)
plt.axis('off')
plt.show()

kanalBlau = np.array([np.array([np.array([0,0,y[2]]) for y in x]) for x in image1])
plt.imshow(kanalBlau)
plt.axis('off')
plt.show()


# Spiegelfunktion (wahlweise Zeilen, oder Spalten umsortieren)
def mirror(image, isVertical=True):
    if isVertical:
        new_image = [list(row) for row in image]
        for i in range(len(new_image)):
            new_image[i].reverse()
        return np.array([np.array(row) for row in new_image])
    else:
        new_image = [row for row in image]
        new_image.reverse()
        return np.array(new_image)

verticalTest = mirror(image1)
plt.imshow(verticalTest)
plt.axis('off')
plt.show()

pointReflection = mirror(mirror(image1), False)
plt.imshow(pointReflection)
plt.axis('off')
plt.show()

horizontalTest = mirror(image1, False)
plt.imshow(horizontalTest)
plt.axis('off')
plt.show()


