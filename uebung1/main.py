from types import new_class

import matplotlib
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def plot_img(img, bw=False):
    if bw:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    else:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def plot_hist(hist):
    plt.figure(figsize=(16, 9))
    plt.bar(list(range(256)), hist, width=1)
    plt.show()


# AUFGABE 1
# Bild laden und anzeigen.
image_path = "Images/BlauGruenRot.jpg"
image1 = io.imread(image_path)
"""
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
"""
# • Zunchst muss eine Funktion geschrieben werden, die ein RGB-Bild in ein Graustufenbild (mit nur einem Kanal) umwandelt.

def grayscale(imge_rgb):
    bw_img = []
    for row in imge_rgb:
        new_row = []
        for pixel in row:
            avg = sum([int(x) for x in pixel]) // 3
            new_row.append(avg)
        bw_img.append(np.array(new_row, dtype="uint8"))
    return np.array(bw_img)

def bw_histogram(img, norm=False):
    hist = [0] * 256
    for row in img:
        for px in row:
            hist[px] += 1

    if norm:
        all = len(img) * len(img[0])
        for i in range(len(hist)):
            hist[i] /= all

    return np.array(hist)

"""
bw_test = grayscale(image1)

plt.imshow(bw_test, cmap='gray')
plt.axis('off')
plt.show()

plt.plot(bw_histogram(bw_test, True))
plt.show()
"""
# 4 a) Welche Aufnahmefehler sind in 01 und 03 zu erkennen? Woran ist dies im Histogramm erkennbar?

# Bild 1 (überbelichtet Himmel) und unterbelichtet

# Bild 2 überbelichtet

# Bild 3 stark überbelichtet

# b) das Histogramm wurde nach rechts gestreckt (in den hellen Pixeln geht Information verloren)

# c) der Kontrast wurde erhöht -> Historgramm gestreckt

"""
for i in range(1, 6):
    image_path = f"Images\\bild0{i}.jpg"
    img = io.imread(image_path)
    img_bw = grayscale(img)

    hist = bw_histogram(img_bw)
    plt.figure(figsize=(16, 9))
    plt.bar(list(range(256)), hist, width=1)
    plt.show()
"""


# 5

def apply(lut, img):
    res = []
    for i in range(len(img)):
        row = []
        for j in range(len(img[0])):
            row.append(lut[int(img[i][j])])
        res.append(row)
    return res


def gamma(img):
    K = 256
    aMax = K - 1
    GAMMA = 2.8

    lut = [0] * K
    for a in range(K):
        aa = a / aMax
        bb = pow(aa,GAMMA)
        b = round(bb * aMax)
        lut[a] = b

    return apply(lut, img)

def brightness_lut(img, lvl):
    lut = list(range(lvl, 256 + lvl))
    lut = [x if x <= 255 else 255 for x in lut]
    return apply(lut, img)
"""
image_path = "Images\\bild04.jpg"
image1 = io.imread(image_path)
plot_img(image1, True)
testGamma = brightness_lut(grayscale(image1), 100)
plot_img(testGamma, True)
"""
# 5 a)

def lossless_brightness_bw(bw_img):
    dist = 0
    hist = bw_histogram(bw_img)
    for i in range(255, -1, -1):
        if hist[i] == 0:
            dist += 1
        else:
            break
    return brightness_lut(bw_img, dist)
"""
image_path = "Images\\bild01.jpg"
image1 = io.imread(image_path)
bw_img = grayscale(image1)
hist1 = bw_histogram(bw_img)
plot_img(bw_img, True)
plot_hist(hist1)

lbimg = lossless_brightness_bw(bw_img)
hist2 = bw_histogram(lbimg)
plot_img(lbimg, True)
plot_hist(hist2)
"""

def bild1_hell_kontrast():
    low40 = []
    mid90 = []
    midhigh90 = []
    high36 = []
    for i in range(60):
        low40.append(i * 2)
    for i in range(60, 130):
        mid90.append(i + 60)
    for i in range(130, 220):
        midhigh90.append(((i - 130) // 3) + 60 + 130)
    for i in range(220, 256):
        high36.append(i)
    lut = low40 + mid90 + midhigh90 + high36

    image_path = "Images/bild01.jpg"
    image1 = io.imread(image_path)

    img1_bw = grayscale(image1)
    hist = bw_histogram(img1_bw)
    plot_hist(hist)
    res = apply(lut, img1_bw)
    
    res_hist = bw_histogram(res)
    plot_img(res, True)
    plot_hist(res_hist)

def bild1_hell_shift():
    low140 = []
    high116 = []
    for i in range(140):
        low140.append(i + 80)
    for i in range(140, 256):
        high116.append(int(220 + 36 * ((i - 140) / 116)))
    lut = low140 + high116

    image_path = "Images/bild01.jpg"
    image1 = io.imread(image_path)

    img1_bw = grayscale(image1)
    hist = bw_histogram(img1_bw)
    plot_hist(hist)
    res = apply(lut, img1_bw)
    res_hist = bw_histogram(res)
    plot_img(res, True)
    plot_hist(res_hist)

#bild1_hell_kontrast()
bild1_hell_shift()