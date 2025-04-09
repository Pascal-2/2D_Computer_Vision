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

def bw_histogram(img, cumulative=False, norm=False):
    hist = np.zeros(256, dtype=int)
    for i in range(len(img)):
        for j in range(len(img[0])):
            hist[img[i][j]] += 1

    if cumulative:
        for i in range(1, len(hist)):
            hist[i] += hist[i - 1]

    if norm:
        all = len(img) * len(img[0])
        hist /= all

    return hist

def grayscale(imge_rgb):
    bw_img = []
    for row in imge_rgb:
        new_row = []
        for pixel in row:
            avg = sum([int(x) for x in pixel]) // 3
            new_row.append(avg)
        bw_img.append(np.array(new_row, dtype="uint8"))
    return np.array(bw_img)

def mirror(image, isVertical=True):
    if isVertical:
        return image[:, ::-1]
    else:
        return image[::-1]
