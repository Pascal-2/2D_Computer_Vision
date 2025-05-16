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

def read_bw_img(path:str):
    return grayscale(io.imread(path))

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

def grayscale(img_rgb):

    bw_img = np.empty((len(img_rgb), len(img_rgb[0])), dtype=np.uint8)

    for i in range(len(img_rgb)):
        for j in range(len(img_rgb[0])):
            bw_img[i][j] = np.sum(img_rgb[i][j]) // 3
    return bw_img

def mirror(image, isVertical=True):
    if isVertical:
        return image[:, ::-1]
    else:
        return image[::-1]


#function for image plotting
def show_comparison(original, processed, title1, title2):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title(title1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap="gray")
    plt.title(title2)
    plt.axis("off")
