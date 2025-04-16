from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.typing as npt
import my_lib

def filter1(img: npt.NDArray[np.uint8], filter: npt.NDArray[np.float64], off: int):
    border = len(filter[0]) // 2
    filtered_img = np.empty(((len(img) - 2 * border) // off, (len(img[0]) - 2 * border) // off), dtype=np.uint8)
    for i in range(border, len(img) - border, off):
        for j in range(border, len(img[0]) - border, off):
            cur_px = 0.

            for k in range(len(filter)):
                for l in range(len(filter)):
                    cur_px += img[i - border + k, j - border + l] * filter[k, l]

            filtered_img[i//off - border][j//off - border] = cur_px
    return filtered_img

img = my_lib.read_bw_img("lena.jpg")
filter_t1 = np.array([[0., 0., 0.],
                   [3., 3., 3.],
                   [0., 0., 0.]], dtype=np.float16) / 9
filter_t2 = np.array([[1, 1, 1, 1, 1, 1, 5],
                    [1, 1, 1, 1, 1, 5, 5],
                    [1, 1, 1, 1, 5, 5, 5],
                    [1, 1, 1, 5, 5, 5, 5],
                    [1, 1, 5, 5, 5, 5, 5],
                    [1, 5, 5, 5, 5, 5, 5],
                    [5, 5, 5, 5, 5, 5, 5]]) / (21 + 28*5)
test = filter1(img, filter_t2, 1)
# io.imsave("./savetest.jpg", test)
my_lib.plot_img(test, True)



def filter2(img, filter, off, edge):
    border = len(filter[0]) // 2

    extended_img = np.empty(((len(img) + 2 * border), (len(img[0]) + 2 * border)), dtype=np.uint8)

    if edge == "min" or edge == "max":
        if edge == "min":
            border_val = 0
        else:
            border_val = 255

        for i in range(border):
            for j in range(len(extended_img[0])):
                extended_img[i][j] = border_val

        for i in range(border, len(img) + border):
            for j in range(border):
                extended_img[i, j] = border_val
            for j in range(border, border + len(img[0])):
                extended_img[i, j] = img[i - border, j - border]
            for j in range(border + len(img[0]), len(extended_img[0])):
                extended_img[i, j] = border_val

        for i in range(border + len(img), len(extended_img)):
            for j in range(len(extended_img[0])):
                extended_img[i][j] = border_val

    else:

        for i in range(border, len(img) + border):
            for j in range(border):
                extended_img[i, j] = img[i - border, border]
            for j in range(border, border + len(img[0])):
                extended_img[i, j] = img[i - border, j - border]
            for j in range(border + len(img[0]), len(extended_img[0])):
                extended_img[i, j] = img[i - border, len(img[0]) - 1]

        for i in range(border):
            for j in range(len(extended_img[0])):
                extended_img[i, j] = extended_img[border, j]

        for i in range(border + len(img), len(extended_img)):
            for j in range(len(extended_img[0])):
                extended_img[i, j] = extended_img[border + len(img) -1, j]
        my_lib.plot_img(extended_img, True)
        io.imsave("./extended_img.png", extended_img)
    return filter1(extended_img, filter, off)


test = filter2(img, filter_t2, 2, "continue")
io.imsave("./savetest.jpg", test)



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

"""
if __name__ == "__main__":

    # read img
    img = io.imread("lena.jpg")

    # convert to numpy array
    img = np.array(img)

    # convert to grayscale
    img = rgb2gray(img)

    fm = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    imgOut = filter2(img, fm , 0, 'min')

    # plot img
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img, cmap = cm.Greys_r)
    # plot imgOut
    plt.figure(1)
    plt.subplot(212)
    plt.imshow(imgOut, cmap = cm.Greys_r)

    plt.show()

"""