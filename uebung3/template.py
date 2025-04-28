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
Fragen 4) 
1. Arten und Eigenschaften von linearen Filtern:
Glättungsfilter (Tiefpässe): Boxfilter, Gaußfilter
Schärfungsfilter (Hochpässe): Ableitungsfilter, 

Separable Filter (zerlegbar in Vektoren die multipliziert wieder den Filter ergeben 
und deswegen zu schnelleren Berechnungen führen) vs nicht separable Filter.

Filter zur Kantendetektion (auch oft über Ableitung)

Eigenschaft: Gewichtete Summe der Eingangspixelwerte  (-> linear)

5) Der Unterschied zwischen linearen und nicht linearen Filtern liegt darin,
dass mehr als nur eine gewichtete Summe der Eingangspixelwerte berechnet werden kann (-> nicht linear).
"""

# Medianfilter:
# Heapsort ist gut geeignet, wegen effizienter Laufzeit (avg case O(log(n))) und in-place Sortierung.
# Heapsort hat nicht den gleichen Worst-Case wie Quicksort.
def medianFilter(in_img, filter_width, offset = 1):
    if filter_width % 2 == 0:
        print("medianFilter says: The filter_width was updated to the next uneven number!")
        filter_width += 1
    k = filter_width // 2
    out_img = np.empty((((len(in_img)-2*k) // offset),
                        ((len(in_img[0])-2*k) // offset)), dtype=np.uint8)
    for i in range(k, len(in_img) - k, offset):
        for j in range(k, len(in_img[0]) - k, offset):
            filter_array = np.empty((filter_width ** 2))
            f_index = 0
            for f_i in range(i - k, i + k + 1):
                for f_j in range(j - k, j + k + 1):
                    filter_array[f_index] = in_img[f_i][f_j]
                    f_index += 1
            filter_array = np.sort(filter_array, kind='heapsort')
            out_img[(i-k) // offset][(j-k) // offset] = filter_array[len(filter_array)//2]
    return out_img
"""
width = 5
medimage0 = img
medimage1 = medianFilter(medimage0, width)
medimage2 = medianFilter(medimage1, width)
medimage3 = medianFilter(medimage2, width)
my_lib.plot_img(medimage1, True)
my_lib.plot_img(medimage2, True)
my_lib.plot_img(medimage3, True)
"""


def medianRecursion(in_img, max_depth):
    if max_depth == 0:
        return in_img
    return medianRecursion(medianFilter(in_img, 5), max_depth - 1)

my_lib.plot_img(medianRecursion(img, 15), True)

# Vergleich der verschienenen Filter: Glättungsfilter erzeugen Unschärfe
# Medianfilter 'verschluckt' Details und 'füllt' Flächen gleichmäßiger aus, erhält aber größere Kanten.
# Wird der Medianfilter recursiv angewendet, verwandelt sich das Bild immer mehr in kaum detaillierte Flächen.
# Bei hoher Rekursionstiefe wird das Bild immer mehr grau in grau erscheinen.
# Kleine Filter erhalten mehr Details bzw. wirken in einem anderen Frequenzrahmen.
# Große Filter können prinzipiell mehr 'verschmieren' und tiefe Frequenzen 'detektieren'.

pepper_image = my_lib.read_bw_img("pepper.jpg")
out_pepper = medianFilter(pepper_image, 3)
my_lib.plot_img(out_pepper, True)


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