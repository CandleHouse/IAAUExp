import tifffile
from fuzzyMethods.utils import *


if __name__ == '__main__':
    squares = txt2Array('./txt/squares.txt')  # (267, 256)

    squares_seg = Otsu_seg(squares)
    tifffile.imwrite('./Otsu_seg.tif', squares_seg)

    squares_seg2 = fuzzy_seg(squares)
    tifffile.imwrite('./fuzzy_seg.tif', squares_seg2)