from morphology.utils import *
import tifffile


if __name__ == '__main__':
    I0 = txt2Array('./txt/butterfly.txt')  # (325, 340)
    SE = txt2Array('./txt/se_1.txt')

    temp = MOpen(ComplementarySet(I0), SE)
    temp2 = ComplementarySet(MClose(I0, SE))

    tifffile.imwrite('./temp.tif', temp - temp2)
