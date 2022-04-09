import tifffile
from morphology.utils import *


if __name__ == '__main__':
    I0 = txt2Array('./txt/butterfly.txt')  # (325, 340)
    SE = txt2Array('./txt/se_1.txt')
    Im = txt2Array('./txt/marker.txt')

    I0_c = ComplementarySet(I0)
    Ir_k = Im
    temp = np.zeros_like(Ir_k)
    i = 0

    while (temp != Ir_k).any():
        temp = Ir_k.copy()
        Ir_k = MDilateGPU(Ir_k, SE) * I0_c
        i += 1

    print(i)
    tifffile.imwrite('./temp.tif', Ir_k)
