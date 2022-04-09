import cv2
import tifffile
from morphology.utils import *


if __name__ == '__main__':
    I0 = cv2.imread('./rice.png', 0)  # [0, 255]
    I_marker = I0 - 45
    SE = txt2Array('./txt/se/se_3.txt')

    Ir_k = I_marker
    temp = np.zeros_like(Ir_k)
    i = 0

    while (temp != Ir_k).any():
        temp = Ir_k.copy()
        Ir_k = MAndGray(MDilateGray(Ir_k, SE), I0)
        i += 1
        if i == 5:
            break

    print(i)
    tifffile.imwrite('./temp.tif', Ir_k)