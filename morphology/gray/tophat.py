import cv2
import tifffile
from morphology.utils import *


if __name__ == '__main__':
    I0 = cv2.imread('./rice.png', 0)  # [0, 255]
    # I0_th = gray2binary(rice, 150)  # [0, 1]
    SE = txt2Array('./txt/se/se_1.txt')

    I_tophat = I0 - MOpenGray(I0, SE)
    I_tophat_th = gray2binary(I_tophat, 60)  # [0, 1]

    SE2 = txt2Array('./txt/se/se_2.txt')
    temp = MOpen(I_tophat_th, SE2)  # binary image
    tifffile.imwrite('./temp.tif', temp)