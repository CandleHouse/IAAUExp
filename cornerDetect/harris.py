import cv2
import numpy as np
import tifffile


if __name__ == '__main__':
    I0 = cv2.imread('./rice.png', 0)  # (256, 256)
    I0 = np.array(I0, dtype=np.float32)
    kernel_x = np.array([[0, 0, 0],
                         [-1, 0, 1],
                         [0, 0, 0]])
    kernel_y = np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, -1, 0]])

    I_x = cv2.filter2D(I0, -1, kernel_x) / 2
    I_y = cv2.filter2D(I0, -1, kernel_y) / 2

    kernel_sum = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    A = cv2.filter2D(I_x ** 2, -1, kernel_sum)
    B = cv2.filter2D(I_x * I_y, -1, kernel_sum)
    C = cv2.filter2D(I_y ** 2, -1, kernel_sum)
    M = np.vstack((np.hstack((A, B)), np.hstack((B, C))))

    k = 0.02
    R = A * C - B * B - k * (A + C) * (A + C)

    R_max = np.max(R)
    CORNER = np.zeros_like(R)
    CORNER[R > 0.2 * R_max] = 1
    tifffile.imwrite('./temp.tif', CORNER * 100 + I0)

