import matplotlib.pyplot as plt
import numpy as np
import tifffile
from perspectiveTrans.utils import *


if __name__ == '__main__':
    img_1 = txt2Array('./txt/img_1.txt')  # (474, 474)
    img_2 = txt2Array('./txt/img_2.txt')  # (474, 474)

    y1, x1, y2, x2 = get_index('./txt/index.txt')
    # plt.imshow(img_1, cmap='gray')
    # plt.scatter(y1, x1)
    # plt.show()
    # plt.imshow(img_2, cmap='gray')
    # plt.scatter(y2, x2)
    # plt.show()

    n = len(y1)  # 10
    A = np.zeros((n * 2, 8), dtype=np.float32)
    for i in range(n):
        A[2*i][0:3] = np.array([x2[i], y2[i], 1])
        A[2*i][6:8] = np.array([-x2[i]*x1[i], -y2[i]*x1[i]])

        A[2*i+1][3:6] = np.array([x2[i], y2[i], 1])
        A[2*i+1][6:8] = np.array([-x2[i]*y1[i], -y2[i]*y1[i]])

    C = np.zeros(n * 2)
    for i in range(n):
        C[2*i] = x1[i]
        C[2*i+1] = y1[i]

    p = np.linalg.pinv(A.T @ A) @ A.T @ C
    P = np.append(p, 1).reshape(3, 3)

    img_1_ = np.zeros_like(img_2)
    for i in range(img_1_.shape[0]):
        for j in range(img_1_.shape[1]):
            p_ = P @ np.array([i, j, 1])
            x, y = p_[0] / p_[2], p_[1] / p_[2]
            kx, ky = int(x), int(y)

            if 0 < kx < img_1.shape[0] and 0 < ky < img_1.shape[1]:
                img_1_[i, j] = img_1[kx, ky]

    tifffile.imwrite('./img_1_.tif', img_1_)
    img_3 = image_fusion(img_1_, img_2)
    tifffile.imwrite('./img_3.tif', img_3)

