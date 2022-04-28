import numpy as np
from numba import cuda
import math


def txt2Array(txt_path: str) -> np.array:
    """
        '.txt' file to array, deal with image
    """
    with open(txt_path, 'r') as f:
        content = f.read().split('\n')[:-1]  # drop '\n'
    image = []
    for i in range(len(content)):
        image.append(content[i].split(','))
    return np.array(image, dtype=np.int32)


@cuda.jit(nopython=True)
def InputMembershipFunc(d):
    """
        Input fuzzy set membership function.
        Membership function is modeled as a truncated gaussian distribution.
        \mu_zero = 0                       if |d| > 2 * \sigma
        \mu_zero = exp(-d^2/(2\sigma^2))   if |d| <= 2 * \sigma
    """
    sigma = 7
    if d > 2 * sigma or d < -2 * sigma:
        return 0
    else:
        return math.e ** (-d**2 / (2 * sigma**2))


@cuda.jit(nopython=True)
def OutputMembershipFunc(z, mode):
    """
        Output fuzzy set membership function.
        mode = ('black', 'while')
        \mu_black = 0                if z > 180
        \mu_black = (180-z) / 180    if z <= 180
        \mu_while = 0                if z < 75
        \mu_while = (z-75) / 180     if z > 75
    """
    if mode == 'black':
        if z > 180:
            return 0
        else:
            return (180-z) / 180

    elif mode == 'white':
        if z < 75:
            return 0
        else:
            return (z-75) / 180


@cuda.jit
def sub_fb(boundary_device, I_device, shape: tuple):
    """
        fuzzy boundary extraction kernel function
    """
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x  # H
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y  # W
    if i >= shape[0] - 1 or j >= shape[1] - 1 or i == 0 or j == 0:  # boundary, carefully determined
        return

    # gray neighbourhood
    d2 = I_device[i-1, j] - I_device[i, j]
    d4 = I_device[i, j-1] - I_device[i, j]
    d6 = I_device[i, j+1] - I_device[i, j]
    d8 = I_device[i+1, j] - I_device[i, j]

    mu_zero_d2, mu_zero_d6 = InputMembershipFunc(d2), InputMembershipFunc(d6)
    mu_zero_d4, mu_zero_d8 = InputMembershipFunc(d4), InputMembershipFunc(d8)

    sum_z5, sum_mu = 0, 0
    for z5 in range(255):
        mu_white_z5 = OutputMembershipFunc(z5, 'white')
        mu1 = min(mu_zero_d2, mu_zero_d6, mu_white_z5)  # mu1
        mu2 = min(mu_zero_d6, mu_zero_d8, mu_white_z5)  # mu2
        mu3 = min(mu_zero_d8, mu_zero_d4, mu_white_z5)  # mu3
        mu4 = min(mu_zero_d4, mu_zero_d2, mu_white_z5)  # mu4
        mu5 = OutputMembershipFunc(z5, 'black')  # mu5
        mu = max(mu1, mu2, mu3, mu4, mu5)
        sum_z5 += z5 * mu
        sum_mu += mu

    boundary_device[i, j] = int(sum_z5 / sum_mu)


def fuzzyBoundary(I: np.array) -> np.array:
    """
        CUDA accelerated fuzzy boundary extraction
    """
    h, w = I.shape
    boundary = I.copy()

    BlockThread = (16, 16)  # use 16x16 threads concurrent
    GridBlock = (h // BlockThread[0] + 1,
                 w // BlockThread[1] + 1)  # do H*W times

    boundary_device = cuda.to_device(boundary)
    I_device = cuda.to_device(I)
    sub_fb[GridBlock, BlockThread](boundary_device, I_device, (h, w))
    cuda.synchronize()
    boundary = boundary_device.copy_to_host()

    return boundary


def gray2binary(image: np.array, threshold: int) -> np.array:
    """
        Gray image to binary image, return int32 type
    """
    img = np.zeros_like(image, dtype=np.int32)
    img[image <= threshold] = 0
    img[image > threshold] = 1
    return img


def Otsu_seg(I: np.array, pixelRange=255) -> np.array:
    """
        Otsu segmentation.
        For image in range [0, 255] as default
    """
    var_max = -1
    best_th = -1
    for th in range(0, pixelRange):
        N0 = np.sum(I < th)  # back
        N1 = np.sum(I >= th)  # fore
        if N0 == 0:  # all is fore, th not enough
            continue
        if N1 == 0:  # all is back, th too high
            break

        N = len(I)
        p0 = N0 / N
        p1 = N1 / N  # pixels proportion
        mu0 = np.sum(I[I < th]) / N0
        mu1 = np.sum(I[I >= th]) / N1
        mu = np.sum(I) / N  # mean pixel value
        var = p0*(mu0-mu)**2 + p1*(mu1-mu)**2
        if var > var_max:  # return the smallest index
            var_max = var
            best_th = th

    return gray2binary(I, best_th)


@cuda.jit
def sub_fs(S_device, I_device, param: tuple, shape: tuple):
    """
        fuzzy segmentation kernel function
    """
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x  # H
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y  # W
    if i >= shape[0] or j >= shape[1]:  # boundary, carefully determined
        return

    mu0, mu1, C, th = param
    mu = mu0 if I_device[i, j] < th else mu1
    abs_minus = I_device[i, j] - mu if I_device[i, j] - mu >= 0 else mu - I_device[i, j]
    mu_x = 1 / (1 + (abs_minus + 1e-5) / C)
    S_device[i, j] = -mu_x * math.log(mu_x) - (1 - mu_x) * math.log(1 - mu_x)


def fuzzy_seg(I: np.array, pixelRange=255) -> np.array:
    """
        fuzzy methods segmentation.
        For image in range [0, 255] as default
    """
    S_min = float('inf')
    C = pixelRange + 1
    h, w = I.shape
    best_th = -1

    BlockThread = (16, 16)  # use 16x16 threads concurrent
    GridBlock = (h // BlockThread[0] + 1,
                 w // BlockThread[1] + 1)  # do H*W times

    for th in range(pixelRange):
        N0 = np.sum(I < th)  # back
        N1 = np.sum(I >= th)  # fore
        if N0 == 0:  # all is fore, th not enough
            continue
        if N1 == 0:  # all is back, th too high
            break
        mu0 = np.sum(I[I < th]) / N0
        mu1 = np.sum(I[I >= th]) / N1

        S = np.zeros_like(I, dtype=np.float32)
        # CUDA accelerate
        S_device = cuda.to_device(S)
        I_device = cuda.to_device(I)
        param = cuda.to_device(np.array([mu0, mu1, C, th]))
        sub_fs[GridBlock, BlockThread](S_device, I_device, param, (h, w))
        cuda.synchronize()
        S = S_device.copy_to_host()

        if np.sum(S) < S_min:  # return the smallest index
            S_min = np.sum(S)
            best_th = th

    return gray2binary(I, best_th)

