import numpy as np
from numba import cuda


def txt2Array(txt_path: str) -> np.array:
    """
        '.txt' file to array, deal with image and structural element
    """
    with open(txt_path, 'r') as f:
        content = f.read().split('\n')[:-1]  # drop '\n'
    image = []
    for i in range(len(content)):
        image.append(content[i].split(','))
    return np.array(image, dtype=np.int32)


def ComplementarySet(I0: np.array, mode='binary') -> np.array:
    """
        :return I0 complementary set
        Support binary image or gray image
    """
    if mode == 'binary':
        return 1 - I0
    elif mode == 'gray':
        return 255 - I0
    else:
        print('Wrong mode')


def MErode(I0: np.array, SE: np.array) -> np.array:
    """
        Binary image
        - Do Morphology erode for one time
    """
    H, W = I0.shape
    h, w = SE.shape
    I = I0.copy()
    nPadU, nPadD, nPadL, nPadR = h//2, h//2, w//2, w//2
    temp = np.pad(I, ((nPadU, nPadD), (nPadL, nPadR)), mode='reflect')

    for i in range(nPadU, H+nPadD):  # 'temp' as axis
        for j in range(nPadL, W+nPadR):
            if np.sum(SE[SE > 0] * temp[i-nPadU: i+nPadD+1, j-nPadL: j+nPadR+1][SE > 0]) < np.sum(SE[SE > 0]):
                I[i-nPadU, j-nPadL] = 0
            else:
                I[i-nPadU, j-nPadL] = 1
    return I


@cuda.jit
def sub_erode(SE_device, temp_device, I_device, shape: tuple, param: tuple):
    """
        Accelerate kernel function.
    """
    nPadU, nPadD, nPadL, nPadR = param
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x  # H
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y  # W
    if i >= shape[0]+nPadD or j >= shape[1]+nPadR:  # boundary, carefully determined
        return

    sum_SE, sum_inter = 0, 0
    for r in range(i-nPadU, i+nPadD+1):  # same as SE shape
        for c in range(j-nPadL, j+nPadR+1):
            if SE_device[r-(i-nPadU), c-(j-nPadL)] > 0:
                sum_SE += SE_device[r-(i-nPadU), c-(j-nPadL)]
                sum_inter += temp_device[r, c]
    if sum_SE != sum_inter:
        I_device[i-nPadU, j-nPadL] = 0
    else:
        I_device[i-nPadU, j-nPadL] = 1


def MErodeGPU(I0: np.array, SE: np.array) -> np.array:
    """
        Binary image
        - Do Morphology erode for one time, CUDA accelerate
    """
    H, W = I0.shape
    h, w = SE.shape
    I = I0.copy()
    nPadU, nPadD, nPadL, nPadR = h//2, h//2, w//2, w//2
    param = (nPadU, nPadD, nPadL, nPadR)
    temp = np.pad(I, ((nPadU, nPadD), (nPadL, nPadR)), mode='reflect')

    BlockThread = (16, 16)  # use 16x16 threads concurrent
    GridBlock = (H // BlockThread[0] + 1,
                 W // BlockThread[1] + 1)  # do H*W times

    SE_device = cuda.to_device(np.ascontiguousarray(SE))
    temp_device = cuda.to_device(temp)
    I_device = cuda.to_device(I)
    sub_erode[GridBlock, BlockThread](SE_device, temp_device, I_device, (H, W), param)
    cuda.synchronize()
    I = I_device.copy_to_host()

    return I

# def MDilate_(I0: np.array, SE: np.array) -> np.array:
#     """
#         Do Morphology dilate for one time
#         - Old version, not consider background ignorance, drop
#     """
#     H, W = I0.shape
#     h, w = SE.shape
#     I = I0.copy()
#     nPadU, nPadD, nPadL, nPadR = h//2, h//2, w//2, w//2
#     temp = np.pad(I, ((nPadU, nPadD), (nPadL, nPadR)), mode='reflect')
#
#     SE_ = np.fliplr(np.flipud(SE))  # SE reflect
#     for i in range(nPadU, H+nPadD):  # 'temp' as axis
#         for j in range(nPadL, W+nPadR):
#             if np.sum(SE_ * temp[i-nPadU: i+nPadD+1, j-nPadL: j+nPadR+1]) > 0:
#                 I[i-nPadU, j-nPadL] = 1
#     return I


def MDilate(I0: np.array, SE: np.array) -> np.array:
    """
        Binary image
        - Do Morphology dilate for one time
    """
    SE_ = np.fliplr(np.flipud(SE))  # SE reflect
    return ComplementarySet(MErode(ComplementarySet(I0), SE_))


def MDilateGPU(I0: np.array, SE: np.array) -> np.array:
    """
        Binary image
        - Do Morphology dilate for one time, CUDA accelerate
    """
    SE_ = np.fliplr(np.flipud(SE))  # SE reflect
    return ComplementarySet(MErodeGPU(ComplementarySet(I0), SE_))


def MOpen(I0: np.array, SE: np.array) -> np.array:
    """
        Binary image
        :return I0.Erode(SE).Dilate(SE)
    """
    return MDilateGPU(MErodeGPU(I0, SE), SE)


def MClose(I0: np.array, SE: np.array) -> np.array:
    """
        Binary image
        :return I0.Dilate(SE).Erode(SE)
    """
    return MErodeGPU(MDilateGPU(I0, SE), SE)


def MAnd(A: np.array, B: np.array) -> np.array:
    """
        Binary image
        :return A and B
    """
    return A * B


def MOr(A: np.array, B: np.array) -> np.array:
    """
        Binary image
        :return A or B
    """
    return A + B - MAnd(A, B)


def gray2binary(image: np.array, threshold: int) -> np.array:
    """
        Gray image to binary image, return int32 type
    """
    img = np.zeros_like(image, dtype=np.int32)
    img[image <= threshold] = 0
    img[image > threshold] = 1
    return img


def MErodeGray(I0: np.array, SE: np.array) -> np.array:
    """
        Gray image
        - Do Morphology erode for one time
    """
    H, W = I0.shape
    h, w = SE.shape
    I = I0.copy()
    nPadU, nPadD, nPadL, nPadR = h//2, h//2, w//2, w//2
    temp = np.pad(I, ((nPadU, nPadD), (nPadL, nPadR)), mode='reflect')

    for i in range(nPadU, H+nPadD):  # 'temp' as axis
        for j in range(nPadL, W+nPadR):
            I[i-nPadU, j-nPadL] = np.min(temp[i-nPadU: i+nPadD+1, j-nPadL: j+nPadR+1][SE > 0])
    return I


def MDilateGray(I0: np.array, SE: np.array) -> np.array:
    """
        Gray image
        - Do Morphology dilate for one time
    """
    SE_ = np.fliplr(np.flipud(SE))  # SE reflect
    return ComplementarySet(MErodeGray(ComplementarySet(I0, mode='gray'), SE_), mode='gray')


def MOpenGray(I0: np.array, SE: np.array) -> np.array:
    """
        Gray image
        :return I0.Erode(SE).Dilate(SE)
    """
    return MDilateGray(MErodeGray(I0, SE), SE)


def MCloseGray(I0: np.array, SE: np.array) -> np.array:
    """
        Gray image
        :return I0.Dilate(SE).Erode(SE)
    """
    return MErodeGray(MDilateGray(I0, SE), SE)


def MAndGray(A: np.array, B: np.array) -> np.array:
    """
        Gray image
        :return min(A, B)
    """
    return np.minimum(A, B)


def MOrGray(A: np.array, B: np.array) -> np.array:
    """
        Gray image
        :return max(A, B)
    """
    return A + B - MAndGray(A, B)