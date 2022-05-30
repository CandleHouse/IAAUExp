import numpy as np


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


def get_index(txt_path) -> np.array:
    """
        '.txt' file to array, deal with index
    """
    with open(txt_path, 'r') as f:
        content = f.read().split('\n')[1: -1]

    for i in range(len(content)):
        content[i] = content[i].split()

    content = np.array(content, dtype=np.int32)

    return content[:, 0], content[:, 1], content[:, 2], content[:, 3]  # y1, x1, y2, x2


def image_fusion(img1: np.array, img2: np.array) -> np.array:
    return np.maximum(img1, img2)