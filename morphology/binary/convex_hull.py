import time
import tifffile
from crip.io import *
from morphology.utils import *


if __name__ == '__main__':
    I0 = txt2Array('./txt/butterfly.txt')  # (325, 340)
    SE = txt2Array('./txt/se_1.txt')
    Im = txt2Array('./txt/marker.txt')

    # marker
    I0_c = ComplementarySet(I0)
    Ir_k = Im
    temp = np.zeros_like(Ir_k)
    while (temp != Ir_k).any():
        temp = Ir_k.copy()
        Ir_k = MDilateGPU(Ir_k, SE) * I0_c
    I1 = Ir_k.copy()  # 29 iterate

    B = list(map(lambda x: txt2Array(x), listDirectory(r'./txt/convex_hull_se/', style='fullpath')))

    # convex hull
    Ich1k = I1.copy()
    temp2 = np.zeros_like(Ich1k)
    i = 0
    start = time.time()

    while (temp2 != Ich1k).any():
        temp2 = Ich1k.copy()
        Ich1k = MOr(MErodeGPU(Ich1k, B[0]) * MErodeGPU(ComplementarySet(Ich1k), B[1]), Ich1k)  # <= change index manually
        i += 1
        if i % 10 == 0:
            print('{} iterations'.format(i))

    print('Total iterations: {}'.format(i))
    print('Time spend: {}'.format(time.time() - start))
    tifffile.imwrite('./temp.tif', Ich1k)

    # Union
    # temp_convex_hull = list(map(lambda x: tifffile.imread(x), listDirectory('./output/convex_hull_temp/', style='fullpath')))
    # convex_hull = MOr(MOr(temp_convex_hull[0], temp_convex_hull[1]), MOr(temp_convex_hull[2], temp_convex_hull[3]))
    # tifffile.imwrite('./temp.tif', convex_hull)
