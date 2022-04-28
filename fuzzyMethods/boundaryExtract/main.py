import tifffile
from fuzzyMethods.utils import *


if __name__ == '__main__':
    ### Input membership function
    # mu_zero = list(map(lambda d: InputMembershipFunc(d), range(-100, 100)))
    ### Output membership function
    # mu_black = list(map(lambda z: OutputMembershipFunc(z, mode='black'), range(255)))
    # mu_white = list(map(lambda z: OutputMembershipFunc(z, mode='white'), range(255)))

    men = txt2Array('./txt/men.txt')  # (363, 381)
    boundary = fuzzyBoundary(men)
    tifffile.imwrite('./boundary.tif', boundary)

