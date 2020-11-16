import numpy as np


def shift_array(array):
    shifted = np.empty(len(array), dtype=array.dtype)
    shifted[1:] = array[:-1]
    shifted[0] = array[0]
    return shifted
