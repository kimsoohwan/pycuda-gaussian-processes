#!/usr/bin/python
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1,
                                 flags='CONTIGUOUS')
libcd = npct.load_library('libgpcuda', ".")
libcd.cos_doubles.restype = None
libcd.cos_doubles.argtypes = [array_1d_double,
                              array_1d_double,
                              c_int]

def cos_doubles_func(in_array, out_array):
    return libcd.cos_doubles(in_array, out_array, len(in_array))
