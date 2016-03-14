#!/usr/bin/python
import numpy as np
import gpcuda

N = 1e3

a = np.linspace(0,10,N)
b = np.zeros(N)
gpcuda.cos_doubles_func(a, b)

c = np.cos(a)

assert np.allclose(c,b)

