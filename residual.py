import numpy as np
from cca import *

def residual(xs_1, xs_2):

    x1 = xs_1[int(1500):, :]
    x2 = xs_2[int(1500):, :]

    A, B, Sigma = cca_model(x1, x2)

    res = np.dot(B.T, xs_2[:1500, :].T) - \
         np.dot(Sigma.T, np.dot(A.T, xs_1[:1500, :].T))

    return res

