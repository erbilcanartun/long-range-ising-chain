import numpy as np
import pandas as pd
from numba import njit
import os





@njit(cache=True)
def logsumexp(values):
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m)))


def build_J(J0, a, D):
    J = np.zeros(D + 1)
    r = np.arange(1, D + 1)
    J[1:] = J0 / (r**a)
    return J