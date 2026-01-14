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

# Deterministic RNG (Numba compatible)
@njit(cache=True)
def _xorshift64star_next(state):
    # state: uint64
    x = state
    x ^= (x >> np.uint64(12))
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    state = x
    # Multiply by constant (xorshift64*)
    out = x * np.uint64(2685821657736338717)
    return state, out

@njit(cache=True)
def _u01_from_uint64(u):
    # Map uint64 -> float64 in [0,1)
    # Use top 53 bits for IEEE double mantissa
    return ((u >> np.uint64(11)) & np.uint64((1 << 53) - 1)) * (1.0 / float(1 << 53))

@njit(cache=True)
def _rand_idx(state, N):
    state, rnd = _xorshift64star_next(state)
    return state, int(rnd % np.uint64(N))

@njit(cache=True)
def _rand_u01(state):
    state, rnd = _xorshift64star_next(state)
    return state, _u01_from_uint64(rnd)