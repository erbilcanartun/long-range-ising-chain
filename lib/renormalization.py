import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from utils import logsumexp, build_J
#from decimation_contiguous import *
from decimation_staggered import *


@njit(cache=True)
def determine_r_max(D):
    return r_max(D)

@njit(cache=True)
def rg_step(J, a=None):
    D = len(J) - 1
    r_max = determine_r_max(D)

    J_new = np.zeros_like(J)
    J_new[0] = 0.0

    # Head: exact renormalization
    for r in range(1, r_max + 1):
        log_R_pp, log_R_pm = log_Rpp_Rpm(r, J)
        J_new[r] = 0.5 * (log_R_pp - log_R_pm)

    # Tail: power-law continuation
    if a:
        anchor = J_new[r_max]
        for r in range(r_max + 1, D + 1):
            J_new[r] = anchor * (r_max / r) ** a
    return J_new

@njit(cache=True)
def renormalized_field(J, H):
    # H' = 1/4 [ln Z{++} - ln Z{--}]
    log_pp, log_mm = log_Rpp_Rmm_nonzero_H(J, H)
    return 0.25 * (log_pp - log_mm)

@njit(cache=True)
def G_r_prime(r, J):
    log_R_pp, log_R_pm = log_Rpp_Rpm(r, J)
    G_r = 0.5 * (log_R_pp + log_R_pm)
    return G_r

@njit(cache=True)
def G_prime(J):
    D = len(J) - 1
    r_max = determine_r_max(D)
    G_new = np.zeros(r_max + 1, dtype=np.float64)
    for r in range(1, r_max + 1):
        G_new[r] = G_r_prime(r, J)
    return G_new

def phi_from_G(J_init, max_steps=8, a=None):
    J = J_init.copy()
    phi = 0.0
    for k in range(max_steps):
        G_vec = G_prime(J)
        G_k = np.sum(G_vec)
        phi += (1 / 3**(k+1)) * G_k
        J = rg_step(J, a=a)
    return phi

@njit(cache=True)
def dH_dH(J, eps=1e-6):
    """
    Derivative dH'/dH at H=0, via finite difference.
    """
    H0    = renormalized_field(J, 0.0)
    H_eps = renormalized_field(J, eps)

    if not np.isfinite(H0) or not np.isfinite(H_eps):
        return np.nan
    return (H_eps - H0) / eps

def find_Jc(a, Jlow=1e-2, Jhigh=1e2, max_steps=6, max_dist_final=9,
            tol=1e-5, growth_threshold=1e3, decay_threshold=1e-3):
    if not (0 < a <= 2):
        raise ValueError("a must be in (0,2)")

    def grows(J0):
        # Build full-length J vector big enough to allow max_steps RG steps
        D0 = required_initial_max_distance(max_dist_final, max_steps)
        J = build_J(J0, a, D0)
        J1_initial = abs(J[1])

        for _ in range(max_steps):

            # early decision:
            if abs(J[1]) > growth_threshold:
                return True   # flows to ordered phase
            if abs(J[1]) < decay_threshold:
                return False  # flows to disordered phase

            # apply full RG step
            J = rg_step(J)

        # fallback: check increasing or decreasing tendency
        return abs(J[1]) > J1_initial

    # Bisection on J0: growth → ordered, decay → disordered
    while (Jhigh - Jlow) > tol:
        Jmid = 0.5 * (Jlow + Jhigh)
        if grows(Jmid):
            Jhigh = Jmid
        else:
            Jlow = Jmid
    return 0.5 * (Jlow + Jhigh)

def construct_transfer_matrix(J, r, normalize=True):
    if r >= len(J):
        raise ValueError(f"Cannot build transfer matrix: distance r={r} > max available {len(J)-1}")

    Jr = J[r]
    T = np.array([
        [np.exp(Jr),   np.exp(-Jr)],
        [np.exp(-Jr),  np.exp(Jr)]
    ], dtype=float)

    if normalize:
        T = T / np.max(T)
    return T

def determine_phase_from_TM(T, threshold=0.1):
    L = T[0, 0]
    R = T[0, 1]

    if L > 1 - threshold and R > 1 - threshold:
        return "disorder"
    if L > 1 - threshold and R < threshold:
        return "ferromagnetic"
    return "undetermined"