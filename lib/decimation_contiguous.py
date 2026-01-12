import numpy as np
from numba import njit
from itertools import product
from utils import logsumexp


@njit(cache=True)
def required_initial_max_distance(max_dist_final, n_steps):
    D = max_dist_final
    for _ in range(n_steps):
        D = 3 * D + 2
    return D


@njit(cache=True)
def r_max(D):
    return (D - 2) // 3


# Majority configurations (3-spin cell)
_all_spins = np.array(list(product([-1, 1], repeat=3)), dtype=int)
plus_configs  = _all_spins[np.sum(_all_spins, axis=1) >=  1]
minus_configs = _all_spins[np.sum(_all_spins, axis=1) <= -1]


@njit(cache=True)
def intracell_energies(spins, J):
    J1 = J[1] if len(J) > 1 else 0.0
    J2 = J[2] if len(J) > 2 else 0.0

    n = spins.shape[0]
    E = np.empty(n, dtype=np.float64)
    for i in range(n):
        s0 = spins[i, 0]
        s1 = spins[i, 1]
        s2 = spins[i, 2]
        E[i] = (J1 * (s0 * s1 + s1 * s2) + J2 * (s0 * s2))
    return E


@njit(cache=True)
def log_Rpp_Rpm(r, J):
    D = len(J) - 1

    # Intracell energies (cell geometry is distance-1 spacing: {1,1,2})
    E_plus  = intracell_energies(plus_configs,  J)
    E_minus = intracell_energies(minus_configs, J)

    # Physical lattice positions
    left_pos  = np.array([1, 2, 3], dtype=np.int64)
    right_pos = (np.array([4, 5, 6], dtype=np.int64) + 3 * (r - 1)).astype(np.int64)

    # Distance matrix
    distances = np.empty((3, 3), dtype=np.int64)
    for a in range(3):
        for b in range(3):
            distances[a, b] = abs(right_pos[b] - left_pos[a])

    n_plus  = plus_configs.shape[0]
    n_minus = minus_configs.shape[0]

    # R(++)
    totals_pp = np.empty(n_plus * n_plus, dtype=np.float64)
    idx = 0
    for iL in range(n_plus):
        sL = plus_configs[iL]
        EL = E_plus[iL]
        for iR in range(n_plus):
            sR = plus_configs[iR]
            ER = E_plus[iR]

            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d <= D:
                        E_int += J[d] * sL[a] * sR[b]

            totals_pp[idx] = EL + ER + E_int
            idx += 1

    # R(+-)
    totals_pm = np.empty(n_plus * n_minus, dtype=np.float64)
    idx = 0
    for iL in range(n_plus):
        sL = plus_configs[iL]
        EL = E_plus[iL]
        for iR in range(n_minus):
            sR = minus_configs[iR]
            ER = E_minus[iR]

            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d <= D:
                        E_int += J[d] * sL[a] * sR[b]

            totals_pm[idx] = EL + ER + E_int
            idx += 1

    log_pp = logsumexp(totals_pp)
    log_pm = logsumexp(totals_pm)

    return log_pp, log_pm


@njit(cache=True)
def log_Rpp_Rmm_nonzero_H(J, H):
    E_plus  = intracell_energies(plus_configs,  J)
    E_minus = intracell_energies(minus_configs, J)

    left_pos  = np.array([1, 2, 3], dtype=np.int64)
    right_pos = np.array([4, 5, 6], dtype=np.int64)

    # Distances between spins in left and right blocks
    distances = np.empty((3, 3), dtype=np.int64)
    for a in range(3):
        for b in range(3):
            distances[a, b] = abs(right_pos[b] - left_pos[a])

    n_plus  = plus_configs.shape[0]
    n_minus = minus_configs.shape[0]
    D = len(J) - 1

    # R(++): left-majority +, right-majority +
    totals_pp = np.empty(n_plus * n_plus, dtype=np.float64)
    idx = 0
    for iL in range(n_plus):
        sL = plus_configs[iL]
        EL = E_plus[iL]
        magL = sL[0] + sL[1] + sL[2]
        for iR in range(n_plus):
            sR = plus_configs[iR]
            ER = E_plus[iR]
            magR = sR[0] + sR[1] + sR[2]

            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d <= D:
                        E_int += J[d] * sL[a] * sR[b]

            totals_pp[idx] = EL + ER + E_int + H * (magL + magR)
            idx += 1

    # R(--): left-majority -, right-majority -
    totals_mm = np.empty(n_minus * n_minus, dtype=np.float64)
    idx = 0
    for iL in range(n_minus):
        sL = minus_configs[iL]
        EL = E_minus[iL]
        magL = sL[0] + sL[1] + sL[2]
        for iR in range(n_minus):
            sR = minus_configs[iR]
            ER = E_minus[iR]
            magR = sR[0] + sR[1] + sR[2]

            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d <= D:
                        E_int += J[d] * sL[a] * sR[b]

            totals_mm[idx] = EL + ER + E_int + H * (magL + magR)
            idx += 1

    log_pp = logsumexp(totals_pp)
    log_mm = logsumexp(totals_mm)

    return log_pp, log_mm