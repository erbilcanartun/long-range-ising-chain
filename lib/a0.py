import numpy as np
from itertools import product
from numba import njit

# ----------------------------
# Majority-rule blocks (same as before)
# ----------------------------
_all_spins = np.array(list(product([-1,1], repeat=3)), dtype=np.int64)
plus_configs  = _all_spins[np.sum(_all_spins, axis=1) >=  1]
minus_configs = _all_spins[np.sum(_all_spins, axis=1) <= -1]

# ----------------------------
# 15 interaction pairs (left intracell, right intracell, intercell)
# ----------------------------
all_pairs = np.array([
    (0,1),(1,2),(0,2),
    (3,4),(4,5),(3,5),
    (0,3),(0,4),(0,5),
    (1,3),(1,4),(1,5),
    (2,3),(2,4),(2,5)
], dtype=np.int64)

# ----------------------------
# Energy of 6 spins, coupling J, field H
# ----------------------------
@njit(cache=True)
def cluster_energy(spinsL, spinsR, J, H):
    """
    E = J sum pairs s_i s_j + H*(sum left + sum right)
    """
    s0, s1, s2 = spinsL
    t0, t1, t2 = spinsR

    spins6 = np.array([s0,s1,s2,t0,t1,t2], dtype=np.int64)

    E = 0.0
    for k in range(all_pairs.shape[0]):
        i = all_pairs[k,0]
        j = all_pairs[k,1]
        E += spins6[i]*spins6[j]
    E *= J

    # add field
    E += H*(s0+s1+s2+t0+t1+t2)
    return E

# ----------------------------
# Compute log R++ , R-- , R+-   (field-dependent)
# ----------------------------
@njit(cache=True)
def compute_logs(J, H):
    nP = plus_configs.shape[0]
    nM = minus_configs.shape[0]

    # R(++) ------------------------------------------------
    tmp_pp = np.empty(nP*nP, dtype=np.float64)
    idx=0
    for i in range(nP):
        for j in range(nP):
            tmp_pp[idx] = cluster_energy(plus_configs[i], plus_configs[j], J, H)
            idx+=1
    m = np.max(tmp_pp)
    s = 0.0
    for k in range(tmp_pp.size):
        s += np.exp(tmp_pp[k]-m)
    logRpp = m + np.log(s)

    # R(-- ) ------------------------------------------------
    tmp_mm = np.empty(nM*nM, dtype=np.float64)
    idx=0
    for i in range(nM):
        for j in range(nM):
            tmp_mm[idx] = cluster_energy(minus_configs[i], minus_configs[j], J, H)
            idx+=1
    m = np.max(tmp_mm)
    s = 0.0
    for k in range(tmp_mm.size):
        s += np.exp(tmp_mm[k]-m)
    logRmm = m + np.log(s)

    # R(+-) ------------------------------------------------
    tmp_pm = np.empty(nP*nM, dtype=np.float64)
    idx=0
    for i in range(nP):
        for j in range(nM):
            tmp_pm[idx] = cluster_energy(plus_configs[i], minus_configs[j], J, H)
            idx+=1
    m = np.max(tmp_pm)
    s = 0.0
    for k in range(tmp_pm.size):
        s += np.exp(tmp_pm[k]-m)
    logRpm = m + np.log(s)

    return logRpp, logRmm, logRpm

# ----------------------------
# Scalar RG step: J' and H'
# ----------------------------
@njit(cache=True)
def rg_step_J(J):
    logRpp, _, logRpm = compute_logs(J, 0.0)
    return 0.5*(logRpp - logRpm)

@njit(cache=True)
def rg_step_H(J, H):
    logRpp, logRmm, _ = compute_logs(J, H)
    return 0.25*(logRpp - logRmm)


def rg_flow_a0(J0, n_steps=10):
    Jvals = np.zeros(n_steps + 1, dtype=float)
    Jvals[0] = J0

    J = J0
    for k in range(1, n_steps + 1):
        J = rg_step_J(J)
        Jvals[k] = J

    return Jvals


# ----------------------------
# Derivative dH'/dH at H=0
# ----------------------------
@njit(cache=True)
def dH_dH(J, eps=1e-6):
    Hp0 = rg_step_H(J, 0.0)
    Hp1 = rg_step_H(J, eps)
    return (Hp1 - Hp0)/eps


def magnetization_a0(T, max_steps=12):
    """
    Magnetization for a=0, assuming the RG flow always ends
    in the ordered sink (J -> infinity), so M_sink = 1.
    """
    J = 1.0 / T
    b = 3.0

    # Always ordered sink for this a=0 RG: M_sink = 1
    M_sink = 1.0

    prod = 1.0
    Jcurr = J

    for _ in range(max_steps):
        lam = dH_dH(Jcurr)
        if not np.isfinite(lam) or lam <= 0.0:
            # If the scaling factor becomes pathological,
            # we stop accumulating further RG steps.
            break
        prod *= lam
        Jcurr = rg_step_J(Jcurr)

    M = (b**(-max_steps)) * prod * M_sink
    return float(M)


def sweep_magnetization(Tmin=0.1, Tmax=5, N=50):
    Ts = np.linspace(Tmin, Tmax, N)
    Ms = np.zeros(N)
    for i in range(N):
        Ms[i] = magnetization_a0(Ts[i], max_steps=5)
    return Ts, Ms
    