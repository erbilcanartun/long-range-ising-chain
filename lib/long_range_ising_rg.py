import numpy as np
from numba import njit
from itertools import product

# ----------  spin configurations (global, read-only) ----------
all_spins = list(product([-1, 1], repeat=3))
plus_spins  = np.array([s for s in all_spins if sum(s) >= 1], dtype=np.float64)
minus_spins = np.array([s for s in all_spins if sum(s) <= -1], dtype=np.float64)

# ----------  intracell energy (J2 = nearest, J4 = next-nearest) ----------
@njit
def _intracell_energy(spins, J2, J4):
    n = spins.shape[0]
    E = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s0, s1, s2 = spins[i, 0], spins[i, 1], spins[i, 2]
        E[i] = J2*s0*s1 + J2*s1*s2 + J4*s0*s2
    return E

# ----------  log-R for a pair of cells (H included) ----------
@njit
def _logR(spinsL, EL, spinsR, ER, dists, Jarr, H):
    nL, nR = spinsL.shape[0], spinsR.shape[0]
    nd = dists.shape[0]
    tot = np.zeros(nL*nR, dtype=np.float64)
    idx = 0
    for i in range(nL):
        magL = spinsL[i,0] + spinsL[i,1] + spinsL[i,2]
        for j in range(nR):
            magR = spinsR[j,0] + spinsR[j,1] + spinsR[j,2]
            Eint = 0.0
            for k in range(nd):
                iL, iR, d = dists[k]
                Eint += Jarr[d] * spinsL[i,iL] * spinsR[j,iR]
            tot[idx] = EL[i] + ER[j] + Eint + H*(magL + magR)
            idx += 1
    if tot.size == 0: return -np.inf
    mx = np.max(tot)
    s = np.sum(np.exp(tot - mx))
    return np.log(s) + mx if s > 0.0 else -np.inf

# ----------  H' for the nearest-neighbour cell pair (r'=1) ----------
@njit
def _Hprime(Jarr, H):
    J2, J4 = Jarr[2], Jarr[4]
    ELp = _intracell_energy(plus_spins,  J2, J4)
    ELm = _intracell_energy(minus_spins, J2, J4)

    # geometry of the two nearest-neighbour cells
    start = 4
    left  = np.array([1,3,5], dtype=np.int64)
    right = np.array([start, start+2, start+4], dtype=np.int64)
    dists = np.empty((9,3), dtype=np.int64)
    k = 0
    for iL in range(3):
        for iR in range(3):
            dists[k] = (iL, iR, abs(right[iR]-left[iL]))
            k += 1

    log_pp = _logR(plus_spins,  ELp, plus_spins,  ELp, dists, Jarr, H)
    log_mm = _logR(minus_spins, ELm, minus_spins, ELm, dists, Jarr, H)
    if np.isinf(log_pp) or np.isinf(log_mm): return np.nan
    return 0.25*(log_pp - log_mm)

# ----------  dH'/dH at H = 0 (finite difference) ----------
@njit
def dHdH(Jarr, eps=1e-8):
    H0   = _Hprime(Jarr, 0.0)
    Heps = _Hprime(Jarr, eps)
    if np.isnan(H0) or np.isnan(Heps): return np.nan
    return (Heps - H0)/eps

# ----------  renormalised coupling J'(r') (no Python callable) ----------
@njit
def _Jprime(start, Jarr):
    J2 = Jarr[2]
    J4 = Jarr[4]
    left  = np.array([1,3,5], dtype=np.int64)
    right = np.array([start, start+2, start+4], dtype=np.int64)
    dists = np.empty((9,3), dtype=np.int64)
    k = 0
    for iL in range(3):
        for iR in range(3):
            dists[k] = (iL, iR, abs(right[iR]-left[iL]))
            k += 1

    ELp = _intracell_energy(plus_spins,  J2, J4)
    ELm = _intracell_energy(minus_spins, J2, J4)

    log_pp = _logR(plus_spins,  ELp, plus_spins,  ELp, dists, Jarr, 0.0)
    log_pm = _logR(plus_spins,  ELp, minus_spins, ELm, dists, Jarr, 0.0)
    if np.isinf(log_pp) or np.isinf(log_pm): return np.inf
    return 0.5*(log_pp - log_pm)

# ----------  find critical Jc(a) (once) ----------
def find_Jc(a, max_k=1200, tol=1e-8, Jlow=0.1, Jhigh=12.0):
    if not (0 < a < 2): raise ValueError("a must be in (0,2)")

    md = 3*max_k + 10
    rs = np.arange(1, max_k+1, dtype=np.int64)

    def grows(J0):
        Jarr = np.zeros(md+1, dtype=np.float64)
        Jarr[1:] = J0 * np.power(np.arange(1, md+1, dtype=np.float64), -a)
        cur = Jarr.copy()
        for _ in range(8):                     # enough steps to see trend
            Jp = np.empty(max_k, dtype=np.float64)
            for ri in range(max_k):
                Jp[ri] = _Jprime(3*rs[ri]+1, cur)
            cur = np.zeros(md+1, dtype=np.float64)
            cur[1:max_k+1] = Jp
            if max_k > 1 and abs(Jp[1]) > 1e7: return True
            if max_k > 1 and abs(Jp[1]) < 1e-7: return False
        return abs(Jp[1]) > abs(Jarr[2])

    while Jhigh - Jlow > tol:
        Jmid = (Jlow + Jhigh)*0.5
        if grows(Jmid): Jhigh = Jmid
        else:           Jlow  = Jmid
    return (Jlow + Jhigh)*0.5

# Magnetisation by back-propagation
def magnetisation(J0, a, Jc, max_k=800, max_steps=30, eps=1e-8):
    """
    J0 – starting coupling (=1/T)
    a  – power-law exponent
    Jc – critical coupling for this a (from find_Jc)
    """
    b, d = 3.0, 1.0
    M_sink = 1.0 if J0 > Jc else 0.0
    if M_sink == 0.0: return 0.0

    md = 3*max_k + 10
    Jarr = np.zeros(md+1, dtype=np.float64)
    Jarr[1:] = J0 * np.power(np.arange(1, md+1, dtype=np.float64), -a)

    rs = np.arange(1, max_k+1, dtype=np.int64)
    cur = Jarr.copy()
    prodR = 1.0

    for step in range(max_steps):
        # ---- renormalise couplings ----
        Jp = np.empty(max_k, dtype=np.float64)
        for ri in range(max_k):
            Jp[ri] = _Jprime(3*rs[ri]+1, cur)

        # ---- dH'/dH at this level ----
        dh = dHdH(cur, eps)
        if np.isnan(dh) or dh <= 0.0: break
        prodR *= dh

        # ---- next lattice ----
        cur = np.zeros(md+1, dtype=np.float64)
        cur[1:max_k+1] = Jp

    scale = (b**(-d))**(step+1)
    M0 = scale * prodR * M_sink
    return M0
    #return np.clip(M0, 0.0, 1.0)

@njit
def _linearise_Jprime(Jarr_crit, max_k=800, eps=1e-8):
    """
    Build the Jacobian ∂J'_i / ∂J_j  at the critical point.
    Returns a (max_k × max_k) matrix.
    """
    md = 3*max_k + 10
    rs = np.arange(1, max_k+1, dtype=np.int64)
    n = max_k
    Jac = np.zeros((n, n), dtype=np.float64)

    # baseline J' (central difference needs two evaluations)
    Jp_base = np.empty(n, dtype=np.float64)
    for ri in range(n):
        Jp_base[ri] = _Jprime(3*rs[ri]+1, Jarr_crit)

    # perturb each input coupling J_k separately
    for k in range(n):
        Jarr_plus  = Jarr_crit.copy()
        Jarr_minus = Jarr_crit.copy()
        dJ = eps * max(1.0, abs(Jarr_crit[rs[k]]))
        if dJ == 0.0:
            dJ = eps
        Jarr_plus[rs[k]]  += dJ
        Jarr_minus[rs[k]] -= dJ

        Jp_plus  = np.empty(n, dtype=np.float64)
        Jp_minus = np.empty(n, dtype=np.float64)
        for ri in range(n):
            Jp_plus[ri]  = _Jprime(3*rs[ri]+1, Jarr_plus)
            Jp_minus[ri] = _Jprime(3*rs[ri]+1, Jarr_minus)

        Jac[:, k] = (Jp_plus - Jp_minus) / (2.0 * dJ)

    return Jac

def compute_exponents(a, max_k=800, eps_J=1e-8, eps_matrix=1e-8):
    """
    Full critical-exponent calculation for a given power-law exponent a.
    Returns a dictionary with y_T, y_H and all standard exponents.
    """
    # 1. find the critical coupling
    Jc = find_Jc(a, max_k=max_k, tol=1e-10)

    # 2. build the critical interaction array (large enough)
    md = 3*max_k + 10
    Jarr_crit = np.zeros(md+1, dtype=np.float64)
    Jarr_crit[1:] = Jc * np.power(np.arange(1, md+1, dtype=np.float64), -a)

    # 3. thermal eigenvalue y_T  (largest eigenvalue of the coupling Jacobian)
    Jac = _linearise_Jprime(Jarr_crit, max_k=max_k, eps=eps_matrix)
    eigenvals = np.linalg.eigvals(Jac)
    # the relevant thermal operator is the largest real eigenvalue > 1
    real_parts = np.real(eigenvals)
    y_T = np.max(real_parts[real_parts > 0])

    # 4. magnetic eigenvalue y_H  (logarithmic derivative of H' w.r.t. H at H=0)
    #    we already have dHdH, just evaluate at the critical point
    y_H = np.log(dHdH(Jarr_crit, eps=eps_J)) / np.log(3.0)   # b=3

    # 5. dimension
    d = 1.0

    # 6. standard exponents (exactly the formulae from the paper)
    beta = (d - y_H) / y_T
    delta = y_H / (d - y_H)
    eta = 2.0 + d - 2.0*y_H
    nu = 1.0 / y_T
    alpha = 2.0 - d / y_T
    gamma = (2.0*y_H - d) / y_T

    exponents = {
        'a'      : a,
        'Jc'     : Jc,
        'y_T'    : y_T,
        'y_H'    : y_H,
        'alpha'  : alpha,
        'beta'   : beta,
        'gamma'  : gamma,
        'delta'  : delta,
        'nu'     : nu,
        'eta'    : eta,
    }
    return exponents