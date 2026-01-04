import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from utils import required_initial_max_distance, logsumexp, build_J
from decimation_contiguous import *
#from decimation_staggered import *


@njit(cache=True)
def rg_step(J):
    D = len(J) - 1
    r_max = (D - 2) // 3

    J_new = np.empty_like(J)
    J_new[0] = 0.0

    for r in range(1, r_max + 1):
        log_R_pp, log_R_pm = log_Rpp_Rpm(r, J)
        J_new[r] = 0.5 * (log_R_pp - log_R_pm)

    return J_new


@njit(cache=True)
def renormalized_field(J, H):
    log_pp, log_mm = log_Rpp_Rpm_nonzero_H(J, H)
    # H' = 1/4 [ln Z{++} - ln Z{--}]
    return 0.25 * (log_pp - log_mm)


@njit(cache=True)
def dH_dH(J, eps=1e-6):
    H0    = renormalized_field(J, 0.0)
    H_eps = renormalized_field(J, eps)

    if not np.isfinite(H0) or not np.isfinite(H_eps):
        return np.nan

    return (H_eps - H0) / eps


def find_Jc(a, Jlow=0.1, Jhigh=12.0, max_steps=6, max_dist_final=5,
            tol=1e-8, growth_threshold=1e6, decay_threshold=1e-6):
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


def recursion_matrix_at_fixed_point(J_star, N, eps=1e-6):
    J_head = np.zeros(N + 1)
    J_head[1:N+1] = J_star[1:N+1]

    # Base image
    Jp_base = rg_map_fixed_head(J_head)
    T = np.zeros((N, N))

    for j in range(1, N+1):
        J_pert = J_head.copy()
        J_pert[j] += eps
        Jp_pert = rg_map_fixed_head(J_pert)

        diff = (Jp_pert[1:N+1] - Jp_base[1:N+1]) / eps
        T[:, j-1] = diff

    return T, J_head


def rg_map_fixed_head(J_head):
    N = len(J_head) - 1
    D_full = 3 * N + 20

    # Build J_full with some tail continuation. Simple option: zero tail.
    J_full = np.zeros(D_full + 1)
    J_full[1:N+1] = J_head[1:N+1]

    # RG step on full vector
    Jp_full = rg_step(J_full)

    D1 = len(Jp_full) - 1
    if D1 < N:
        raise RuntimeError(f"RG output has only {D1} couplings, less than N={N}")

    # Project back to head
    Jp_head = np.zeros(N + 1)
    Jp_head[1:N+1] = Jp_full[1:N+1]
    return Jp_head


def thermal_exponent_from_T(T, b=3.0):
    eigvals = np.linalg.eigvals(T)
    # sort by magnitude descending
    eigvals_sorted = sorted(eigvals, key=lambda z: abs(z), reverse=True)

    # pick the first eigenvalue with |λ| > 1 as thermal candidate
    lambda_T = None
    for lam in eigvals_sorted:
        if abs(lam) > 1.0:
            lambda_T = lam
            break

    if lambda_T is None:
        raise RuntimeError("No relevant (|λ|>1) eigenvalue found; cannot define y_T.")

    y_T = np.log(abs(lambda_T)) / np.log(b)
    return y_T, lambda_T, eigvals_sorted





def magnetic_exponent_yH(J_star, eps=1e-8, b=3.0):
    alpha = dH_dH(J_star, eps=eps)

    if not np.isfinite(alpha) or alpha <= 0:
        raise RuntimeError(f"dH'/dH invalid or non-positive at fixed point: {alpha}")

    y_H = np.log(alpha) / np.log(b)
    return y_H, alpha


def compute_exponents_over_a(
        a_min=1.0, a_max=2.0, num_points=20,
        D_init=200,
        N_matrix=10,
        d_dim=1.0,
        b=3.0):

    a_values = np.linspace(a_min, a_max, num_points)

    # Storage arrays
    yT_arr = np.zeros(num_points)
    yH_arr = np.zeros(num_points)

    beta_arr = np.zeros(num_points)
    delta_arr = np.zeros(num_points)
    eta_arr = np.zeros(num_points)
    nu_arr = np.zeros(num_points)
    alpha_arr = np.zeros(num_points)
    gamma_arr = np.zeros(num_points)

    for i, a in enumerate(a_values):
        print(f"\n=== Computing exponents at a = {a:.4f} ===")

        Jc = find_Jc(a=a, Jlow=1e-2, Jhigh=1e2,
                     max_steps=6, max_dist_final=6,
                     tol=1e-5, growth_threshold=1e4, decay_threshold=1e-4)
        J0 = build_J(Jc, a=a, D=D_init)
        J_star, info = newton_rg_lstsq(J0, max_iter=10,
                                       tol=1e-5, eps=1e-6,
                                       damping=1.0, verbose=False)

        # Compute yT (thermal exponent)
        Tmat, Jhead = recursion_matrix_at_fixed_point(J_star, N_matrix)
        yT, lambda_T, eigs = thermal_exponent_from_T(Tmat, b=b)
        print("  yT =", yT)
        yT_arr[i] = yT

        # Compute yH (magnetic exponent)
        yH, alphaH = magnetic_exponent_yH(J_star, eps=1e-8, b=b)
        print("  yH =", yH)
        yH_arr[i] = yH

        # Derived critical exponents
        nu = 1.0 / yT
        nu_arr[i] = nu

        beta = (d_dim - yH) / yT
        beta_arr[i] = beta

        delta = yH / (d_dim - yH)
        delta_arr[i] = delta

        eta = 2 + d_dim - 2 * yH
        eta_arr[i] = eta

        alpha = 2 - d_dim / yT
        alpha_arr[i] = alpha

        gamma = (2 * yH - d_dim) / yT
        gamma_arr[i] = gamma

        print(f"  ν={nu:.4f}, β={beta:.4f}, δ={delta:.4f}, η={eta:.4f}, α={alpha:.4f}, γ={gamma:.4f}")

    return {
        "a": a_values,
        "yT": yT_arr,
        "yH": yH_arr,
        "beta": beta_arr,
        "delta": delta_arr,
        "eta": eta_arr,
        "nu": nu_arr,
        "alpha": alpha_arr,
        "gamma": gamma_arr,
    }
