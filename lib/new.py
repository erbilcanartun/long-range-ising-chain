import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# =========================
# Utilities
# =========================

def required_initial_max_distance(max_dist_final, n_steps):
    D = max_dist_final
    for _ in range(n_steps):
        D = 3 * D + 2
    return D

def logsumexp(values):
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m)))

# =========================
# Majority configurations (3-spin cell)
# =========================

_all_spins = np.array(list(product([-1, 1], repeat=3)), dtype=int)
plus_configs  = _all_spins[np.sum(_all_spins, axis=1) >=  1]
minus_configs = _all_spins[np.sum(_all_spins, axis=1) <= -1]

# =========================
# Intracell energies
# =========================

def intracell_energies(spins, J):
    J1 = J[1] if len(J) > 1 else 0.0
    J2 = J[2] if len(J) > 2 else 0.0

    E = np.empty(spins.shape[0], dtype=float)
    for i, (s0, s1, s2) in enumerate(spins):
        E[i] = J1 * (s0 * s1 + s1 * s2) + J2 * (s0 * s2)
    return E

# =========================
# RG step: J -> J'
# =========================

def renormalized_coupling_for_r(r, J):
    D = len(J) - 1  # max distance available

    # Precompute intracell energies
    E_plus  = intracell_energies(plus_configs,  J)
    E_minus = intracell_energies(minus_configs, J)

    # Positions of spins in the left and right cells
    left_pos  = np.array([0, 1, 2])
    right_pos = np.array([3 * r + 0, 3 * r + 1, 3 * r + 2])

    # Precompute distances between them
    distances = np.abs(right_pos[None, :] - left_pos[:, None])  # shape (3,3)

    # R(++): left-majority+, right-majority+
    totals_pp = []
    for iL, sL in enumerate(plus_configs):
        EL = E_plus[iL]
        for iR, sR in enumerate(plus_configs):
            ER = E_plus[iR]
            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d <= D:
                        E_int += J[d] * sL[a] * sR[b]
            totals_pp.append(EL + ER + E_int)

    # R(+-): left-majority+, right-majority-
    totals_pm = []
    for iL, sL in enumerate(plus_configs):
        EL = E_plus[iL]
        for iR, sR in enumerate(minus_configs):
            ER = E_minus[iR]
            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d <= D:
                        E_int += J[d] * sL[a] * sR[b]
            totals_pm.append(EL + ER + E_int)

    log_R_pp = logsumexp(totals_pp)
    log_R_pm = logsumexp(totals_pm)

    # Spin-flip symmetry: J'_r = 1/2 [ln R(++ ) - ln R(+-)]
    return 0.5 * (log_R_pp - log_R_pm)

def rg_step(J):
    D = len(J) - 1
    r_max = (D - 2) // 3
    if r_max < 1:
        raise ValueError("Not enough range in J to perform another RG step.")

    J_new = np.zeros(r_max + 1, dtype=float)  # index 0 unused
    for r in range(1, r_max + 1):
        J_new[r] = renormalized_coupling_for_r(r, J)

    return J_new

# =========================
# RG flow driver (full, proper)
# =========================

def generate_rg_flow(J0, a, max_dist_final, n_steps, trace_TM=False, TM_r=1):
    D0 = required_initial_max_distance(max_dist_final, n_steps)

    # initial couplings
    J = np.zeros(D0 + 1)
    d = np.arange(1, D0 + 1)
    J[1:] = J0 / (d ** a)

    J_list = []
    TM_list = [] if trace_TM else None

    for step in range(n_steps + 1):
        J_list.append(J)

        if trace_TM and TM_r < len(J):
            T = construct_transfer_matrix(J, TM_r, normalize=True)
            TM_list.append(T)

        if step == n_steps:
            break

        J = rg_step(J)

    return J_list, TM_list


def extract_flows(J_list, max_dist_final):
    n_steps_plus_1 = len(J_list)
    flows = np.zeros((n_steps_plus_1, max_dist_final + 1), dtype=float)
    for step, J in enumerate(J_list):
        D_curr = len(J) - 1
        r_max_record = min(max_dist_final, D_curr)
        flows[step, :r_max_record + 1] = J[:r_max_record + 1]
    return flows

def plot_rg_flow(flows, distances_to_plot=None):
    n_steps_plus_1, max_dist_plus_1 = flows.shape
    max_dist = max_dist_plus_1 - 1
    steps = np.arange(n_steps_plus_1)

    if distances_to_plot is None:
        distances_to_plot = range(1, max_dist + 1)

    for r in distances_to_plot:
        if r <= max_dist:
            plt.plot(steps, flows[:, r], marker='o', label=f"J_{r}")

    plt.xlabel("RG step")
    plt.ylabel("Coupling J_r")
    plt.title("RG flow of couplings")
    plt.legend()
    plt.tight_layout()
    plt.show()

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

def find_phase(J0, a, TM_r=1, max_dist_final=5, n_steps=5, threshold=0.1):
    # initial long-range cutoff
    D0 = required_initial_max_distance(max_dist_final, n_steps)

    # initial interaction function
    J = np.zeros(D0 + 1, dtype=float)
    d = np.arange(1, D0 + 1)
    J[1:] = J0 / (d ** a)

    for step in range(n_steps + 1):
        # construct transfer matrix if possible
        if TM_r < len(J):
            T = construct_transfer_matrix(J, TM_r, normalize=True)
            phase = determine_phase_from_TM(T, threshold=threshold)

            if phase != "undetermined":
                return phase, step

        if step == n_steps:
            break

        J = rg_step(J)
    return "undetermined", n_steps

def find_Jc(a, tol=1e-6, Jlow=0.1, Jhigh=12.0, max_steps=12, growth_threshold=1e5, decay_threshold=1e-5):
    if not (0 < a < 2):
        raise ValueError("a must be in (0,2)")

    def grows(J0):
        # Build initial full-length J vector of some safe size.
        # Choose a length that survives max_steps RG iterations.
        D0 = required_initial_max_distance(1, max_steps)
        J = np.zeros(D0 + 1, dtype=float)
        r = np.arange(1, D0 + 1, dtype=float)
        J[1:] = J0 / (r ** a)

        J1_initial = J[1]

        for _ in range(max_steps):
            # If J[1] already exhibits extreme growth/decay, decide early.
            if abs(J[1]) > growth_threshold:
                return True
            if abs(J[1]) < decay_threshold:
                return False

            # Do one full RG step
            J = rg_step(J)

            # If range collapsed, treat as flowing to disorder
            if len(J) <= 2:
                return False

        # If after max_steps: compare trend to initial
        return abs(J[1]) > abs(J1_initial)

    # Standard bisection
    while Jhigh - Jlow > tol:
        Jmid = 0.5 * (Jlow + Jhigh)
        if grows(Jmid):
            Jhigh = Jmid
        else:
            Jlow = Jmid

    return 0.5 * (Jlow + Jhigh)


def find_critical_J(a, J_low=1e-2, J_high=1e2, TM_r=1, max_dist_final=5,
                    n_steps=5, threshold=0.1, tol=1e-3, max_iter=20):

    # --- Ensure bracketing validity ---
    phase_low, _  = find_phase(J_low,  a,
                               TM_r=TM_r,
                               max_dist_final=max_dist_final,
                               n_steps=n_steps,
                               threshold=threshold)
    if phase_low != "disorder":
        raise ValueError(f"J_low={J_low} does not yield disorder (got {phase_low})")
        
    phase_high, _ = find_phase(J_high, a,
                               TM_r=TM_r,
                               max_dist_final=max_dist_final,
                               n_steps=n_steps,
                               threshold=threshold)
    if phase_high != "ferromagnetic":
        raise ValueError(f"J_high={J_high} does not yield ferromagnetic (got {phase_high})")

    # --- Bisection loop ---
    for _ in range(max_iter):

        J_mid = 0.5 * (J_low + J_high)

        phase_mid, _ = find_phase(J_mid, a,
                                  TM_r=TM_r,
                                  max_dist_final=max_dist_final,
                                  n_steps=n_steps,
                                  threshold=threshold)

        # Disorder side → shift lower bound up
        if phase_mid == "disorder":
            J_low = J_mid
        # Ferro side → shift upper bound down
        elif phase_mid == "ferromagnetic":
            J_high = J_mid
        else:
            # Undetermined → leave interval but continue bisecting
            # (this matches behavior in the attached RG code)
            pass

        # Check convergence
        if abs(J_high - J_low) < tol:
            break

    return 0.5 * (J_low + J_high)

def compute_critical_line(a_values, J_low_init=1e-2, J_high_init=1e2,
                          TM_r=1, max_dist_final=5, n_steps=5,
                          threshold=0.1, tol=1e-3, max_iter=20):
    Jc_list = []

    J_low  = J_low_init
    J_high = J_high_init

    for a in a_values:

        # ---------------------------
        # Find J_c for this a
        # ---------------------------
        Jc = find_critical_J(
            a,
            J_low, J_high,
            TM_r=TM_r,
            max_dist_final=max_dist_final,
            n_steps=n_steps,
            threshold=threshold,
            tol=tol,
            max_iter=max_iter
        )

        Jc_list.append(Jc)

        # Update bracketing interval around the new Jc
        #J_low  = 0.5 * Jc
        #J_high = 2.0 * Jc

    return np.array(Jc_list, dtype=float)

def plot_critical_temperatures(a_values, Jc_values):
    Tc = 1.0 / Jc_values

    plt.plot(a_values, Tc, marker="o", linestyle="-")
    plt.xlabel("Interaction Range Exponent   a")
    plt.ylabel("Critical Temperature   1 / J_c")
    plt.title("Critical Line:  Temperature vs Interaction Exponent")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_and_plot_critical_line(a_min=1.0, a_max=2.0, n_points=20, J_low_init=0.01, J_high_init=10.0,
                                   TM_r=1, max_dist_final=8, n_steps=25, threshold=0.1):
    a_values = np.linspace(a_min, a_max, n_points)
    
    Jc_values = compute_critical_line(a_values, J_low_init=J_low_init, J_high_init=J_high_init,
        TM_r=TM_r, max_dist_final=max_dist_final, n_steps=n_steps, threshold=threshold)

    plot_critical_temperatures(a_values, Jc_values)
    return a_values, Jc_values

def full_rg_map(J_full):
    return rg_step(J_full)

def compute_full_recursion_matrix(J_full, eps=1e-6):
    """
    Compute the full recursion (Jacobian) matrix for the *actual* RG map.

      J_full (size D)  ->  Jp_full (size D1)
      T_{ij} ≈ d J'_i / d J_j

    Shape:
      D  = len(J_full) - 1
      D1 = len(Jp_full) - 1
      T: (D1 x D)
    """
    D  = len(J_full) - 1
    Jp = full_rg_map(J_full)
    D1 = len(Jp) - 1

    T = np.zeros((D1, D), dtype=float)

    for j in range(1, D+1):
        J_pert = J_full.copy()
        J_pert[j] += eps
        Jp_pert = full_rg_map(J_pert)

        # assumes same D1 (RG structure fixed by input length)
        diff = (Jp_pert[1:D1+1] - Jp[1:D1+1]) / eps
        T[:, j-1] = diff
    return T, Jp

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


def build_initial_guess(Jc, a, D):
    J = np.zeros(D+1)
    for r in range(1, D+1):
        J[r] = Jc / (r**a)
    return J
    
def newton_rg_lstsq(J_init,
                    max_iter=10,
                    tol=1e-8,
                    eps=1e-6,
                    damping=1.0,
                    verbose=False):
    J = J_init.copy()
    D = len(J) - 1

    for it in range(max_iter):
        T, Jp = compute_full_recursion_matrix(J, eps=eps)
        D1 = len(Jp) - 1

        r = Jp[1:D1+1] - J[1:D1+1]
        res_norm = np.linalg.norm(r)

        if verbose:
            print(f"[lstsq] iter {it}: D={D}, D1={D1}, ||J'-J||={res_norm:.3e}")

        if res_norm < tol:
            return J, {"converged": True, "iterations": it}

        # Solve rectangular system  T δ ≈ r
        delta, *_ = np.linalg.lstsq(T, r, rcond=None)

        J_new = J.copy()
        J_new[1:D+1] = J[1:D+1] - damping * delta

        J = J_new

    return J, {"converged": False, "iterations": max_iter}

def rg_map_fixed_head(J_head):
    N = len(J_head) - 1
    # Choose a physical max distance so that after one RG step we still
    # have at least N couplings in the output. For safety, start from
    # a decent multiple of N; you can adjust this.
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

def check_fixed_point(J_star, tol=1e-6):
    # physical RG step
    Jp = rg_step(J_star)

    D  = len(J_star) - 1
    D1 = len(Jp) - 1

    # compare only the overlapping region
    diff = Jp[1:D1+1] - J_star[1:D1+1]
    err  = np.linalg.norm(diff)

    print(f"Fixed-point error norm = {err:.3e}")

    # detailed pass/fail
    if err < tol:
        print("✔ Fixed point verified.")
    else:
        print("✘ Not a fixed point (or tolerance too strict).")
    return err


################

def renormalized_field(J, H):
    """
    Compute renormalized magnetic field H' for a pair of neighbouring 3-spin blocks
    at couplings J and microscopic field H.

    This mirrors the logic of long_range_ising_rg._Hprime but uses the full-length
    coupling vector J and the geometry used in renormalized_coupling_for_r(r=1).
    """
    # Intracell energies for plus / minus majority blocks
    E_plus  = intracell_energies(plus_configs,  J)
    E_minus = intracell_energies(minus_configs, J)

    # Geometry of two neighbouring blocks in the 1D chain:
    # left block spins at positions 0,1,2; right block at 3,4,5
    left_pos  = np.array([0, 1, 2], dtype=int)
    right_pos = np.array([3, 4, 5], dtype=int)

    # Precompute pairwise distances between spins in left and right blocks
    distances = np.abs(right_pos[None, :] - left_pos[:, None])  # shape (3,3)

    totals_pp = []
    totals_mm = []

    # R(++): left-majority+, right-majority+
    for iL, sL in enumerate(plus_configs):
        EL = E_plus[iL]
        magL = np.sum(sL)
        for iR, sR in enumerate(plus_configs):
            ER = E_plus[iR]
            magR = np.sum(sR)

            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d < len(J):
                        E_int += J[d] * sL[a] * sR[b]

            totals_pp.append(EL + ER + E_int + H * (magL + magR))

    # R(--): left-majority-, right-majority-
    for iL, sL in enumerate(minus_configs):
        EL = E_minus[iL]
        magL = np.sum(sL)
        for iR, sR in enumerate(minus_configs):
            ER = E_minus[iR]
            magR = np.sum(sR)

            E_int = 0.0
            for a in range(3):
                for b in range(3):
                    d = distances[a, b]
                    if d < len(J):
                        E_int += J[d] * sL[a] * sR[b]

            totals_mm.append(EL + ER + E_int + H * (magL + magR))

    if not totals_pp or not totals_mm:
        return np.nan

    log_pp = logsumexp(totals_pp)
    log_mm = logsumexp(totals_mm)

    if not np.isfinite(log_pp) or not np.isfinite(log_mm):
        return np.nan

    # Standard relation: H' = 1/4 [ln Z_{++} - ln Z_{--}]
    return 0.25 * (log_pp - log_mm)


def dH_dH(J, eps=1e-8):
    """
    Finite-difference estimate of dH'/dH at H=0 for given couplings J.
    Uses renormalized_field(J, H).
    """
    H0    = renormalized_field(J, 0.0)
    H_eps = renormalized_field(J, eps)

    if not np.isfinite(H0) or not np.isfinite(H_eps):
        return np.nan

    return (H_eps - H0) / eps


def magnetic_exponent_yH(J_star, eps=1e-8, b=3.0):
    """
    Compute magnetic exponent y_H at the fixed point J_star.

    Parameters
    ----------
    J_star : 1D numpy array
        Fixed-point couplings (index 0 unused).
    eps : float
        Small field for finite-difference derivative.
    b : float
        RG length-rescaling factor (b=3 for 3-spin block).

    Returns
    -------
    y_H : float
        Magnetic scaling exponent.
    alpha : float
        Magnetic eigenvalue dH'/dH at the fixed point.
    """
    alpha = dH_dH(J_star, eps=eps)

    if not np.isfinite(alpha) or alpha <= 0:
        raise RuntimeError(f"dH'/dH invalid or non-positive at fixed point: {alpha}")

    y_H = np.log(alpha) / np.log(b)
    return y_H, alpha

def magnetization(J0, a, Jc, max_dist_final=3, max_steps=10, eps=1e-8):
    """
    Back-propagate the magnetization from the RG sink to the microscopic scale,
    using the full-length coupling vectors implemented in this file.

    This mirrors the logic of long_range_ising_rg.magnetisation, but instead of
    truncating the couplings to max_k we let rg_step(J) decide the range at
    each iteration.

    Parameters
    ----------
    J0 : float
        Bare nearest-neighbour coupling (= 1 / T).
    a : float
        Power-law exponent in J_r = J0 / r^a.
    Jc : float
        Critical coupling for this a (e.g. from find_critical_coupling).
    max_dist_final : int, optional
        Desired maximum interaction range at the *final* RG step. This is only
        used to decide how long the *initial* coupling vector must be in order
        to support `max_steps` RG iterations.  By default we keep at least the
        nearest-neighbour coupling (max_dist_final = 1).
    max_steps : int, optional
        Maximum number of RG iterations to perform.
    eps : float, optional
        Small field increment used in the finite-difference estimate of dH'/dH.

    Returns
    -------
    M0 : float
        Magnetization at the microscopic scale, up to an overall non-universal
        prefactor. As in the numba implementation, this is not clipped to the
        interval [0, 1], so amplitudes larger than 1 are possible and simply
        reflect this prefactor.
    """
    # If we start on the disordered side, magnetization is zero.
    M_sink = 1.0 if J0 > Jc else 0.0
    if M_sink == 0.0:
        return 0.0

    # Length-rescaling factor and spatial dimension
    b = 3.0
    d_dim = 1.0

    # Build initial full-length coupling vector that can support `max_steps`
    # RG iterations and still retain couplings up to `max_dist_final`.
    D0 = required_initial_max_distance(max_dist_final, max_steps)
    J = np.zeros(D0 + 1, dtype=float)
    r = np.arange(1, D0 + 1, dtype=float)
    J[1:] = J0 / (r ** a)

    prod_R = 1.0
    steps_done = 0

    for step in range(max_steps):
        # dH'/dH at this level
        dH = dH_dH(J, eps=eps)
        if not np.isfinite(dH) or dH <= 0.0:
            break
        prod_R *= dH
        steps_done = step + 1

        # Next RG step. If the range collapses (r_max < 1) we stop.
        D = len(J) - 1
        r_max = (D - 2) // 3
        if r_max < 1:
            break
        J = rg_step(J)

    if steps_done == 0:
        return 0.0

    scale = (b ** (-d_dim)) ** steps_done
    M0 = scale * prod_R * M_sink
    return float(M0)

import numpy as np
import matplotlib.pyplot as plt


# ===============================================================
#  USER-DEFINED: these MUST already exist in your new.py module.
#  I assume these signatures match your actual implementation.
# ===============================================================

# find critical coupling using TM-free / full-length method
#   Jc = find_Jc_simple(a)
#
# build initial long-range coupling vector
#   J0 = build_initial_guess(Jc, a, D)
#
# find fixed point using Newton RG
#   J_star, info = newton_rg_lstsq(J0)
#
# thermal exponent
#   T, J_head = recursion_matrix_at_fixed_point(J_star, N)
#   y_T, lambda_T, eigs = thermal_exponent_from_T(T, b=3.0)
#
# magnetic exponent
#   y_H, alphaH = magnetic_exponent_yH(J_star, eps, b)



# ===============================================================
# Compute all exponents for a range of "a"
# ===============================================================

def compute_exponents_over_a(
        a_min=1.0, a_max=2.0, num_points=20,
        D_init=200,
        N_matrix=10,
        d_dim=1.0,   # geometric dim (or effective dim later)
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

        # -------------------------------------------------------
        # 1. Find critical Jc
        # -------------------------------------------------------
        Jc = find_critical_J(a, J_low=1e-3, J_high=1e2, TM_r=1, max_dist_final=5,
                             n_steps=6, threshold=0.1, tol=1e-3, max_iter=20)
        print("  Jc =", Jc)

        # -------------------------------------------------------
        # 2. Build initial guess (full-length)
        # -------------------------------------------------------
        J0 = build_initial_guess(Jc, a=a, D=D_init)

        # -------------------------------------------------------
        # 3. Newton solver to find RG fixed point
        # -------------------------------------------------------
        J_star, info = newton_rg_lstsq(J0)
        # optional: check_fixed_point(J_star)

        # -------------------------------------------------------
        # 4. Compute yT (thermal exponent)
        # -------------------------------------------------------
        Tmat, Jhead = recursion_matrix_at_fixed_point(J_star, N_matrix)
        yT, lambda_T, eigs = thermal_exponent_from_T(Tmat, b=b)
        print("  yT =", yT)
        yT_arr[i] = yT

        # -------------------------------------------------------
        # 5. Compute yH (magnetic exponent)
        # -------------------------------------------------------
        yH, alphaH = magnetic_exponent_yH(J_star, eps=1e-8, b=b)
        print("  yH =", yH)
        yH_arr[i] = yH

        # -------------------------------------------------------
        # 6. Derived critical exponents
        # -------------------------------------------------------

        # ν
        nu = 1.0 / yT
        nu_arr[i] = nu

        # β
        beta = (d_dim - yH) / yT
        beta_arr[i] = beta

        # δ
        delta = yH / (d_dim - yH)
        delta_arr[i] = delta

        # η
        eta = 2 + d_dim - 2 * yH
        eta_arr[i] = eta

        # α
        alpha = 2 - d_dim / yT
        alpha_arr[i] = alpha

        # γ
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


# ===============================================================
# Plot 4×4 grid of exponents vs a
# ===============================================================

def plot_exponents(result_dict):

    a = result_dict["a"]

    exps = [
        ("yT",     result_dict["yT"]),
        ("yH",     result_dict["yH"]),
        ("β",      result_dict["beta"]),
        ("δ",      result_dict["delta"]),
        ("η",      result_dict["eta"]),
        ("ν",      result_dict["nu"]),
        ("α",      result_dict["alpha"]),
        ("γ",      result_dict["gamma"]),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    axes = axes.flatten()

    for i, (name, vals) in enumerate(exps):
        ax = axes[i]
        ax.plot(a, vals, marker="o")
        ax.set_title(name)
        ax.set_xlabel("a")
        ax.grid(True)

    # Hide the unused 8 panels in 4×4 grid
    for j in range(len(exps), 16):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
