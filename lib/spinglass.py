# Rewritten renormalization_sg.py using NumPy and Numba for speed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from numba import njit

# Precompute all possible spin configurations once (global)
all_spins_lists = list(product([-1, 1], repeat=3))

# Group them into plus, minus (no zero for Ising)
plus_configs_spins = []
minus_configs_spins = []
for spins_list in all_spins_lists:
    sig = sum(spins_list)
    if sig >= 1:
        plus_configs_spins.append(spins_list)
    elif sig <= -1:
        minus_configs_spins.append(spins_list)

plus_spins = np.array(plus_configs_spins, dtype=np.float64)
minus_spins = np.array(minus_configs_spins, dtype=np.float64)

# Utility to get long-range parameter (float)
def get_param(d, param0, a):
    if d <= 0:
        return 0.0
    return param0 / d**a

@njit
def compute_El(spins, J2, J4):
    n = spins.shape[0]
    Els = np.zeros(n)
    for i in range(n):
        s0 = spins[i, 0]
        s1 = spins[i, 1]
        s2 = spins[i, 2]
        El = J2 * s0 * s1 + J2 * s1 * s2 + J4 * s0 * s2
        Els[i] = El
    return Els

@njit
def compute_log_R(left_spins, left_Els, right_spins, right_Els, distances, J_arr):
    nL = left_spins.shape[0]
    nR = right_spins.shape[0]
    nd = distances.shape[0]
    totals = np.zeros(nL * nR)
    idx = 0
    for i in range(nL):
        El = left_Els[i]
        for j in range(nR):
            Er = right_Els[j]
            Eint = 0.0
            for k in range(nd):
                iL = distances[k, 0]
                iR = distances[k, 1]
                d = distances[k, 2]
                sLi = left_spins[i, iL]
                sRi = right_spins[j, iR]
                Eint += J_arr[d] * sLi * sRi
            totals[idx] = El + Er + Eint
            idx += 1
    if totals.size == 0:
        return -np.inf
    max_t = np.max(totals)
    sum_exp = np.sum(np.exp(totals - max_t))
    if sum_exp <= 0:
        return -np.inf
    return np.log(sum_exp) + max_t

def compute_J_prime_func(start, J_arr):
    J2 = J_arr[2]
    J4 = J_arr[4]
    left_pos = np.array([1, 3, 5])
    right_pos = np.array([start, start + 2, start + 4])
    distances_list = []
    for iL in range(3):
        for iR in range(3):
            d = abs(right_pos[iR] - left_pos[iL])
            distances_list.append((iL, iR, d))
    distances = np.array(distances_list, dtype=np.int64)

    El_plus = compute_El(plus_spins, J2, J4)
    El_minus = compute_El(minus_spins, J2, J4)

    log_R_pp = compute_log_R(plus_spins, El_plus, plus_spins, El_plus, distances, J_arr)
    log_R_pm = compute_log_R(plus_spins, El_plus, minus_spins, El_minus, distances, J_arr)

    if np.isinf(log_R_pm) or np.isinf(log_R_pp):
        return np.inf
    return 0.5 * (log_R_pp - log_R_pm)

def generate_rg_flow(J0, n, p, max_k, num_steps, show_first=0, seed=42):
    np.random.seed(seed)
    max_d = 3 * max_k + 10
    rs = list(range(1, max_k + 1))
    d_arr = np.arange(1, max_d + 1, dtype=np.float64)
    signs = np.random.choice([-1, 1], size=max_d, p=[p, 1 - p])
    J_arr = np.zeros(max_d + 1)
    J_arr[1:] = signs * (J0 * np.power(d_arr, -n))

    initial_Js = J_arr[1:max_k + 1].copy()
    all_Js = [initial_Js]

    if show_first > 0:
        print("Step 0 (initial):")
        for i in range(min(show_first, len(initial_Js))):
            print(f"  J({i+1}) = {initial_Js[i]}")

    for step in range(1, num_steps + 1):
        Jps = np.zeros(max_k)
        for r_idx, r in enumerate(rs):
            start = 3 * r + 1
            Jp = compute_J_prime_func(start, J_arr)
            Jps[r_idx] = Jp
        all_Js.append(Jps)

        if show_first > 0:
            print(f"Step {step}:")
            for i in range(min(show_first, len(Jps))):
                print(f"  J({i+1}) = {Jps[i]}")

        J_arr = np.zeros(max_d + 1)
        J_arr[1:max_k + 1] = Jps

    return all_Js

def phase_sink(all_Js, skip_steps=3, track_rs=[2, 3]):
    if len(all_Js) < skip_steps + 2:
        return 'undecided'

    tracked_flow = all_Js[skip_steps:skip_steps + 2]

    if len(tracked_flow) < 2:
        return 'undecided'
    prev = tracked_flow[-2]
    last = tracked_flow[-1]

    all_positive = all(last[r - 1] > 0 for r in track_rs)

    S_prev = sum(abs(prev[r - 1]) for r in track_rs)
    S_last = sum(abs(last[r - 1]) for r in track_rs)

    if all_positive and S_last > S_prev:
        return 'ferro'
    elif S_last < S_prev:
        return 'disorder'
    else:
        return 'undecided'

def find_Tc_fixed_p(n, p, max_k=1000, tol=1e-6, J_low=0.01, J_high=10.0, skip_steps=3, track_rs=[2, 3], seed=42):
    if n <= 0 or n >= 2:
        raise ValueError("n must be between 0 and 2, excluding the edges.")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1.")

    def compute_phase(J0):
        all_Js = generate_rg_flow(J0, n, p, max_k, num_steps=skip_steps + 2, seed=seed)
        phase = phase_sink(all_Js, skip_steps, track_rs)
        if phase == 'undecided':
            print(f"Warning: Phase undecided for n={n}, p={p}, J={J0}")
        return phase == 'disorder'

    iter_count = 0
    while J_high - J_low > tol and iter_count < 100:
        iter_count += 1
        J_mid = (J_low + J_high) / 2
        phase_mid = compute_phase(J_mid)
        if phase_mid:
            J_low = J_mid
        else:
            J_high = J_mid
    Jc = (J_low + J_high) / 2
    return 1 / Jc

def find_pc_fixed_T(n, T, max_k=1000, tol=1e-6, p_low=0.0, p_high=1.0, skip_steps=3, track_rs=[2, 3], seed=42):
    if n <= 0 or n >= 2:
        raise ValueError("n must be between 0 and 2, excluding the edges.")
    if T <= 0:
        raise ValueError("T must be positive.")

    J0 = 1.0 / T

    def compute_phase(curr_p):
        all_Js = generate_rg_flow(J0, n, curr_p, max_k, num_steps=skip_steps + 2, seed=seed)
        phase = phase_sink(all_Js, skip_steps, track_rs)
        if phase == 'undecided':
            print(f"Warning: Phase undecided for n={n}, p={curr_p}, J={J0}")
        return phase == 'ferro'

    iter_count = 0
    while p_high - p_low > tol and iter_count < 100:
        iter_count += 1
        p_mid = (p_low + p_high) / 2
        phase_mid = compute_phase(p_mid)
        if phase_mid:
            p_low = p_mid
        else:
            p_high = p_mid
    return (p_low + p_high) / 2

def phase_identify(n, p, J, skip_steps, track_rs):
    all_Js = generate_rg_flow(J, n, p, max_k=5000, num_steps=skip_steps + 2)
    phase = phase_sink(all_Js, skip_steps, track_rs)
    return phase

def plot_phase_diagram(n, p_values, one_over_J_values, max_k=5000, skip_steps=3, track_rs=[2, 3]):
    Disorder_Phase, Ferro_Phase, Undecided_Phase = [], [], []
    for p in tqdm(p_values):
        for one_over_j in one_over_J_values:
            J = 1.0 / one_over_j
            phase = phase_identify(n, p, J, skip_steps, track_rs)
            point = [p, one_over_j]
            if phase == "disorder":
                Disorder_Phase.append(point)
            elif phase == "ferro":
                Ferro_Phase.append(point)
            else:
                Undecided_Phase.append(point)
    grid_size = len(p_values) * len(one_over_J_values)
    base_marker_size = 25
    reference_grid_size = 100
    ms = base_marker_size * np.sqrt(reference_grid_size / grid_size)
    ms = max(1, min(ms, 12))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=100)
    fig.set_facecolor("white")
    plt.rc(group="font", family="Arial", weight="bold", size=10)
    plt.rc(group="lines", linewidth=1)
    plt.rc(group="axes", linewidth=2)
    cdic = {"disorder": "grey", "ferro": "blue", "undecided": "orange"}
    if Disorder_Phase: ax.plot(np.array(Disorder_Phase)[:, 0], np.array(Disorder_Phase)[:, 1], ls="", marker="s", mfc=cdic["disorder"], mec=cdic["disorder"], ms=ms, alpha=1)
    if Ferro_Phase: ax.plot(np.array(Ferro_Phase)[:, 0], np.array(Ferro_Phase)[:, 1], ls="", marker="s", mfc=cdic["ferro"], mec=cdic["ferro"], ms=ms, alpha=1)
    if Undecided_Phase: ax.plot(np.array(Undecided_Phase)[:, 0], np.array(Undecided_Phase)[:, 1], ls="", marker="s", mfc=cdic["undecided"], mec=cdic["undecided"], ms=ms, alpha=1)
    ax.set_xlabel("p")
    ax.set_ylabel("1/J")
    ax.tick_params(axis="both", direction="in", width=2, length=4)
    fig.tight_layout()
    plt.show()
    return Disorder_Phase, Ferro_Phase, Undecided_Phase