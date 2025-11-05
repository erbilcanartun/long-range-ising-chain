# Rewritten renormalization_hp_aferro.py using NumPy and Numba for speed
import numpy as np
import matplotlib.pyplot as plt
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

def generate_rg_flow(J0, n, max_k, num_steps, show_first=0, num_r_to_plot=5):
    max_d = 3 * max_k + 10
    rs = list(range(1, max_k + 1))
    d_arr = np.arange(1, max_d + 1, dtype=np.float64)
    J_arr = np.zeros(max_d + 1)
    if np.isfinite(J0):
        J_arr[1:] = J0 * np.power(d_arr, -n)
    else:
        J_arr[1:] = np.inf

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

    # Plot J vs RG step for first num_r_to_plot r
    plt.figure()
    steps = range(len(all_Js))
    for r_idx in range(min(num_r_to_plot, max_k)):
        Js_step = [all_Js[s][r_idx] for s in steps]
        plt.plot(steps, Js_step, label=f'r={r_idx+1}', marker='o')
    plt.xlabel('RG Step')
    plt.ylabel('J(r)')
    plt.title(f'RG Flow for J={J0}, n={n}')
    plt.xticks(steps)
    plt.legend()
    plt.grid(True)
    plt.show()

def find_J_c(n, max_k=1000, tol=1e-6, J_low=-3.0, J_high=-1e-10):
    if n <= 0 or n >= 2:
        raise ValueError("n must be between 0 and 2, excluding the edges.")

    start_track = 3
    max_steps = 5
    min_max_k = 2
    if max_k < min_max_k:
        raise ValueError(f"max_k should be at least {min_max_k}")

    def compute_flow(J0):
        max_d = 3 * max_k + 10
        d_arr = np.arange(1, max_d + 1, dtype=np.float64)
        J_arr = np.zeros(max_d + 1)
        J_arr[1:] = J0 * np.power(d_arr, -n)
        rs = list(range(1, max_k + 1))
        all_Js = [J_arr[1:max_k + 1].copy()]

        for step in range(1, max_steps + 1):
            Jps = np.zeros(max_k)
            for r_idx, r in enumerate(rs):
                start = 3 * r + 1
                Jp = compute_J_prime_func(start, J_arr)
                Jps[r_idx] = Jp
            all_Js.append(Jps)
            J_arr = np.zeros(max_d + 1)
            J_arr[1:max_k + 1] = Jps

            if step >= start_track:
                J_r2_current = Jps[1]
                J_r2_previous = all_Js[-2][1]
                if J_r2_current > J_r2_previous:
                    return all_Js, True
                if J_r2_current < J_r2_previous:
                    return all_Js, False

        J_r2_initial = all_Js[0][1]
        J_r2_final = all_Js[-1][1]
        return all_Js, J_r2_final > J_r2_initial

    iter_count = 0
    while J_high - J_low > tol and iter_count < 100:
        iter_count += 1
        J_mid = (J_low + J_high) / 2
        all_Js, is_growing = compute_flow(J_mid)
        if is_growing:
            J_high = J_mid
        else:
            J_low = J_mid
    return J_mid

def plot_J_c_vs_n(n_values, max_k=500, tol=1e-4, J_low=-3.0, J_high=-1e-10):
    J_c_values = []
    for n in n_values:
        try:
            J_c = find_J_c(n, max_k, tol, J_low, J_high)
            J_c_values.append(1 / J_c)
            print(f"For n={n}, J_c={J_c}")
        except ValueError as e:
            print(f"For n={n}, error: {e}")
            J_c_values.append(None)
    plt.figure()
    plt.plot(n_values, J_c_values, marker='o')
    plt.xlabel('n')
    plt.ylabel('$1/J_c$')
    plt.grid(True)
    plt.show()

def find_n_c(J0, max_k=1000, tol=1e-6, n_low=0.8, n_high=2.0):
    if J0 <= 0:
        raise ValueError("J0 must be positive for the ferromagnetic case.")

    start_track = 3
    max_steps = 5
    min_max_k = 2
    if max_k < min_max_k:
        raise ValueError(f"max_k must be at least {min_max_k}")

    def compute_flow(n_val):
        max_d = 3 * max_k + 10
        d_arr = np.arange(1, max_d + 1, dtype=np.float64)
        J_arr = np.zeros(max_d + 1)
        J_arr[1:] = J0 * np.power(d_arr, -n_val)
        rs = list(range(1, max_k + 1))
        all_Js = [J_arr[1:max_k + 1].copy()]

        for step in range(1, max_steps + 1):
            Jps = np.zeros(max_k)
            for r_idx, r in enumerate(rs):
                start = 3 * r + 1
                Jp = compute_J_prime_func(start, J_arr)
                Jps[r_idx] = Jp
            all_Js.append(Jps)
            J_arr = np.zeros(max_d + 1)
            J_arr[1:max_k + 1] = Jps

            if step >= start_track:
                J_r2_curr = Jps[1]
                J_r2_prev = all_Js[-2][1]
                if abs(J_r2_curr - J_r2_prev) < tol:
                    return all_Js, False
                if J_r2_curr > J_r2_prev:
                    return all_Js, True
                else:
                    return all_Js, False

        J_init = all_Js[0][1]
        J_final = all_Js[-1][1]
        return all_Js, J_final > J_init

    iter_count = 0
    while n_high - n_low > tol and iter_count < 100:
        iter_count += 1
        n_mid = (n_low + n_high) / 2
        _, is_growing = compute_flow(n_mid)
        if is_growing:
            n_low = n_mid
        else:
            n_high = n_mid
    n_c = (n_low + n_high) / 2
    return n_c