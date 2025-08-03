from mpmath import mp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils import mp_logsumexp

mp.dps = 40  # Set desired precision

plus_configs = [
    ([1,1,1], lambda J2, J4: 2*J2 + J4),
    ([1,1,-1], lambda J2, J4: -J4),
    ([1,-1,1], lambda J2, J4: -2*J2 + J4),
    ([-1,1,1], lambda J2, J4: -J4)
]

minus_configs = [
    ([-1,-1,-1], lambda J2, J4: 2*J2 + J4),
    ([-1,-1,1], lambda J2, J4: -J4),
    ([-1,1,-1], lambda J2, J4: -2*J2 + J4),
    ([1,-1,-1], lambda J2, J4: -J4)
]

def get_J(d, J0, n):
    if d <= 0:
        return mp.mpf(0)
    return mp.mpf(J0) / mp.power(mp.mpf(d), mp.mpf(n))

def compute_J_prime(start, J0, n):
    J2 = get_J(2, J0, n)
    J4 = get_J(4, J0, n)
    left_pos = [1, 3, 5]
    right_pos = [start, start + 2, start + 4]
    distances = []
    for iL in range(3):
        for iR in range(3):
            d = abs(right_pos[iR] - left_pos[iL])
            distances.append((iL, iR, d))

    def collect_totals(is_pp):
        confsR = plus_configs if is_pp else minus_configs
        totals = []
        for spinsL, El_func in plus_configs:
            El = El_func(J2, J4)
            for spinsR, Er_func in confsR:
                Er = Er_func(J2, J4)
                Eint = mp.mpf(0)
                for iL, iR, d in distances:
                    sign = mp.mpf(spinsL[iL] * spinsR[iR])
                    Eint += sign * get_J(d, J0, n)
                total = El + Er + Eint
                totals.append(total)
        return totals

    totals_pp = collect_totals(True)
    totals_pm = collect_totals(False)
    log_R_pp = mp_logsumexp(totals_pp)
    log_R_pm = mp_logsumexp(totals_pm)
    if log_R_pm == mp.ninf:
        return mp.inf
    return mp.mpf('0.5') * (log_R_pp - log_R_pm)

def compute_J_prime_func(start, J_func):
    J2 = J_func(2)
    J4 = J_func(4)
    left_pos = [1, 3, 5]
    right_pos = [start, start + 2, start + 4]
    distances = []
    for iL in range(3):
        for iR in range(3):
            d = abs(right_pos[iR] - left_pos[iL])
            distances.append((iL, iR, d))

    def collect_totals(is_pp):
        confsR = plus_configs if is_pp else minus_configs
        totals = []
        for spinsL, El_func in plus_configs:
            El = El_func(J2, J4)
            for spinsR, Er_func in confsR:
                Er = Er_func(J2, J4)
                Eint = mp.mpf(0)
                for iL_idx, iR_idx, d in distances:
                    sign = mp.mpf(spinsL[iL_idx] * spinsR[iR_idx])
                    Eint += sign * J_func(d)
                total = El + Er + Eint
                totals.append(total)
        return totals

    totals_pp = collect_totals(True)
    totals_pm = collect_totals(False)
    log_R_pp = mp_logsumexp(totals_pp)
    log_R_pm = mp_logsumexp(totals_pm)
    if log_R_pm == mp.ninf:
        return mp.inf
    return mp.mpf('0.5') * (log_R_pp - log_R_pm)


# Utility functions that are common

def plot_first_renormalized_J(max_k, J0, n):
    ks = []
    Jps = []
    for k in range(1, max_k + 1):
        start = 4 + 3 * (k - 1)
        Jp = compute_J_prime(start, J0, n)
        ks.append(k)
        Jps.append(Jp)
        print(f"For cluster {k} (right starting at {start}), J' = {Jp}")

    plt.figure()
    plt.plot(ks, [float(j) for j in Jps], marker='o')
    plt.xlabel('Cluster number k')
    plt.ylabel("Renormalized bond J'")
    plt.title(f"J' vs k for J0={J0}, n={n}")
    plt.grid(True)
    plt.show()

def plot_non_renormalized_J(J0, n, max_r=30):
    rs = list(range(1, max_r + 1))
    Js = [get_J(r, J0, n) for r in rs]

    plt.figure()
    plt.plot(rs, [float(j) for j in Js], marker='o')
    plt.xlabel('Distance r')
    plt.ylabel("Non-renormalized J(r)")
    plt.title(f"J(r) = {J0} / r^{n} vs r")
    plt.grid(True)
    plt.show()

def plot_both(J0, n, max_k=10):
    # Non-renormalized
    rs = list(range(1, max_k + 1))
    Js = [get_J(r, J0, n) for r in rs]

    # Renormalized
    effective_rs = list(range(1, max_k + 1))
    Jps = []
    for k in range(1, max_k + 1):
        start = 4 + 3 * (k - 1)
        Jp = compute_J_prime(start, J0, n)
        Jps.append(Jp)
        print(f"For cluster {k} (effective r={effective_rs[k-1]}), J' = {Jp}")

    plt.figure()
    plt.plot(rs, [float(j) for j in Js], label='Non-renormalized J(r)', marker='o', linestyle='-')
    plt.plot(effective_rs, [float(j) for j in Jps], label='Renormalized J\'(r)', marker='x', linestyle='--')
    plt.xlabel('Distance r')
    plt.ylabel('J')
    plt.title(f"Non-renormalized and Renormalized J vs r for J0={J0}, n={n}")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_rg_flow(J0, n, max_k, num_steps, show_first=0, num_r_to_plot=5):
    J_func = lambda d: get_J(d, J0, n) if d > 0 else mp.mpf(0)
    rs = list(range(1, max_k + 1))
    initial_Js = [J_func(r) for r in rs]
    all_Js = [initial_Js]

    if show_first > 0:
        print("Step 0 (initial):")
        for i in range(min(show_first, len(initial_Js))):
            print(f"  J({i+1}) = {initial_Js[i]}")

    for step in range(1, num_steps + 1):
        Jps = []
        for r in rs:
            start = 3 * r + 1
            Jp = compute_J_prime_func(start, J_func)
            Jps.append(Jp)
        all_Js.append(Jps)

        if show_first > 0:
            print(f"Step {step}:")
            for i in range(min(show_first, len(Jps))):
                print(f"  J({i+1}) = {Jps[i]}")

        # Update J_func for next step
        J_dict = {r: Jps[r-1] for r in rs}
        J_func = lambda d: J_dict.get(d, mp.mpf(0)) if d > 0 else mp.mpf(0)

    # Plot J vs RG step for first num_r_to_plot r
    plt.figure()
    steps = range(len(all_Js))
    for r_idx in range(min(num_r_to_plot, max_k)):
        Js_step = [all_Js[s][r_idx] for s in steps]
        plt.plot(steps, [float(j) for j in Js_step], label=f'r={r_idx+1}', marker='o')
    plt.xlabel('RG Step')
    plt.ylabel('J(r)')
    plt.title(f'RG Flow for J={J0}, n={n}')
    plt.xticks(steps)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rg_steps_vs_r(J0, n, max_k=10, num_steps=1, plot_up_to_r=None, colormap=False, fig_name=False):
    J_func = lambda d: get_J(d, J0, n) if d > 0 else mp.mpf(0)
    rs = list(range(1, max_k + 1))
    all_Js = [[J_func(r) for r in rs]]

    for step in range(1, num_steps + 1):
        Jps = []
        for r in rs:
            start = 3 * r + 1
            Jp = compute_J_prime_func(start, J_func)
            Jps.append(Jp)
        all_Js.append(Jps)

        # Update J_func for next step
        J_dict = {r: Jps[r-1] for r in rs}
        J_func = lambda d: J_dict.get(d, mp.mpf(0)) if d > 0 else mp.mpf(0)

    up_to = plot_up_to_r if plot_up_to_r is not None else max_k
    plot_rs = list(range(1, up_to + 1))

    if colormap:
        # Create figure and axes
        fig, ax = plt.subplots()

        # Plot J vs r for each RG step with color gradient
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=num_steps)
        for step in range(num_steps + 1):
            Js = all_Js[step][:up_to]
            color = cmap(norm(step))
            ax.plot(plot_rs, [float(j) for j in Js], color=color, marker='o')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='RG Step')

        # Set labels and title
        ax.set_xlabel('Distance r')
        ax.set_ylabel('J(r)')
        ax.set_title(f'J={J0}, n={n}')
        ax.grid(True)
        plt.show()

    else:
        # Plot J vs r for each RG step
        fig, ax = plt.subplots()
        for step in range(num_steps + 1):
            Js = all_Js[step][:up_to]
            ax.plot(plot_rs, [float(j) for j in Js], label=f'Step {step}', marker='o')
        # Set labels and title
        ax.set_xlabel('Distance r')
        ax.set_ylabel('J(r)')
        ax.set_title(f'J={J0}, n={n}')
        ax.grid(True)
        plt.legend()
        plt.show()

    if fig_name:
        fig.savefig("../results/" + fig_name)

def find_J_c(n, max_k=1000, tol=1e-6, J_low=1e-10, J_high=3.0):
    """
    Finds the critical J_c for the given n using binary search and RG flow simulation.

    This function performs a binary search to find the critical coupling J_c where the 
    renormalization group (RG) flow transitions between growing and decaying for the 
    interaction J(d) = J0 / d^n. It simulates the RG flow and checks the behavior of 
    the coupling at r=2 after a minimum number of steps to avoid initial fluctuations.

    Parameters:
    n (float): The exponent in the power-law interaction. Must be 0 < n < 2.
    max_k (int): Maximum distance r to consider in the RG flow.
    tol (float, optional): Tolerance for the binary search convergence. Default is 1e-6.
    J_low (float, optional): Lower bound for binary search. Default is 0.0.
    J_high (float, optional): Upper bound for binary search. Default is 3.0.

    Returns:
    mp.mpf: The critical J_c value.

    Raises:
    ValueError: If n is not in (0, 2) or max_k is too small.
    """
    # Check if n is within the allowed range (0 < n < 2)
    if n <= 0 or n >= 2:
        raise ValueError("n must be between 0 and 2, excluding the edges.")

    start_track = 3  # Start checking after 3 RG steps to avoid initial fluctuations
    max_steps = 5   # Maximum number of RG steps to perform
    tol = mp.mpf(tol)
    J_low = mp.mpf(J_low)
    J_high = mp.mpf(J_high)
    min_max_k = 2  # Since we only need at least r=2
    if max_k < min_max_k:
        raise ValueError(f"max_k should be at least {min_max_k} for smooth and correct calculation with r=2")

    def compute_flow(J0):
        # Initialize J function for initial step
        J_func = lambda d: get_J(d, J0, n) if d > 0 else mp.mpf(0)
        rs = list(range(1, max_k + 1))
        all_Js = [[J_func(r) for r in rs]]  # Store J values at each step

        for step in range(1, max_steps + 1):
            Jps = []
            for r in rs:
                start = 3 * r + 1
                Jp = compute_J_prime_func(start, J_func)
                Jps.append(Jp)
            all_Js.append(Jps)

            # After start_track steps, check the change in J(r=2) from previous to current
            if step >= start_track:
                J_r2_current = Jps[1]  # J at r=2 (index 1)
                J_r2_previous = all_Js[-2][1]  # J at r=2 from previous step
                if J_r2_current > J_r2_previous:
                    return all_Js, True  # Growing (ferromagnetic)
                if J_r2_current < J_r2_previous:
                    return all_Js, False  # Decaying (paramagnetic)

            # Update J_func for the next step
            J_dict = {r: Jps[r-1] for r in rs}
            J_func = lambda d: J_dict.get(d, mp.mpf(0)) if d > 0 else mp.mpf(0)

        # Fallback if no decision within max_steps: compare initial and final J(r=2)
        J_r2_initial = all_Js[0][1]
        J_r2_final = all_Js[-1][1]
        return all_Js, J_r2_final > J_r2_initial

    # Binary search loop
    iter_count = 0
    while J_high - J_low > tol and iter_count < 100:
        iter_count += 1
        J_mid = (J_low + J_high) / 2
        all_Js, is_growing = compute_flow(J_mid)
        if is_growing:
            J_high = J_mid  # Growing: search lower half
        else:
            J_low = J_mid  # Decaying: search upper half
    return J_mid

def plot_J_c_vs_n(n_values, max_k=500, tol=1e-4):
    J_c_values = []
    min_max_k = 2  # Since we only need r=2
    if max_k < min_max_k:
        raise ValueError(f"max_k should be at least {min_max_k} for smooth and correct calculation with r=2")

    for n in n_values:
        try:
            J_c = find_J_c(n, max_k, tol, J_low=1e-3, J_high=3.0)
            J_c_values.append(1/float(J_c))
            print(f"For n={n}, J_c={J_c}")
        except ValueError as e:
            print(f"For n={n}, error: {e}")
            J_c_values.append(None)  # Append None for invalid n

    plt.figure()
    plt.plot(n_values, J_c_values, marker='o')
    plt.xlabel('n')
    plt.ylabel('$1/J_c$')
    plt.grid(True)
    plt.show()