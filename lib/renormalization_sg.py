from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import mp_logsumexp

mp.dps = 30  # Set desired precision

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

def get_J(d, J0, n, signs):
    if d <= 0:
        return mp.mpf(0)
    # Get the sign for distance d from the signs dictionary
    sign = signs.get(d, 1)  # Default to +1 if d not in signs
    return mp.mpf(float(sign) * J0) / mp.power(mp.mpf(d), mp.mpf(n))

def compute_J_prime(start, J0, n, signs):
    J2 = get_J(2, J0, n, signs)
    J4 = get_J(4, J0, n, signs)
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
                    Eint += sign * get_J(d, J0, n, signs)
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

def generate_rg_flow(J0, n, p, max_k, num_steps, show_first=0, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)
    signs = {d: np.random.choice([-1, 1], p=[p, 1-p]) for d in range(1, max_k + 1)}
    J_func = lambda d: get_J(d, J0, n, signs) if d > 0 else mp.mpf(0)

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

    return all_Js

def phase_sink(all_Js, skip_steps=3, track_rs=[2,3]):
    if len(all_Js) < skip_steps + 2:
        return 'undecided'

    # Skip the first skip_steps
    tracked_flow = all_Js[skip_steps:skip_steps+2]

    # Check the last two steps
    if len(tracked_flow) < 2:
        return 'undecided'
    prev = tracked_flow[-2]
    last = tracked_flow[-1]

    # Check if both J(2) and J(3) are positive in the last step
    all_positive = all(float(last[r-1]) > 0 for r in track_rs)

    # Compute strengths (absolute values)
    S_prev = sum(abs(float(prev[r-1])) for r in track_rs)
    S_last = sum(abs(float(last[r-1])) for r in track_rs)

    if all_positive and S_last > S_prev:
        return 'ferro'
    elif S_last < S_prev:  # Strength decreasing towards zero
        return 'disorder'
    else:
        return 'undecided'

def find_Tc_fixed_p(n, p, max_k=1000, tol=1e-6, J_low=0.01, J_high=10.0, skip_steps=3, track_rs=[2,3], seed=42):
    """
    For fixed p, search for Tc (1/Jc) assuming disorder at high T (low J) and ordered at low T (high J).
    """
    if n <= 0 or n >= 2:
        raise ValueError("n must be between 0 and 2, excluding the edges.")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1.")

    def compute_phase(J0):
        all_Js = generate_rg_flow(J0, n, p, max_k, num_steps=skip_steps+2, seed=seed)
        phase = phase_sink(all_Js, skip_steps, track_rs)
        if phase == 'undecided':
            print(f"Warning: Phase undecided for n={n}, p={p}, J={J0}")
        return phase == 'disorder'  # True if disorder, False if not

    # Bisection: find where phase changes from disorder (high T) to ordered (low T)
    iter_count = 0
    while J_high - J_low > tol and iter_count < 100:
        iter_count += 1
        J_mid = (J_low + J_high) / 2
        phase_mid = compute_phase(J_mid)
        if phase_mid:
            J_low = J_mid  # Disorder: move to higher J (lower T)
        else:
            J_high = J_mid  # Ordered: move to lower J (higher T)
    Jc = (J_low + J_high) / 2
    return 1 / float(Jc)  # Tc = 1 / Jc

def find_pc_fixed_T(n, T, max_k=1000, tol=1e-6, p_low=0.0, p_high=1.0, skip_steps=3, track_rs=[2,3], seed=42):
    """
    For fixed T (J = 1/T), search for critical p assuming ferro at low p.
    """
    if n <= 0 or n >= 2:
        raise ValueError("n must be between 0 and 2, excluding the edges.")
    if T <= 0:
        raise ValueError("T must be positive.")

    J0 = mp.mpf(1.0 / T)  # Fixed J = 1/T

    def compute_phase(curr_p):
        all_Js = generate_rg_flow(J0, n, float(curr_p), max_k, num_steps=skip_steps+2, seed=seed)
        phase = phase_sink(all_Js, skip_steps, track_rs)
        if phase == 'undecided':
            print(f"Warning: Phase undecided for n={n}, p={curr_p}, J={J0}")
        return phase == 'ferro'  # True if ferro, False if not

    # Bisection: find where phase changes from ferro (low p) to disorder (high p)
    iter_count = 0
    while p_high - p_low > tol and iter_count < 100:
        iter_count += 1
        p_mid = (p_low + p_high) / 2
        phase_mid = compute_phase(p_mid)
        if phase_mid:
            p_low = p_mid  # Ferro: move to higher p
        else:
            p_high = p_mid  # Disorder: move to lower p
    return float((p_low + p_high) / 2)

def phase_identify(n, p, J):
    all_Js = generate_rg_flow(J, n, p, max_k=5000, num_steps=5)
    phase = phase_sink(all_Js)
    return phase

def plot_phase_diagram(n, p_values, one_over_J_values, max_k=5000, num_steps=5):
    """
    Generate phase diagram by scanning p, 1/J parameter space for fixed n
    The marker size is dynamically adjusted based on the grid dimensions.
    """
    Disorder_Phase, Ferro_Phase, Undecided_Phase = [],[],[]
    for i, p in enumerate(tqdm(p_values)):
        for j, one_over_j in enumerate(one_over_J_values):
            J = 1.0 / one_over_j
            phase = phase_identify(n, p, J)
            if phase == "disorder":
                Disorder_Phase.append([p, one_over_j])
            elif phase == "ferro":
                Ferro_Phase.append([p, one_over_j])
            else:
                Undecided_Phase.append([p, one_over_j])
    # Calculate appropriate marker size for plotting
    grid_size = len(p_values) * len(one_over_J_values)
    base_marker_size = 25 # Base size for a 10x10 grid
    reference_grid_size = 100
    ms = base_marker_size * np.sqrt(reference_grid_size / grid_size)
    ms = max(1, min(ms, 12)) # Limit between 1 and 12
    # Plot the results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=100)
    fig.set_facecolor("white")
    plt.rc(group="font", family="Arial", weight="bold", size=10)
    plt.rc(group="lines", linewidth=1)
    plt.rc(group="axes", linewidth=2)
    cdic = {"disorder":"grey",
            "ferro":"red",
            "undecided":"yellow"}
    if Disorder_Phase: ax.plot(np.array(Disorder_Phase)[:,0], np.array(Disorder_Phase)[:,1], ls="", marker="s", mfc=cdic["disorder"], mec=cdic["disorder"], ms=ms, alpha=1)
    if Ferro_Phase: ax.plot(np.array(Ferro_Phase)[:,0], np.array(Ferro_Phase)[:,1], ls="", marker="s", mfc=cdic["ferro"], mec=cdic["ferro"], ms=ms, alpha=1)
    if Undecided_Phase: ax.plot(np.array(Undecided_Phase)[:,0], np.array(Undecided_Phase)[:,1], ls="", marker="s", mfc=cdic["undecided"], mec=cdic["undecided"], ms=ms, alpha=1)
    ax.set_xlabel("p")
    ax.set_ylabel("1/J")
    ax.tick_params(axis="both", direction="in", width=2, length=4)
    fig.tight_layout()
    plt.show()
    return Disorder_Phase, Ferro_Phase, Undecided_Phase