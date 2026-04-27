import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from numba import njit
from utils import _xorshift64star_next
from decimation_staggered import (
    required_initial_max_distance,
    r_max,
    log_Rpp_Rpm,
)


# ----------------------------
# Deterministic Fisher-Yates on a length-D array (indices 1..D), Numba-safe
# ----------------------------

@njit(cache=True)
def _fisher_yates_1toD(arr, seed_mix):
    """
    In-place Fisher-Yates shuffle of arr[1..len(arr)-1], leaving arr[0] alone.
    `seed_mix` is a uint64 already mixed by the caller.
    """
    state = seed_mix
    if state == np.uint64(0):
        state = np.uint64(0xD1B54A32D192ED03)
    n = arr.shape[0] - 1  # number of shuffle-eligible entries, indices 1..n
    for i in range(n, 1, -1):
        state, rnd = _xorshift64star_next(state)
        j = 1 + int(rnd % np.uint64(i))  # j in [1, i]
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


# ----------------------------
# Three-state coupling sign/zero pattern
# ----------------------------

@njit(cache=True)
def generate_states(D, p, r_frac, seed):
    """
    Return an int8 array `state[0..D]` with state[0]=0 unused, and for r=1..D:
        state[r] =  0 with exact count floor(r_frac * D)
        state[r] = -1 with exact count floor(p * (D - n_zero))
        state[r] = +1 with the remainder

    Then shuffled (Fisher-Yates, deterministic by seed).
    """
    state_arr = np.zeros(D + 1, dtype=np.int8)

    n_zero  = int(r_frac * D)
    n_nz    = D - n_zero
    n_minus = int(p * n_nz)
    n_plus  = n_nz - n_minus

    idx = 1
    for _ in range(n_zero):
        state_arr[idx] = 0
        idx += 1
    for _ in range(n_minus):
        state_arr[idx] = -1
        idx += 1
    for _ in range(n_plus):
        state_arr[idx] = 1
        idx += 1

    seed_mix = np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
    _fisher_yates_1toD(state_arr, seed_mix)
    return state_arr


@njit(cache=True)
def build_J_dilute_signed(J0, a, D, p, r_frac, seed):
    """
    Initial coupling vector with three-state distribution.
    J[0] = 0. For r>=1, J[r] = state[r] * J0 / r^a, where state[r] in {-1,0,+1}.
    """
    states = generate_states(D, p, r_frac, seed)
    J = np.zeros(D + 1, dtype=np.float64)
    for r in range(1, D + 1):
        if states[r] != 0:
            J[r] = (float(states[r]) * J0) / (r ** a)
        # else leave as 0.0
    return J


# ----------------------------
# Direction-aware phase determination with early stopping
# ----------------------------

def determine_phase_at_directional_early_stop(
    J0, a, p, r_frac,
    max_dist_final,
    n_steps_total,
    seed=12345,
    TM_rs=(2, 3, 4),
    min_check_step=3,
    thresh_one=0.8,
    thresh_zero=0.2,
):
    """
    Generate RG flow and classify phases from successive transfer matrices.

    After min_check_step, compare

        T_current(r) - T_previous(r)

    for every r in TM_rs.

    If all chosen distances classify to the same non-undetermined sink,
    stop early and return that phase.
    """
    TM_rs = tuple(TM_rs)

    D0 = required_initial_max_distance(max_dist_final, n_steps_total)
    J = build_J_dilute_signed(J0, a, D0, p, r_frac, seed)

    flow = [J.copy()]

    TM_history = []

    # initial transfer matrices
    TM0 = {}
    for r in TM_rs:
        if r < len(J):
            TM0[r] = construct_transfer_matrix(J, r, normalize=True)
    TM_history.append(TM0)

    for step in range(1, n_steps_total + 1):
        J = rg_step(J)
        flow.append(J.copy())

        D_now = len(J) - 1

        if any(r > D_now for r in TM_rs):
            return "undetermined", {
                "reason": "tracked_distance_out_of_range",
                "stop_step": step,
                "D_now": D_now,
                "TM_rs": TM_rs,
                "flow": flow,
                "TM_history": TM_history,
                "J_final": J.copy(),
            }

        TM_current = {}
        for r in TM_rs:
            TM_current[r] = construct_transfer_matrix(J, r, normalize=True)

        TM_history.append(TM_current)

        if step < min_check_step:
            continue

        TM_prev = TM_history[-2]

        phases = []

        for r in TM_rs:
            ph = classify_TM_by_direction(
                T_current=TM_current[r],
                T_prev=TM_prev[r],
                thresh_one=thresh_one,
                thresh_zero=thresh_zero,
            )
            phases.append(ph)

        if len(set(phases)) == 1 and phases[0] != "undetermined":
            return phases[0], {
                "reason": "directional_early_sink_reached",
                "stop_step": step,
                "TM_rs": TM_rs,
                "TM_phases": phases,
                "TM_current": TM_current,
                "TM_prev": TM_prev,
                "TM_history": TM_history,
                "flow": flow,
                "J_final": J.copy(),
            }

    return "undetermined", {
        "reason": "max_steps_reached",
        "stop_step": n_steps_total,
        "TM_rs": TM_rs,
        "TM_history": TM_history,
        "flow": flow,
        "J_final": J.copy(),
    }


# ----------------------------
# One RG step: staggered, head-only, shrinking vector
# ----------------------------

@njit(cache=True)
def rg_step(J):
    """
    One head-only staggered RG step. No tail reconstruction.
    Output length = r_max(D) + 1, so the vector shrinks each iteration.

    Uses exactly the same per-distance recursion as the existing code:
        J'_r = 0.5 * (log R_++ - log R_+-)
    with R_++ and R_+- coming from decimation_staggered.log_Rpp_Rpm.
    Zero entries in J are handled implicitly: a zero bond contributes e^0=1
    inside the cell-pair partition sums, so nothing special is needed here.
    """
    D = J.shape[0] - 1
    rstop = r_max(D)

    J_new = np.zeros(rstop + 1, dtype=np.float64)
    for rr in range(1, rstop + 1):
        log_pp, log_pm = log_Rpp_Rpm(rr, J)
        J_new[rr] = 0.5 * (log_pp - log_pm)
    return J_new

# ----------------------------
# Transfer matrix at a single distance
# ----------------------------

def construct_transfer_matrix(J, r, normalize=True):
    """
    2x2 transfer matrix for the spin-1/2 Ising bond at distance r:
        T(J_r) = [[exp(J_r), exp(-J_r)],
                  [exp(-J_r), exp(J_r)]]
    Normalized by its maximum entry when normalize=True.
    """
    if r >= len(J):
        raise ValueError(
            f"distance r={r} > max available {len(J)-1}"
        )
    Jr = float(J[r])
    T = np.array(
        [[np.exp(Jr),  np.exp(-Jr)],
         [np.exp(-Jr), np.exp(Jr)]],
        dtype=np.float64,
    )
    if normalize:
        T = T / T.max()
    return T


# ----------------------------
# Classify a 2x2 normalized TM using sink direction
# ----------------------------

def classify_TM_by_direction(
    T_current,
    T_prev,
    thresh_one=0.8,
    thresh_zero=0.2,
):
    """
    Direction-aware classification of a normalized 2x2 Ising transfer matrix.

    Ferro sink:
        [[1, 0],
         [0, 1]]

    Antiferro sink:
        [[0, 1],
         [1, 0]]

    Disorder sink:
        [[1, 1],
         [1, 1]]

    A sink is accepted only if:
      - entries expected to become 1 are already large and increasing,
      - entries expected to become 0 are already small and decreasing.
    """
    if T_current is None or T_prev is None:
        return "undetermined"

    diff = T_current - T_prev

    def matches_sink(one_positions, zero_positions):
        for i, j in one_positions:
            if T_current[i, j] < thresh_one:
                return False
            if diff[i, j] < 0:
                return False

        for i, j in zero_positions:
            if T_current[i, j] > thresh_zero:
                return False
            if diff[i, j] >= 0:
                return False

        return True

    one_ferro = [(0, 0), (1, 1)]
    zero_ferro = [(0, 1), (1, 0)]

    one_antiferro = [(0, 1), (1, 0)]
    zero_antiferro = [(0, 0), (1, 1)]

    one_disorder = [(0, 0), (0, 1), (1, 0), (1, 1)]
    zero_disorder = []

    if matches_sink(one_ferro, zero_ferro):
        return "ferro"

    if matches_sink(one_antiferro, zero_antiferro):
        return "antiferro"

    if matches_sink(one_disorder, zero_disorder):
        return "disorder"

    return "undetermined"




def scan_phase_sinks_p_T_directional_early_stop_seed_avg(
    p_values,
    T_values,
    a,
    r_frac,
    max_dist_final,
    n_steps_total,
    n_seeds=25,
    base_seed=12345,
    TM_rs=(2, 3, 4),
    min_check_step=3,
    thresh_one=0.8,
    thresh_zero=0.2,
    n_jobs=1,
    progress="text",
    report_every=None,
):
    """
    Seed-averaged directional (p, T) scan, optionally parallel.

    For each (p, T) point, runs `n_seeds` independent realizations using
    `determine_phase_at_directional_early_stop` and records per-seed phase
    counts, the majority-among-determined phase, and the confidence.

    Parallelism
    -----------
    n_jobs : int
        Number of worker processes for joblib's loky backend.
            n_jobs == 1   -> serial execution (no joblib dependency required)
            n_jobs == -1  -> use all available cores
            n_jobs == k>1 -> use k workers
        Tasks are individual (point, seed) RG runs, flattened across the
        whole grid for good load balance (early-stopped runs are short,
        max-step runs are long, and mixing them across workers evens out).
        Results are fully reproducible regardless of n_jobs because each
        run is keyed by its own seed = base_seed + k.

    Returns
    -------
    point_results : dict
        Maps (float(p), float(T)) -> summary dict (see seed_average_at_point).
    bins : dict
        Convenience aggregation by majority phase:
            "ferro" / "antiferro" / "disorder" / "undetermined"
        each mapping to an (N, 2) array of (p, T) points.
        Confidence info is NOT in bins -- use point_results for that.
    """
    import time

    p_values = np.asarray(p_values, dtype=np.float64)
    T_values = np.asarray(T_values, dtype=np.float64)

    n_points = len(p_values) * len(T_values)
    n_runs = n_points * n_seeds

    # ---- build the flat task list: one task per (point, seed) ----
    # Each task is fully independent, returns ((p, T), phase).
    tasks = []
    for p in p_values:
        for T in T_values:
            for k in range(n_seeds):
                tasks.append((float(p), float(T), int(base_seed) + k))

    # ---- progress setup ----
    use_tqdm = False
    pbar = None
    if progress == "tqdm":
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(
                total=n_runs,
                desc=(
                    f"seed-avg scan a={a:g}, r_frac={r_frac:g}, "
                    f"n_seeds={n_seeds}, n_jobs={n_jobs}"
                ),
            )
            use_tqdm = True
        except ImportError:
            progress = "text"

    if report_every is None:
        report_every = max(1, n_runs // 20)

    def _fmt_time(s):
        if s < 0 or not np.isfinite(s):
            return "?"
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    # ---- accumulators ----
    # counts_grid[(p, T)] -> {"ferro": int, "antiferro": int,
    #                        "disorder": int, "undetermined": int}
    counts_grid = {}
    for p in p_values:
        for T in T_values:
            counts_grid[(float(p), float(T))] = {
                "ferro": 0, "antiferro": 0,
                "disorder": 0, "undetermined": 0,
            }

    t0 = time.time()
    done = 0

    def _consume(result):
        """Fold one ((p, T), phase) result into counts_grid + progress."""
        nonlocal done
        (pp, TT), ph = result
        counts_grid[(pp, TT)][ph] += 1
        done += 1
        if use_tqdm:
            pbar.update(1)
        elif progress == "text" and (done % report_every == 0 or done == n_runs):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_runs - done) / rate if rate > 0 else float("inf")
            print(
                f"  [{done:>{len(str(n_runs))}}/{n_runs}] "
                f"{100.0 * done / n_runs:5.1f}%  "
                f"elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}",
                flush=True,
            )

    # ---- execute ----
    if n_jobs == 1:
        # Serial path: no joblib import, identical numerics.
        for (p, T, seed_k) in tasks:
            res = _run_one(
                p, T, seed_k,
                a, r_frac, max_dist_final, n_steps_total,
                TM_rs, min_check_step, thresh_one, thresh_zero,
            )
            _consume(res)
    else:
        # Parallel path via joblib's loky backend.
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise ImportError(
                "n_jobs != 1 requires joblib. Install it via "
                "`pip install joblib`, or call with n_jobs=1."
            ) from e

        # `return_as='generator_unordered'` (joblib >= 1.3) lets us update
        # the progress bar as each result comes in. Fall back to a plain
        # list if the installed joblib is older.
        try:
            results_iter = Parallel(
                n_jobs=n_jobs, backend="loky",
                return_as="generator_unordered",
            )(
                delayed(_run_one)(
                    p, T, seed_k,
                    a, r_frac, max_dist_final, n_steps_total,
                    TM_rs, min_check_step, thresh_one, thresh_zero,
                )
                for (p, T, seed_k) in tasks
            )
            for res in results_iter:
                _consume(res)
        except TypeError:
            # Older joblib: no streaming -> get the full list at once,
            # then update progress.
            all_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_one)(
                    p, T, seed_k,
                    a, r_frac, max_dist_final, n_steps_total,
                    TM_rs, min_check_step, thresh_one, thresh_zero,
                )
                for (p, T, seed_k) in tasks
            )
            for res in all_results:
                _consume(res)

    if use_tqdm:
        pbar.close()

    # ---- summarize per point and bin by majority phase ----
    point_results = {}
    bin_lists = {
        "ferro":        [],
        "antiferro":    [],
        "disorder":     [],
        "undetermined": [],
    }
    for (pp, TT), counts in counts_grid.items():
        summary = _summarize_counts(counts, n_total=n_seeds)
        point_results[(pp, TT)] = summary
        bin_lists[summary["majority_phase"]].append((pp, TT))

    def _arr(lst):
        return np.array(lst, dtype=np.float64).reshape((-1, 2))

    bins = {key: _arr(val) for key, val in bin_lists.items()}
    return point_results, bins



def plot_phase_points_p_T_seed_avg(
    point_results,
    a=None,
    r_frac=None,
    title=None,
    marker_size=40,
    conf_full=0.8,
    conf_min=0.6,
    ax=None,
    show=True,
    legend=True,
):
    """
    Scatter plot of seed-averaged phase points in the (p, T) plane.

    Each (p, T) point is colored by its majority phase among determined
    seeds, with saturation set by the confidence:

        confidence >= conf_full  -> full base color
        conf_min   <= confidence <  conf_full
                                -> linearly desaturated toward white
        confidence <  conf_min  -> light gray (ambiguous)
        all-undetermined        -> orange (own category)

    Parameters
    ----------
    point_results : dict
        Output of `scan_phase_sinks_p_T_directional_early_stop_seed_avg`.
        Maps (p, T) -> summary dict with keys 'majority_phase' and
        'confidence'.
    conf_full, conf_min : float
        The full-saturation and ambiguity thresholds. Must satisfy
        0 <= conf_min < conf_full <= 1.
    """
    if not (0.0 <= conf_min < conf_full <= 1.0):
        raise ValueError(
            f"need 0 <= conf_min < conf_full <= 1, "
            f"got conf_min={conf_min}, conf_full={conf_full}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
    else:
        fig = ax.figure

    # Build per-point coordinate / color arrays in one pass.
    pts_x = []
    pts_y = []
    pts_c = []  # RGB tuples

    # Track which legend categories actually appear in the data.
    seen_phases = set()
    seen_ambiguous = False

    for (p, T), s in point_results.items():
        phase = s["majority_phase"]
        conf = s["confidence"]

        if phase == "undetermined":
            color = to_rgb(_PHASE_COLORS["undetermined"])
            seen_phases.add("undetermined")
        elif conf < conf_min:
            color = to_rgb(_AMBIGUOUS_COLOR)
            seen_ambiguous = True
        else:
            color = _shade_color(
                base_color=_PHASE_COLORS[phase],
                confidence=conf,
                conf_full=conf_full,
                conf_min=conf_min,
            )
            seen_phases.add(phase)

        pts_x.append(p)
        pts_y.append(T)
        pts_c.append(color)

    ax.scatter(pts_x, pts_y, c=pts_c, marker='s', s=marker_size,
               edgecolors='none')

    ax.set_xlabel(r"Antiferromagnetic bond concentration $p$")
    ax.set_ylabel(r"Temperature $1/J$")

    if title is None and (a is not None) and (r_frac is not None):
        title = (
            rf"Seed-averaged phase diagram   "
            rf"$a={a:g}$,   $r_{{\rm frac}}={r_frac:g}$"
        )
    if title is not None:
        ax.set_title(title)

    if legend:
        # Build a legend that shows each phase at full saturation, plus
        # the ambiguous category if it appears.
        handles = []
        for ph in ("ferro", "antiferro", "disorder"):
            if ph in seen_phases:
                handles.append(Patch(
                    facecolor=_PHASE_COLORS[ph],
                    edgecolor='none',
                    label=_PHASE_LABELS[ph],
                ))
        if seen_ambiguous:
            handles.append(Patch(
                facecolor=_AMBIGUOUS_COLOR,
                edgecolor='none',
                label=_AMBIGUOUS_LABEL,
            ))
        if "undetermined" in seen_phases:
            handles.append(Patch(
                facecolor=_PHASE_COLORS["undetermined"],
                edgecolor='none',
                label=_PHASE_LABELS["undetermined"],
            ))

        # Confidence note
        conf_note = (
            f"Saturation: full at conf $\\geq$ {conf_full:g}, "
            f"fades to white by conf = {conf_min:g}"
        )
        if handles:
            leg = ax.legend(handles=handles, loc='best',
                            frameon=True, fontsize=9, title=conf_note,
                            title_fontsize=8)
            # Compact title look
            leg._legend_box.align = "left"

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ----------------------------
# Per-run worker (module-level so joblib can pickle it across processes)
# ----------------------------

def _run_one(
    p, T, seed_k,
    a, r_frac, max_dist_final, n_steps_total,
    TM_rs, min_check_step, thresh_one, thresh_zero,
):
    """
    Single (point, seed) RG run. Returns ((p, T), phase_string).

    Lives at module scope so joblib's loky backend can pickle and ship it
    to worker processes. Thin wrapper around
    `determine_phase_at_directional_early_stop`.
    """
    J0 = 1.0 / float(T)
    phase, _info = determine_phase_at_directional_early_stop(
        J0=J0,
        a=a,
        p=float(p),
        r_frac=float(r_frac),
        max_dist_final=max_dist_final,
        n_steps_total=n_steps_total,
        seed=int(seed_k),
        TM_rs=TM_rs,
        min_check_step=min_check_step,
        thresh_one=thresh_one,
        thresh_zero=thresh_zero,
    )
    if phase not in ("ferro", "antiferro", "disorder"):
        phase = "undetermined"
    return (float(p), float(T)), phase


# ----------------------------
# Single-point seed average
# ----------------------------

def seed_average_at_point(
    J0, a, p, r_frac,
    max_dist_final,
    n_steps_total,
    n_seeds,
    base_seed=12345,
    TM_rs=(2, 3, 4),
    min_check_step=3,
    thresh_one=0.8,
    thresh_zero=0.2,
):
    """
    Run `n_seeds` independent disorder realizations at a single (p, T) point
    and aggregate the directional-TM phase outcomes.

    Seeds used are `base_seed + k` for k=0..n_seeds-1.

    Returns
    -------
    summary : dict with keys
        counts            : {phase: int}
        n_total           : int                 (== n_seeds)
        n_determined      : int                 (counts excluding 'undetermined')
        majority_phase    : str
            'ferro' | 'antiferro' | 'disorder' | 'undetermined'
            (undetermined only if n_determined == 0)
        confidence        : float in [0, 1]
            For non-undetermined majority: count(majority) / n_determined.
            For all-undetermined points: 0.0.
        undetermined_frac : float in [0, 1]
            count('undetermined') / n_total.
    """
    counts = {"ferro": 0, "antiferro": 0, "disorder": 0, "undetermined": 0}

    for k in range(n_seeds):
        seed_k = int(base_seed) + k
        phase, _info = determine_phase_at_directional_early_stop(
            J0=J0,
            a=a,
            p=p,
            r_frac=r_frac,
            max_dist_final=max_dist_final,
            n_steps_total=n_steps_total,
            seed=seed_k,
            TM_rs=TM_rs,
            min_check_step=min_check_step,
            thresh_one=thresh_one,
            thresh_zero=thresh_zero,
        )
        if phase not in counts:
            phase = "undetermined"
        counts[phase] += 1

    n_total = n_seeds
    n_und = counts["undetermined"]
    n_determined = n_total - n_und

    if n_determined == 0:
        majority_phase = "undetermined"
        confidence = 0.0
    else:
        # argmax over the three determined categories only
        determined_items = [
            ("ferro",     counts["ferro"]),
            ("antiferro", counts["antiferro"]),
            ("disorder",  counts["disorder"]),
        ]
        # Stable argmax: ties broken by the listed order above.
        majority_phase, majority_count = max(
            determined_items, key=lambda kv: kv[1]
        )
        confidence = majority_count / n_determined

    return {
        "counts": counts,
        "n_total": n_total,
        "n_determined": n_determined,
        "majority_phase": majority_phase,
        "confidence": confidence,
        "undetermined_frac": n_und / n_total,
    }


# ----------------------------
# Seed-averaged (p, T) scan
# ----------------------------

def _summarize_counts(counts, n_total):
    """
    Build the per-point summary dict from a {phase: count} dictionary.
    Same contract as seed_average_at_point's return value.
    """
    n_und = counts["undetermined"]
    n_determined = n_total - n_und

    if n_determined == 0:
        majority_phase = "undetermined"
        confidence = 0.0
    else:
        determined_items = [
            ("ferro",     counts["ferro"]),
            ("antiferro", counts["antiferro"]),
            ("disorder",  counts["disorder"]),
        ]
        majority_phase, majority_count = max(
            determined_items, key=lambda kv: kv[1]
        )
        confidence = majority_count / n_determined

    return {
        "counts": counts,
        "n_total": n_total,
        "n_determined": n_determined,
        "majority_phase": majority_phase,
        "confidence": confidence,
        "undetermined_frac": n_und / n_total,
    }



# ----------------------------
# Confidence-aware plotting
# ----------------------------

# Phase color palette (kept consistent with the single-seed plot).
_PHASE_COLORS = {
    "ferro":        "tab:blue",
    "antiferro":    "tab:red",
    "disorder":     "tab:gray",
    "undetermined": "tab:orange",
}

_PHASE_LABELS = {
    "ferro":        "Ferromagnetic",
    "antiferro":    "Antiferromagnetic",
    "disorder":     "Disorder",
    "undetermined": "Undetermined (all seeds)",
}

# Color used for points whose confidence is below conf_min (ambiguous).
_AMBIGUOUS_COLOR = "lightgray"
_AMBIGUOUS_LABEL = "Ambiguous (low confidence)"


def _shade_color(base_color, confidence, conf_full, conf_min):
    """
    Map a confidence in [conf_min, conf_full] to an RGB color that is
    linearly interpolated between white (at conf_min) and the base color
    (at conf_full). Confidences >= conf_full clamp to the base color.
    """
    base_rgb = np.array(to_rgb(base_color))
    white = np.array([1.0, 1.0, 1.0])

    if conf_full <= conf_min:
        # Degenerate range; just return the base color.
        return tuple(base_rgb)

    if confidence >= conf_full:
        return tuple(base_rgb)
    if confidence <= conf_min:
        # By contract, we don't call _shade_color for sub-conf_min points;
        # but be defensive and return white-equivalent.
        return tuple(white)

    t = (confidence - conf_min) / (conf_full - conf_min)
    rgb = white + t * (base_rgb - white)
    return tuple(np.clip(rgb, 0.0, 1.0))
