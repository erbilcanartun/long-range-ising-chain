import math
import matplotlib.pyplot as plt


plus_configs = [
    ([1,1,1], lambda J1, J2: 2*J1 + J2),
    ([1,1,-1], lambda J1, J2: -J2),
    ([1,-1,1], lambda J1, J2: -2*J1 + J2),
    ([-1,1,1], lambda J1, J2: -J2)
]

minus_configs = [
    ([-1,-1,-1], lambda J1, J2: 2*J1 + J2),
    ([-1,-1,1], lambda J1, J2: -J2),
    ([-1,1,-1], lambda J1, J2: -2*J1 + J2),
    ([1,-1,-1], lambda J1, J2: -J2)
]

def get_J(d, J0, n):
    if d <= 0:
        return 0.0
    return J0 / d**n

def compute_J_prime(start, J0, n):
    J1 = get_J(1, J0, n)
    J2 = get_J(2, J0, n)
    distances = []
    for iL in range(3):
        for iR in range(3):
            d = (start + iR) - (1 + iL)
            distances.append((iL, iR, d))

    def compute_R(is_pp):
        confsR = plus_configs if is_pp else minus_configs
        R = 0.0
        for spinsL, El_func in plus_configs:
            El = El_func(J1, J2)
            for spinsR, Er_func in confsR:
                Er = Er_func(J1, J2)
                Eint = 0.0
                for iL, iR, d in distances:
                    sign = spinsL[iL] * spinsR[iR]
                    Eint += sign * get_J(d, J0, n)
                total = El + Er + Eint
                R += math.exp(total)
        return R

    R_pp = compute_R(True)
    R_pm = compute_R(False)
    if R_pm == 0:
        return float('inf')
    return 0.5 * math.log(R_pp / R_pm)

def compute_J_prime_func(start, J_func):
    J1 = J_func(1)
    J2 = J_func(2)
    distances = []
    for iL in range(3):
        for iR in range(3):
            d = (start + iR) - (1 + iL)
            distances.append((iL, iR, d))
    
    def compute_R(is_pp):
        confsR = plus_configs if is_pp else minus_configs
        R = 0.0
        for spinsL, El_func in plus_configs:
            El = El_func(J1, J2)
            for spinsR, Er_func in confsR:
                Er = Er_func(J1, J2)
                Eint = 0.0
                for iL_idx, iR_idx, d in distances:
                    sign = spinsL[iL_idx] * spinsR[iR_idx]
                    Eint += sign * J_func(d)
                total = El + Er + Eint
                R += math.exp(total)
        return R
    
    R_pp = compute_R(True)
    R_pm = compute_R(False)
    if R_pm == 0:
        return float('inf')
    return 0.5 * math.log(R_pp / R_pm)


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
    plt.plot(ks, Jps, marker='o')
    plt.xlabel('Cluster number k')
    plt.ylabel("Renormalized bond J'")
    plt.title(f"J' vs k for J0={J0}, n={n}")
    plt.grid(True)
    plt.show()

def plot_non_renormalized_J(J0, n, max_r=30):
    rs = list(range(1, max_r + 1))
    Js = [get_J(r, J0, n) for r in rs]
    
    plt.figure()
    plt.plot(rs, Js, marker='o')
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
    plt.plot(rs, Js, label='Non-renormalized J(r)', marker='o', linestyle='-')
    plt.plot(effective_rs, Jps, label='Renormalized J\'(r)', marker='x', linestyle='--')
    plt.xlabel('Distance r')
    plt.ylabel('J')
    plt.title(f"Non-renormalized and Renormalized J vs r for J0={J0}, n={n}")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_rg_flow(J0, n, max_k, num_steps, show_first=0, num_r_to_plot=5):
    J_func = lambda d: J0 / d**n if d > 0 else 0.0
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
        J_func = lambda d: J_dict.get(d, 0.0) if d > 0 else 0.0
    
    # Plot J vs RG step for first num_r_to_plot r
    plt.figure()
    steps = range(len(all_Js))
    for r_idx in range(min(num_r_to_plot, max_k)):
        Js_step = [all_Js[s][r_idx] for s in steps]
        plt.plot(steps, Js_step, label=f'r={r_idx+1}', marker='o')
    plt.xlabel('RG Step')
    plt.ylabel('J(r)')
    plt.title(f'RG Flow: J(r) vs Step for J0={J0}, n={n}')
    plt.xticks(steps)
    plt.legend()
    plt.grid(True)
    plt.show()