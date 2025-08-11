from mpmath import mp

def mp_logsumexp(values):
    if not values:
        return mp.ninf
    max_val = max(values)
    sum_exp = mp.mpf(0)
    for v in values:
        sum_exp += mp.exp(v - max_val)
    return max_val + mp.log(sum_exp)


import os
import pandas as pd

# Function to save results to a CSV file
def save_exponents_csv(n_values, Jcs, nus, alphas, etas, deltas, filename="results/exponents.csv"):
    """
    Save critical exponents to a CSV file.
    
    Args:
        n_values (list): List of interaction exponents n.
        Jcs (list): List of critical coupling strengths J_c.
        nus (list): List of correlation length exponents ν.
        alphas (list): List of specific heat exponents α.
        etas (list): List of correlation function exponents η.
        deltas (list): List of magnetization exponents δ.
        filename (str): Name of the file to save results (default: 'results/exponents.csv').
    """
    data = {
        "n": list(map(float, n_values)),
        "Jc": list(map(float, Jcs)),
        "nu": list(map(float, nus)),
        "alpha": list(map(float, alphas)),
        "eta": list(map(float, etas)),
        "delta": list(map(float, deltas))
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Function to load results from a CSV file
def load_exponents_csv(filename="results/exponents.csv"):
    """
    Load critical exponents from a CSV file.
    
    Args:
        filename (str): Name of the file to load results from (default: 'results/exponents.csv').
        
    Returns:
        tuple: (n_values, Jcs, nus, alphas, etas, deltas) as numpy arrays, or None if error.
    """
    try:
        df = pd.read_csv(filename)
        return (
            df["n"].to_numpy(),
            df["Jc"].to_numpy(),
            df["nu"].to_numpy(),
            df["alpha"].to_numpy(),
            df["eta"].to_numpy(),
            df["delta"].to_numpy()
        )
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except KeyError as e:
        print(f"Error: Missing column {e} in {filename}.")
        return None
