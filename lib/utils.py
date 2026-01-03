import numpy as np
import pandas as pd
from numba import njit
import os


@njit(cache=True)
def required_initial_max_distance(max_dist_final, n_steps):
    D = max_dist_final
    for _ in range(n_steps):
        D = 3 * D + 2
    return D


@njit(cache=True)
def logsumexp(values):
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m)))


def build_J(J0, a, D):
    J = np.zeros(D + 1)
    r = np.arange(1, D + 1)
    J[1:] = J0 / (r**a)
    return J


# Function to save results to a CSV file
def save_exponents_csv(n_values, Jcs, nus, alphas, etas, deltas, betas, gammas, filename="../data/exponents.csv"):
    """
    Save critical exponents and critical couplings to a CSV file.
    
    Parameters:
    n_values (array): Array of power-law exponents (a in the paper)
    Jcs (array): Array of critical couplings
    nus, alphas, etas, deltas, betas, gammas (arrays): Critical exponents
    filename (str): Path to save the CSV file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create DataFrame
    data = {
        'n': n_values,
        'Jc': Jcs,
        'nu': nus,
        'alpha': alphas,
        'eta': etas,
        'delta': deltas,
        'beta': betas,
        'gamma': gammas
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Exponents saved to {filename}")


# Function to load results from a CSV file
def load_exponents_csv(filename="../data/exponents.csv"):
    """
    Load critical exponents from a CSV file.
    
    Parameters:
    filename (str): Path to the CSV file
    
    Returns:
    tuple: Arrays of n_values, Jcs, nus, alphas, etas, deltas, betas, gammas
    """
    df = pd.read_csv(filename)
    return (df['n'].values, df['Jc'].values, df['nu'].values, df['alpha'].values,
            df['eta'].values, df['delta'].values, df['beta'].values, df['gamma'].values)
