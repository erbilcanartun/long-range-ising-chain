# Long-Range Ising Chain: Renormalization Group Solutions

This repository contains numerical data from renormalization group (RG) calculations for the one-dimensional Ising chain with long-range interactions. The project studies the critical behavior and phase transitions of the model, where spin interactions decay as a power law, $J(r) \sim 1/r^{1+\sigma}$, with $r$ being the distance between spins and $\sigma$ controlling the interaction range.

## Overview

The Ising chain is a fundamental model in statistical physics for studying magnetism and phase transitions. Unlike the standard nearest-neighbor Ising model, which lacks a finite-temperature phase transition in one dimension, long-range interactions can induce criticality. This repository provides datasets from RG analysis, including critical interaction strengths, critical exponents, and magnetization profiles, to explore how the decay exponent $\sigma$ influences the system's universality class and thermodynamic properties.

The data is intended for researchers in condensed matter physics, statistical mechanics, or related fields to analyze critical phenomena or validate theoretical predictions against numerical results.

## Repository Contents

The repository includes the following files:

- **data/critical_interaction_data.npy**: A NumPy array containing critical interaction strengths (e.g., coupling constants $J_c$) at which the system undergoes a phase transition, computed for various values of the interaction decay exponent $\sigma$.
- **data/exponents.csv**: A CSV file tabulating critical exponents (e.g., $\beta$ for magnetization, $\gamma$ for susceptibility, $\nu$ for correlation length) as functions of $\sigma$ or other model parameters. These characterize the universality class of the phase transition.
- **data/magnetization.npz**: A compressed NumPy archive storing magnetization data, likely spontaneous magnetization curves below the critical temperature for different $\sigma$ values or system sizes.

## Usage

### Prerequisites
To work with the data files, you need Python with the following libraries:
- `numpy` for loading `.npy` and `.npz` files.
- `pandas` for reading and analyzing the `.csv` file.
- Optionally, `matplotlib` or `seaborn` for visualization.

Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn
```

### Loading Data
Example Python code to load and inspect the data:

```python
import numpy as np
import pandas as pd

# Load critical interaction data
critical_data = np.load('data/critical_interaction_data.npy')
print("Critical interaction strengths:", critical_data)

# Load critical exponents
exponents = pd.read_csv('data/exponents.csv')
print("Critical exponents:\n", exponents)

# Load magnetization data
magnetization = np.load('data/magnetization.npz')
for key in magnetization:
    print(f"Magnetization array {key}:", magnetization[key])
```

### Notes
- The `.npy` and `.npz` files are binary NumPy arrays, optimized for efficient storage and loading.
- The `.csv` file is human-readable and suitable for quick inspection or import into tools like Excel.
- The data assumes familiarity with the Ising model and RG methods. For theoretical background, refer to standard texts on statistical mechanics or relevant publications.

## Methodology

The data was generated using numerical RG techniques applied to the one-dimensional Ising chain with long-range interactions. The RG approach iteratively coarse-grains the system to identify fixed points and compute critical properties, such as the critical coupling $J_c$ and exponents $\beta$, $\gamma$, and $\nu$. The interaction strength follows $J(r) \sim 1/r^{1+\sigma}$, where $\sigma$ determines the range of interactions:
- Small $\sigma$: Long-range regime, approaching mean-field behavior.
- Large $\sigma$: Short-range limit, resembling the nearest-neighbor Ising model.

The datasets cover a range of $\sigma$ values to map the transition from classical to non-classical critical behavior.

## Limitations

- This repository contains only output data, not the source code used to generate it. For reproducibility, contact the repository owner or refer to the associated publication (if available).
- No visualizations or analysis scripts are included. Users are encouraged to create their own plots (e.g., critical exponents vs. $\sigma$) using the provided data.

## License

This project is licensed under the terms specified in the `LICENSE` file.

## Contact

For questions or collaboration inquiries, please contact the repository owner via GitHub or refer to the associated research publication for further details.