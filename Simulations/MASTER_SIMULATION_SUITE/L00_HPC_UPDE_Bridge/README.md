# SCPN Simulation Suite: L00_HPC_UPDE_Bridge
# Foundational Dynamics: HPC-UPDE Mathematical Bridge Validation

## 1. Objective
This script is not a dynamic simulation but a **computational proof**.

Its purpose is to validate the mathematical derivation from *Paper 0* which claims that the Unified Phase Dynamics Equation (UPDE) is identical to a gradient descent on the Variational Free Energy ($F$) potential.

- **Hypothesis:** `d(theta)/dt` from the **UPDE (Kuramoto) equation** is mathematically identical to `d(theta)/dt` from **-grad(F)**.

## 2. Simulation Logic
The script `run_hpc_upde_proof.py` performs a direct, one-to-one comparison:
1. **Initializes State:** Creates a random phase vector $\theta$ for $N$ oscillators.
2. **Defines Functions:**
* `calculate_free_energy_potential(theta, K)`: Calculates the scalar $F = -\sum K_{ij} \cos(\theta_j - \theta_i)$.
* `calculate_F_gradient(theta, K)`: *Analytically* calculates the gradient vector $\nabla F$ (i.e., $[\frac{\partial F}{\partial \theta_1}, \frac{\partial F}{\partial \theta_2}, ...]$).
* `calculate_UPDE_dynamics(theta, K, omegas)`: *Analytically* calculates the dynamics vector from the UPDE formula $d\theta_i/dt = \omega_i + \sum_j K \sin(\theta_j - \theta_i)$.
3. **Runs Proof:**
* It computes the dynamics vector `dtheta_from_F = -calculate_F_gradient(...)`.
* It computes the dynamics vector `dtheta_from_UPDE = calculate_UPDE_dynamics(...)` (with $\omega=0$ for a pure comparison).
4. **Asserts Equivalence:** It uses `numpy.allclose()` to assert that the two vectors are numerically identical within machine precision.

## 3. How to Run
```bash
# Install dependencies
pip install numpy

# Run the validation proof
python run_hpc_upde_proof.py
4. Expected Output
The script will print the two calculated vectors and then a final "VERDICT."

If successful: It will print VERDICT: CONFIRMED. The UPDE is a gradient descent on Free Energy.

If it fails: It will raise an AssertionError, indicating the mathematical bridge is broken.