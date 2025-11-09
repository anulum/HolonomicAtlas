import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- 1. Simulation Parameters ---
N_SYSTEMS = 500
KB = 1.0 # Set Boltzmann constant to 1 for simplicity

# SEC Weights
W_K = 1.0 # Complexity
W_C = 1.0 # Coherence
W_Q = 1.0 # Qualia

np.random.seed(42)

print("--- SCPN Paper 0: L15 SEC == Causal Path Entropy (S_C) Simulation ---")
print("Validating that SEC and S_C are monotonically correlated.")

# --- 2. Define Causal Path Functions ---
#

def N_states(K):
    """Number of states (N_states) as a function of Complexity (K).

    Args:
        K (float): The complexity of the system.

    Returns:
        float: The number of states.
    """
    # Assumed exponential relationship
    return np.exp(K * 2.0)

def f_acc(C):
    """Fraction of accessible trajectories (f_acc) as a function of Coherence (C).

    This is a Gaussian peaking at the "quasicritical" point (e.g., C=0.8).

    Args:
        C (float): The coherence of the system.

    Returns:
        float: The fraction of accessible trajectories.
    """
    COHERENCE_PEAK = 0.8
    COHERENCE_WIDTH = 0.3
    return np.exp(-((C - COHERENCE_PEAK)**2) / (2 * COHERENCE_WIDTH**2))

def D_paths(Q):
    """Diversity of paths (D_paths) as a function of Qualia Capacity (Q).

    Args:
        Q (float): The qualia capacity of the system.

    Returns:
        float: The diversity of paths.
    """
    # Assumed exponential relationship
    return np.exp(Q * 1.5)

# --- 3. Run Simulation ---
print(f"Generating {N_SYSTEMS} mock systems...")

# Generate random systems
K_values = np.random.uniform(0.1, 1.0, N_SYSTEMS)
C_values = np.random.uniform(0.0, 1.0, N_SYSTEMS)
Q_values = np.random.uniform(0.1, 1.0, N_SYSTEMS)

SEC_scores = []
S_C_scores = []

for K, C, Q in zip(K_values, C_values, Q_values):
    # --- Calculate SEC (The Teleological Metric) ---
    # SEC = w_C*C + w_K*K + w_Q*Q
    sec = (W_K * K) + (W_C * C) + (W_Q * Q)
    SEC_scores.append(sec)

    # --- Calculate S_C (The Physical Metric) ---
    # W_paths = N_states(K) * f_acc(C) * D_paths(Q)
    W_paths = N_states(K) * f_acc(C) * D_paths(Q)

    # S_C = k_B * log(W_paths)
    S_C = KB * np.log(W_paths + 1e-9) # add epsilon to avoid log(0)
    S_C_scores.append(S_C)

print("Calculation complete.")

# --- 4. Analysis & Plotting ---
correlation, p_value = spearmanr(SEC_scores, S_C_scores)

print(f"\n--- Analysis Results ---")
print(f"Spearman Rank Correlation (rho): {correlation:.6f}")
print(f"p-value: {p_value:.6e}")

print("\n--- 5. Falsification Verdict ---")
if correlation > 0.9 and p_value < 0.001:
    print("VERDICT: \033[1mCONFIRMED\033[0m")
    print("SEC and S_C are shown to be strongly and monotonically correlated.")
    print("The teleological axiom is equivalent to the physical imperative.")
else:
    print("VERDICT: \033[1mFALSIFIED\033[0m")
    print("No significant monotonic correlation was found.")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(SEC_scores, S_C_scores, alpha=0.5, label=f"Systems (N={N_SYSTEMS})")
plt.title('Validation: SEC vs. Causal Path Entropy ($S_C$)')
plt.xlabel('Sustainable Ethical Coherence (SEC) Score [Teleological Metric]')
plt.ylabel('Causal Path Entropy ($S_C$) [Physical Metric, log scale]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.9, f'Spearman $\\rho$ = {correlation:.4f}',
    transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

filename = 'L15_SEC_vs_SC_Equivalence.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()
