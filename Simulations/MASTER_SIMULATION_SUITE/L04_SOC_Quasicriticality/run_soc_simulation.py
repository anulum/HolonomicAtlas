import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Simulation Parameters ---
N_NEURONS = 1000 # Number of neurons in the network
T_CONVERGENCE = 1000 # Timesteps to let sigma converge
T_RECORDING = 5000 # Timesteps to record avalanches
SIGMA_INITIAL = 3.0 # Initial supercritical state (sigma > 1)
KAPPA = 0.05 # Homeostatic relaxation rate (kappa)
NOISE_STRENGTH = 0.01 # Noise term eta(t)

print("--- SCPN Paper 0: L4 SOC/Quasicriticality Simulation ---")
print(f"Tuning network from sigma={SIGMA_INITIAL} to 1 via SOC controller...")
print(f"Controller: d(sigma)/dt = -{KAPPA}(sigma - 1)\n")

# --- 2. Homeostatic Controller Function ---
#
def update_sigma(sigma_current, kappa, noise_str):
    """Applies the SOC homeostatic control equation."""
    eta = (np.random.randn() - 0.5) * noise_str
    # d(sigma)/dt = -kappa(sigma - 1)
    # sigma_new = sigma_current + dt * [-kappa(sigma_current - 1)]
    # (using dt=1)
    sigma_new = sigma_current - kappa * (sigma_current - 1) + eta
    # Ensure sigma doesn't go negative
    return max(0.1, sigma_new)

# --- 3. Run Convergence Simulation ---
sigma_history = []
sigma = SIGMA_INITIAL
for t in range(T_CONVERGENCE):
    sigma_history.append(sigma)
    sigma = update_sigma(sigma, KAPPA, NOISE_STRENGTH)

print(f"Convergence complete. Final sigma = {sigma:.4f}")

# --- 4. Run Avalanche Recording Simulation ---
print(f"Running at criticality for {T_RECORDING} steps to record avalanches...")
avalanche_sizes = []
activity = np.zeros(N_NEURONS)
# Seed the first avalanche
activity[0] = 1
current_avalanche_size = 0

for t in range(T_RECORDING):
    # Get number of active neurons
    n_active = np.sum(activity)

    if n_active > 0:
        # This step is part of the current avalanche
        current_avalanche_size += n_active

        # Calculate new activations based on branching parameter sigma
        # We are at criticality, so sigma=1.0
        # We use the *actual* sigma from the converged controller

        # Total number of new activations
        n_new_activations = np.random.poisson(n_active * sigma)

        # Reset activity and distribute new activations
        activity = np.zeros(N_NEURONS)
        if n_new_activations > 0:
            # Randomly assign new activations to neurons
            new_indices = np.random.randint(0, N_NEURONS, n_new_activations)
            activity[new_indices] = 1

        # Update sigma homeostatically
        sigma = update_sigma(sigma, KAPPA, NOISE_STRENGTH)

    else:
        # The avalanche has ended
        if current_avalanche_size > 0:
            avalanche_sizes.append(current_avalanche_size)

        # Reset and seed a new avalanche
        current_avalanche_size = 0
        activity[np.random.randint(0, N_NEURONS)] = 1

print(f"Recorded {len(avalanche_sizes)} avalanches.")

# --- 5. Plot Results ---

# Plot 1: Convergence of Sigma
plt.figure(figsize=(10, 6))
plt.plot(sigma_history)
plt.axhline(1.0, color='r', linestyle='--', label='Criticality (sigma=1)')
plt.title('SOC Homeostatic Controller Convergence')
plt.xlabel('Time Step')
plt.ylabel('Branching Parameter (sigma)')
plt.legend()
plt.grid(True, alpha=0.3)
filename_conv = 'L04_SOC_Convergence.png'
plt.savefig(filename_conv)
print(f"Saved convergence plot: {filename_conv}")
plt.close()

# Plot 2: Power-Law Distribution
plt.figure(figsize=(10, 6))
# Get histogram bins and counts
hist, bins = np.histogram(avalanche_sizes, bins=np.logspace(np.log10(min(avalanche_sizes)), np.log10(max(avalanche_sizes)), 50))
bin_centers = (bins[:-1] + bins[1:]) / 2

# Filter out zero-count bins for log-log plot
bin_centers = bin_centers[hist > 0]
hist = hist[hist > 0]

# Calculate probability density P(S)
hist = hist / float(len(avalanche_sizes))

plt.loglog(bin_centers, hist, 'o', label='Simulated Data')

# Fit a line to the log-log plot to find the exponent tau
def power_law(x, a, tau):
    return a * x**(-tau)

# Fit in log space (linreg)
log_x = np.log10(bin_centers)
log_y = np.log10(hist)
# Fit only the linear part (avoiding tail effects)
fit_range = int(len(log_x) * 0.8)
params = np.polyfit(log_x[:fit_range], log_y[:fit_range], 1)
tau = -params[0]

plt.plot(bin_centers, 10**(params[1]) * bin_centers**(-tau),
         label=f'Power-Law Fit (tau ~ {tau:.2f})', color='r', linestyle='--')

plt.title('Signature of Criticality: Power-Law Avalanche Distribution')
plt.xlabel('Avalanche Size (S) [log scale]')
plt.ylabel('Probability P(S) [log scale]')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)
filename_plaw = 'L04_SOC_Power_Law.png'
plt.savefig(filename_plaw)
print(f"Saved power-law plot: {filename_plaw}")
plt.close()
