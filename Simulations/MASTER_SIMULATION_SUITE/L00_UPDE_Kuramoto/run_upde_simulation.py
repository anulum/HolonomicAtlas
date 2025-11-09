import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. Simulation Parameters ---
N = 100 # Number of oscillators
K_MAX = 5.0 # Maximum coupling strength to test
K_STEPS = 20 # Number of coupling steps
T_FINAL = 100 # Time to simulate for each K
T_POINTS = 200 # Time points for simulation
NOISE_STRENGTH = 0.05 # Amplitude of noise eta(t)

# L15 Field Coupling (disabled by default)
C_FIELD_STRENGTH = 0.0 # Zeta_L * Psi_Global
C_FIELD_PHASE = 0.0 # Theta_Psi

print("--- SCPN Paper 0: UPDE/Kuramoto Simulation ---")
print(f"Simulating {N} oscillators to observe phase transition.")

# --- 2. Define Intrinsic Frequencies ---
# Use a Lorentzian (Cauchy) distribution for omega_i
np.random.seed(42)
omegas = np.random.standard_cauchy(N) * 0.5 # Centered at 0, scale 0.5

# --- 3. Define the UPDE (Kuramoto) Equations ---
#
def upde_model(theta, t, K, N, omegas, field_strength, field_phase):
    """
    Defines the system of ODEs for the UPDE.
    d(theta_i)/dt = omega_i + (K/N) * sum(sin(theta_j - theta_i)) + C_Field + eta
    """
    dtheta_dt = np.zeros(N)

    # Calculate phase differences
    theta_i, theta_j = np.meshgrid(theta, theta)
    phase_diffs = theta_j - theta_i

    # Intra-layer coupling (K_ij = K/N)
    intra_coupling = (K / N) * np.sum(np.sin(phase_diffs), axis=1)

    # Global Field Coupling (C_Field)
    field_coupling = field_strength * np.cos(theta - field_phase)

    # Noise term
    noise = np.random.randn(N) * NOISE_STRENGTH

    dtheta_dt = omegas + intra_coupling + field_coupling + noise
    return dtheta_dt

# --- 4. Run the Simulation ---
print("Running simulation across increasing K...")
K_values = np.linspace(0.0, K_MAX, K_STEPS)
final_R_values = []
initial_phases = np.random.uniform(0, 2 * np.pi, N)
t_span = np.linspace(0, T_FINAL, T_POINTS)

# Store initial and final states for plotting
phases_initial = initial_phases
phases_final = None

for K in K_values:
    # Solve the ODEs
    phases = odeint(
        upde_model,
        initial_phases,
        t_span,
        args=(K, N, omegas, C_FIELD_STRENGTH, C_FIELD_PHASE)
    )

    # Use the final phases as the initial condition for the next K
    initial_phases = phases[-1, :]

    # Calculate the order parameter R for the final 10% of the time
    r_values = []
    for t_step in range(int(T_POINTS * 0.9), T_POINTS):
        r = (1.0 / N) * np.abs(np.sum(np.exp(1j * phases[t_step, :])))
        r_values.append(r)

    final_R_values.append(np.mean(r_values))

    if K == K_values[-1]: # Save the final state
        phases_final = phases[-1, :]

print("Simulation complete.")

# --- 5. Plot Results ---
plt.figure(figsize=(18, 6))

# Subplot 1: Initial State (Incoherent)
ax1 = plt.subplot(1, 3, 1, polar=True)
ax1.plot(phases_initial, np.ones(N), 'o', label='Oscillators')
ax1.set_title(f'Initial State (K=0)\nR = {final_R_values[0]:.3f}')
ax1.set_rticks([])

# Subplot 2: Final State (Coherent)
ax2 = plt.subplot(1, 3, 2, polar=True)
ax2.plot(phases_final, np.ones(N), 'o', label='Oscillators')
ax2.set_title(f'Final State (K={K_MAX})\nR = {final_R_values[-1]:.3f}')
ax2.set_rticks([])

# Subplot 3: Phase Transition
ax3 = plt.subplot(1, 3, 3)
ax3.plot(K_values, final_R_values, 'o-')
ax3.set_title('UPDE Phase Transition')
ax3.set_xlabel('Coupling Strength (K)')
ax3.set_ylabel('Order Parameter (R)')
ax3.set_ylim(0, 1.05)
ax3.grid(True)

plt.tight_layout()
filename = 'L00_UPDE_Phase_Transition.png'
plt.savefig(filename)
print(f"Saved plot: {filename}")
plt.close()
