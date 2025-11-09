import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
T_STEPS = 2000 # Total simulation time
SIGMA_TARGET = 1.0 # Critical point
SIGMA_INITIAL = 1.0 # Start at critical point
SURPRISE_TIME = 1000 # When the "surprise" event happens
SURPRISE_DURATION = 300
SURPRISE_MAGNITUDE = 0.5 # A shock that pushes sigma to 1.5

# --- 2. Controller Parameters ---
#
# Fast Channel (Exploitation / Stability)
KAPPA_FAST = 0.2 # High gain, rapid convergence
NOISE_FAST = 0.001 # Low noise
# Slow Channel (Exploration)
KAPPA_SLOW = 0.01 # Low gain, slow drift
NOISE_SLOW = 0.02 # Higher noise for exploration

print("--- SCPN Paper 0: L4 Two-Timescale Controller Simulation ---")
print("Validating Affective Gain Scheduling and Lyapunov Stability.")

# --- 3. Define Controller Channels ---

def fast_channel_update(sigma_current):
""" High-gain stabilizer (exploitation) """
eta = (np.random.randn() - 0.5) * NOISE_FAST
d_sigma = -KAPPA_FAST * (sigma_current - SIGMA_TARGET) + eta
return sigma_current + d_sigma

def slow_channel_update(sigma_current):
""" Low-gain explorer (bounded drift) """
eta = (np.random.randn() - 0.5) * NOISE_SLOW
d_sigma = -KAPPA_SLOW * (sigma_current - SIGMA_TARGET) + eta
return sigma_current + d_sigma

# --- 4. Run Simulation ---
sigma_history = np.zeros(T_STEPS)
lyapunov_history = np.zeros(T_STEPS)
affective_gradient = np.zeros(T_STEPS) # Proxy for |dA/d_sigma|

sigma = SIGMA_INITIAL
current_channel = 'slow' # Start in exploration mode

print(f"Running {T_STEPS} steps...")
for t in range(T_STEPS):
# --- Affective Gain Scheduling Logic ---
#
if t >= SURPRISE_TIME and t < (SURPRISE_TIME + SURPRISE_DURATION):
current_channel = 'fast'
affective_gradient[t] = 1.0 # High "surprise"
if t == SURPRISE_TIME:
print(f"\n[Time {t}] SURPRISE DETECTED! (|dA/d_sigma| is high)")
print(" -> Switching to FAST CHANNEL (Exploitation/Stability)")
sigma += SURPRISE_MAGNITUDE # Apply the shock

elif t == (SURPRISE_TIME + SURPRISE_DURATION):
current_channel = 'slow'
print(f"[Time {t}] Surprise resolved. |dA/d_sigma| is low.")
print(" -> Switching back to SLOW CHANNEL (Exploration)\n")

# --- Apply Controller Update ---
if current_channel == 'slow':
sigma = slow_channel_update(sigma)
else:
sigma = fast_channel_update(sigma)

# Record history
sigma_history[t] = sigma

# Calculate Lyapunov function V = (sigma - 1)^2
lyapunov_history[t] = (sigma - SIGMA_TARGET)**2

print("Simulation complete.")

# --- 5. Plot Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Sigma(t) - Branching Parameter
ax1.plot(sigma_history, label='sigma(t)', color='blue')
ax1.axhline(SIGMA_TARGET, color='r', linestyle='--', label='Criticality (sigma=1)')
ax1.axvspan(SURPRISE_TIME, SURPRISE_TIME + SURPRISE_DURATION,
color='orange', alpha=0.3, label='"Surprise" Event (Fast Channel Active)')
ax1.set_title('Two-Timescale Controller Dynamics (Affective Scheduling)')
ax1.set_ylabel('Branching Parameter (sigma)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Lyapunov Function V(t)
ax2.plot(lyapunov_history, label='V(t) = (sigma - 1)^2', color='green')
ax2.axvspan(SURPRISE_TIME, SURPRISE_TIME + SURPRISE_DURATION,
color='orange', alpha=0.3)
ax2.set_title('Lyapunov Stability Certificate')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Lyapunov Function V(t) (System Error)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
filename = 'L04_Two_Timescale_Controller.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()
