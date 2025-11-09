import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
N_SAMPLES = 5000       # Total time-points for the session
TRUE_LAG = 15          # The true time lag (tau) for the CEF effect
EVENT_TIME = 2500      # Center of the coherence event
EVENT_WIDTH = 50       # Half-width of the event
EVENT_KERNEL_LEN = (EVENT_WIDTH * 2) + 1 # 101 (Odd length for true center)

SIGNAL_NOISE = 1.0     # Noise level for C(t)
DETECTOR_NOISE = 1.5   # Noise level for D(t)
SMOOTHING_WINDOW = 11  # Must be ODD for zero-phase shift

print("--- SCPN Paper 0: Prediction II (QRNG-CEF) Analysis Simulation (v4.0 - Corrected Injection) ---")
print(f"Simulating {N_SAMPLES} time-points...")
print(f"Injecting signature: Peak in C(t) at {EVENT_TIME} causes peak in D(t) at {EVENT_TIME + TRUE_LAG}\n")

# --- 2. Generate Mock Time-Series Data ---

# Generate baseline noise for both signals
np.random.seed(42)
c_noise = np.random.randn(N_SAMPLES) * SIGNAL_NOISE
d_noise = np.random.randn(N_SAMPLES) * DETECTOR_NOISE

# Create C(t): Collective Coherence Metric
C_t = c_noise
# Create a symmetric Gaussian kernel with an ODD length (101)
x_kernel = np.arange(EVENT_KERNEL_LEN)
event_kernel = np.exp(-0.5 * ((x_kernel - EVENT_WIDTH) / (EVENT_WIDTH / 2))**2)

# *** BUG FIX: Correct the slice indices to be 101 elements long ***
# Center index = 2500, half-width = 50.
# Slice should be [2500 - 50] to [2500 + 50], inclusive.
# This is indices 2450 to 2550.
# The slice C_t[2450:2551] has length 101.
start_idx_c = EVENT_TIME - EVENT_WIDTH
end_idx_c = start_idx_c + EVENT_KERNEL_LEN # 2450 + 101 = 2551
C_t[start_idx_c : end_idx_c] += event_kernel * 5.0 # Strong signal

# Create D(t): Randomness Deviation Metric
D_t = d_noise
# Add the correlated "deviation event" (weaker peak, lagged in time)
start_idx_d = (EVENT_TIME + TRUE_LAG) - EVENT_WIDTH
end_idx_d = start_idx_d + EVENT_KERNEL_LEN # (2515 - 50) + 101 = 2566
D_t[start_idx_d : end_idx_d] += event_kernel * 2.5 # Weaker signal

# --- 3. Analysis Pipeline ---
print("[Analysis Pipeline Running...]")

# Use np.convolve with an odd window for smoothing
def smooth_numpy(y, box_pts):
    """Applies a centered, zero-phase-shift moving average using numpy."""
    if box_pts % 2 == 0:
        raise ValueError("Smoothing window must be an odd number.")
    box = np.ones(box_pts)/box_pts
    # 'same' mode with an ODD kernel is correctly centered
    return np.convolve(y, box, mode='same')

C_t_smooth = smooth_numpy(C_t, SMOOTHING_WINDOW)
D_t_smooth = smooth_numpy(D_t, SMOOTHING_WINDOW)

# Calculate cross-correlation
C_norm = (C_t_smooth - np.mean(C_t_smooth)) / np.std(C_t_smooth)
D_norm = (D_t_smooth - np.mean(D_t_smooth)) / np.std(D_t_smooth)

# Use np.correlate and compute lags manually
correlation = np.correlate(D_norm, C_norm, mode='full')

# Manually compute the lags for 'full' mode
# Output length is N + M - 1.
# The "zero" lag (index 0) is at index (N - 1)
lags = np.arange(len(correlation)) - (N_SAMPLES - 1)

# Find the max POSITIVE correlation
detected_lag_index = np.argmax(correlation)
detected_lag = lags[detected_lag_index]
max_corr = correlation[detected_lag_index]

print(f"Analysis complete.")
print(f"  -> Max positive correlation found: {max_corr:.4f}")
print(f"  -> Detected at lag (tau): {detected_lag}")

# --- 4. Falsification Verdict ---
print("\n--- 5. Falsification Verdict ---")
if detected_lag == TRUE_LAG:
    print("VERDICT: \033[1mCONFIRMED\033[0m")
    print(f"The analysis pipeline successfully detected the correlation at the true lag of {TRUE_LAG}.")
else:
    print("VERDICT: \033[1mFALSIFIED (Analysis Failed)\033[0m")
    print(f"The pipeline failed. Detected lag {detected_lag} does not match true lag {TRUE_LAG}.")

# --- 5. Plotting ---
plt.figure(figsize=(15, 10))

# Plot C(t)
plt.subplot(3, 1, 1)
plt.plot(C_t_smooth, label='C(t) - Collective Coherence', color='blue')
plt.title(f'Simulated Time-Series Data (v4.0 Fix)')
plt.legend()

# Plot D(t)
plt.subplot(3, 1, 2)
plt.plot(D_t_smooth, label='D(t) - QRNG Deviation', color='red')
plt.legend()

# Plot Cross-Correlation
plt.subplot(3, 1, 3)
plt.plot(lags, correlation, label='Cross-Correlation X_CD(tau)')
plt.axvline(detected_lag, color='red', linestyle='--', label=f'Detected Lag = {detected_lag}')
plt.axvline(TRUE_LAG, color='green', linestyle=':', label=f'True Lag = {TRUE_LAG}')
plt.title('Cross-Correlation Analysis (NumPy correlate)')
plt.xlabel('Lag (tau)')
plt.ylabel('Correlation Coefficient')
plt.xlim(-EVENT_WIDTH, EVENT_WIDTH * 2) # Zoom in
plt.legend()

plt.tight_layout()
filename = 'L15_CEF_QRNG_CrossCorrelation_v4.0.png'
plt.savefig(filename)
print(f"\nSaved analysis plot: {filename}")
plt.close()