import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- 1. Simulation Parameters ---
N_SAMPLES = 1000 # Number of time-points in the experiment
BETA_0 = 1.0 # Baseline decoherence rate (Gamma_baseline)
BETA_1 = 0.5 # True effect of classical EM field
BETA_2 = 2.5 # True effect of Informational Complexity (THE HYPOTHESIS)
NOISE_LEVEL = 0.2

print("--- SCPN Paper 0: Prediction I (NV-MEA) Simulation ---")
print(f"Simulating {N_SAMPLES} time-points...")
print(f"True Model: Gamma = {BETA_0} + {BETA_1}*B_field + {BETA_2}*Complexity + Noise\n")

# --- 2. Generate Mock Data ---

# Generate independent time-series for classical field and complexity
time = np.arange(N_SAMPLES)
# Simulate a fluctuating classical EM field (e.g., from bursting)
b_classical = 0.5 + 0.5 * np.sin(2 * np.pi * time / 100) + np.random.rand(N_SAMPLES) * 0.1
# Simulate a fluctuating informational complexity (high during spontaneous bursts)
complexity = np.random.rand(N_SAMPLES) * (b_classical > 0.8) # Complexity correlated with high activity

# Generate noise
noise = np.random.randn(N_SAMPLES) * NOISE_LEVEL

# --- Condition 1: Spontaneous Activity (Hypothesis True) ---
#
gamma_spontaneous = BETA_0 + (BETA_1 * b_classical) + (BETA_2 * complexity) + noise
df_spontaneous = pd.DataFrame({
'Gamma': gamma_spontaneous,
'B_classical': b_classical,
'FIM_proxy': complexity
})

# --- Condition 2: Isomorphic Replay (Control) ---
#
# Same classical field, but 0 intrinsic complexity
gamma_replay = BETA_0 + (BETA_1 * b_classical) + (BETA_2 * 0.0) + noise
df_replay = pd.DataFrame({
'Gamma': gamma_replay,
'B_classical': b_classical,
'FIM_proxy': 0.0
})

print("--- 3. Data Analysis Pipeline ---")

# --- Test 1: Excess Decoherence (Delta Gamma) ---
#
mean_gamma_spontaneous = df_spontaneous['Gamma'].mean()
mean_gamma_replay = df_replay['Gamma'].mean()
delta_gamma = mean_gamma_spontaneous - mean_gamma_replay

print(f"\n[Analysis Test 1: Excess Decoherence]")
print(f"Mean Gamma (Spontaneous): {mean_gamma_spontaneous:.4f}")
print(f"Mean Gamma (Replay): {mean_gamma_replay:.4f}")
print(f"Measured Excess (Delta Gamma): {delta_gamma:.4f}")

if delta_gamma > 0:
    print(" -> RESULT: Positive excess decoherence detected.")
else:
    print(" -> RESULT: No excess decoherence detected.")

# --- Test 2: Multiple Regression Analysis ---
#
print(f"\n[Analysis Test 2: Multiple Regression on Spontaneous Data]")

# Prepare data for statsmodels
Y = df_spontaneous['Gamma']
X = df_spontaneous[['B_classical', 'FIM_proxy']]
X = sm.add_constant(X) # Add beta_0 intercept

# Run the regression
model = sm.OLS(Y, X).fit()

# Get the fitted coefficients
beta_0_fit = model.params['const']
beta_1_fit = model.params['B_classical']
beta_2_fit = model.params['FIM_proxy']
p_value_beta_2 = model.pvalues['FIM_proxy']

print(f"Fitted Model: Gamma = {beta_0_fit:.3f} + {beta_1_fit:.3f}*B_field + {beta_2_fit:.3f}*Complexity")
print(f"P-value for Complexity (beta_2): {p_value_beta_2:.6f}")

if p_value_beta_2 < 0.05 and beta_2_fit > 0:
    print(" -> RESULT: Complexity coefficient is statistically significant and positive.")
else:
    print(" -> RESULT: Complexity coefficient is NOT significant.")

# --- 4. Final Falsification Check ---
#
print("\n--- 5. Falsification Verdict ---")
if delta_gamma > 0 and p_value_beta_2 < 0.05 and beta_2_fit > 0:
    print("VERDICT: \033[1mCONFIRMED\033[0m")
    print("The simulation successfully recovered the predicted signatures.")
    print("The analysis pipeline correctly isolated the information-geometric effect.")
else:
    print("VERDICT: \033[1mFALSIFIED\033[0m")
    print("The simulation failed to recover the predicted signatures.")
    print("This indicates a null result in a real experiment would falsify the hypothesis.")
