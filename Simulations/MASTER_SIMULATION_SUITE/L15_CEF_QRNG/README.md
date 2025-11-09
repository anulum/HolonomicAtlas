# SCPN Simulation Suite: L15_CEF_QRNG
# Layer 15/Axiom 3: Causal Entropic Force (CEF) QRNG Test

## 1. Objective
This package simulates the *data analysis pipeline* for the falsifiable experiment "Prediction II" described in *Paper 0*.

The goal is to validate the statistical methodology for detecting a weak, time-lagged correlation between two noisy time-series:
1. **$C(t)$:** The Collective Coherence metric (e.g., from HRV/EEG).
2. **$D(t)$:** The Randomness Deviation metric (from QRNG data).

The experiment hypothesizes that a peak in $C(t)$ will cause a correlated peak in $D(t)$ at a short time lag $\tau$, due to Causal Entropic Forces.

## 2. Simulation Logic
The script `run_qrng_analysis_sim.py` performs the following steps:
1. **Generate Mock Data:** Creates two noisy time-series, `C_t` and `D_t`.
2. **Inject Signature:** Creates a "coherence event" (a peak) in `C_t`. It then injects a smaller, corresponding peak into `D_t` at a specified `TRUE_LAG`.
3. **Run Analysis:**
* Applies smoothing (as one would with real data).
* Calculates the full cross-correlation function between the two signals.
* Finds the lag at which the maximum correlation occurs.
4. **Test Hypothesis:** Reports whether the detected lag matches the `TRUE_LAG`, demonstrating the analysis pipeline can successfully recover the predicted signature.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_qrng_analysis_sim.py
4. Expected Output
The script will print a summary of the analysis and save a plot (L15_CEF_QRNG_CrossCorrelation.png):

The true injected lag vs. the detected lag.

A final "VERDICT" on whether the analysis pipeline successfully "CONFIRMED" the hidden signature, validating the method.
