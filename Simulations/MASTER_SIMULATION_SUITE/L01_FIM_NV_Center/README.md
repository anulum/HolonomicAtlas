# SCPN Simulation Suite: L01_FIM_NV_Center
# Layer 1/Axiom 2: Information-Geometric Deviations (NV-Center Test)

## 1. Objective
This package does not simulate the full quantum dynamics of an NV-MEA hybrid system. Instead, it simulates the *expected data and analysis pipeline* for the falsifiable experiment "Prediction I" described in *Paper 0*.

The goal is to test the hypothesis that a quantum sensor's decoherence rate ($\Gamma$) correlates with the *informational complexity* (a proxy for the Fisher Information Metric, $g_{\text{FIM}}$) of a nearby neural culture, independent of its classical electromagnetic (EM) field.

## 2. Simulation Logic
The script `run_nv_mea_analysis.py` performs the following steps:
1. **Generate Mock Data:** Creates a time-series of simulated data for two conditions:
* **"Spontaneous" (Hypothesis True):** Generates a classical EM field $B(t)$ and an independent complexity metric $C(t)$. The decoherence $\Gamma(t)$ is modeled as $\Gamma = \beta_0 + \beta_1 \cdot B(t) + \beta_2 \cdot C(t) + \text{noise}$, where $\beta_2 > 0$.
* **"Isomorphic Replay" (Control):** Generates the *exact same* $B(t)$ but sets $C(t) = 0$ (no intrinsic complexity). $\Gamma(t)$ is modeled as $\Gamma = \beta_0 + \beta_1 \cdot B(t) + \text{noise}$.
2. **Run Statistical Analysis:**
* It calculates the "Excess Decoherence" $\Delta\Gamma = \text{mean}(\Gamma_{\text{spontaneous}}) - \text{mean}(\Gamma_{\text{replay}})$.
* It performs the multiple linear regression analysis on the "Spontaneous" data to recover the coefficients $\beta_1$ and $\beta_2$.
3. **Test Falsification:** Reports whether the analysis successfully finds $\Delta\Gamma > 0$ and $\beta_2 > 0$, validating the predicted signature.

## 3. How to Run
```bash
# Install dependencies
pip install numpy pandas statsmodels

# Run the simulation and analysis
python run_nv_mea_analysis.py
```
4. Expected Output
The script will print a summary of the analysis, showing:
The true $\Delta\Gamma$ vs. the measured $\Delta\Gamma$.
The true regression coefficients vs. the fitted coefficients.
A final statement on whether the hypothesis was "Confirmed" or "Falsified" based on the simulated data, demonstrating the viability of the experimental design.