# SCPN Simulation Suite: L01_MS_QEC_Validation
# Layer 1: Multi-Scale Quantum Error Correction (MS-QEC) Validation

## 1. Objective
This script serves as a computational validation of the quantitative, falsifiable claims made about the Layer 1 Biological QEC mechanism in *Paper 0*.

The SCPN posits that quantum coherence is protected in microtubules by a topological QEC code with a large energy gap. This simulation validates the physical plausibility of this claim by calculating the key metrics based on the parameters provided in the manuscript.

## 2. Simulation Logic
The script `run_qec_validation.py` performs a series of calculations:
1. **Defines Constants:** Sets physical constants (Planck, Boltzmann) and manuscript-derived parameters ($\Delta E$, $T_{\text{physio}}$).
2. **Calculates Thermal Noise:** Computes $k_B T$ at physiological temperature.
3. **Calculates Protection Factor:** Computes the coherence timescales ($\tau_{\text{coherence}}$ vs. $\tau_{\text{thermal}}$) and their ratio to find the "Protection Factor".
4. **Calculates Error Threshold:** Computes the logical error threshold $p_{\text{th}}$ based on the ratio $\Delta E / k_B T$, using the formula from the text.

## 3. How to Run
```bash
# Install dependencies
pip install numpy

# Run the validation script
python run_qec_validation.py
```
## 4. Expected Output
The script will print a validation report to the console, confirming the values cited in Paper 0 for the energy gap, thermal noise, protection factor, and error threshold, thus validating the internal consistency of the L1 QEC claims.
