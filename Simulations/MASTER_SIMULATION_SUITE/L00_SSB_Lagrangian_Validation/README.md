# SCPN Simulation Suite: L00_SSB_Lagrangian_Validation
# Foundational Physics: Spontaneous Symmetry Breaking and Mass Generation

## 1. Objective
This simulation provides the core validation for the physical mechanism of Spontaneous Symmetry Breaking (SSB) as described in *Paper 0*. It models the "Mexican Hat" potential of the $\Psi$-field and demonstrates how this leads to the generation of mass for the predicted infoton ($m_A$) and $\Psi$-Higgs ($m_h$) particles.

This script serves as the primary computational basis for the analyses in *Paper 18* and the falsification conditions in *Paper 19*.

## 2. Simulation Logic
The script `run_ssb_validation.py` performs the following:
1. **Defines Parameters:** Sets the core Lagrangian parameters:
* `mu_sq`: The quadratic coefficient (e.g., 2.0)
* `lambda_`: The quartic coefficient (e.g., 1.0)
* `g`: The U(1) gauge coupling constant (e.g., 0.65)
2. **Calculates VEV:** Solves for the minimum of the potential $V(|\Psi|) = -\mu^2 |\Psi|^2 + \lambda |\Psi|^4$ to find the Vacuum Expectation Value $v = \sqrt{\mu^2 / (2\lambda)}$.
3. **Calculates Masses:** Uses the derived VEV ($v$) to calculate the predicted particle masses based on the theory:
* **$\Psi$-Higgs mass:** $m_h = \sqrt{4\mu^2}$
* **Infoton mass:** $m_A = g \cdot v$
4. **Generates Plot:** Saves a plot (`L00_Mexican_Hat_Potential.png`) visualizing the potential and its non-zero minimum, which is the VEV.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_ssb_validation.py
4. Expected Output
The script will print the derived VEV and particle masses to the console and save a .png file of the potential.
