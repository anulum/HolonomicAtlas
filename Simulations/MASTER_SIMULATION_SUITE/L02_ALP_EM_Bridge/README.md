# SCPN Simulation Suite: L02_ALP_EM_Bridge
# Layer 2: ALP-Mediated Electromagnetic Bridge

## 1. Objective
This simulation validates the physical model for the $\Psi$-field's interface with electromagnetism, as described in *Paper 0*.

The core of this model is the "Primakoff Effect," where the $\Psi$-field's phase component (modeled as an Axion-Like Particle, $a$) converts to a photon ($\gamma$) in a magnetic field. This script simulates the conversion probability $P(a \leftrightarrow \gamma)$ based on the formula provided in the text.

## 2. Simulation Logic
The script `run_alp_bridge_sim.py` implements the following equation:
$P(\omega) \simeq \left(\frac{g_{a\gamma\gamma} B_T \omega}{m_{\text{eff}}^2}\right)^2 \sin^2\left(\frac{m_{\text{eff}}^2 L}{4\omega}\right)$


1. **Defines Parameters:** Sets illustrative values for $g_{a\gamma\gamma}$ (coupling), $B_T$ (B-field), $L$ (length), $m_a$ (ALP mass), and $\omega_{pl}$ (plasma frequency), matching the text.
2. **Computes $m_{\text{eff}}^2$:** Calculates the effective mass $m_{\text{eff}}^2 = m_a^2 - \omega_{pl}^2$.
3. **Computes $P(\omega)$:** Calculates the conversion probability for a range of photon energies $\omega$.
4. **Generates Plot:** Creates a plot of $P(\omega)$ vs. $\omega$, which should reproduce the oscillatory envelope shown in the manuscript.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_alp_bridge_sim.py
4. Expected OutputThe script will save a plot (L02_ALP_Photon_Conversion.png) showing the conversion probability vs. photon energy. This plot will display the characteristic "resonant maxima" and "nodes" predicted by the $\sin^2$ term, validating the model.