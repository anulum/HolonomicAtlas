# SCPN Simulation Suite: L01_QSR_Amplification
# Layer 1/4: Quantum Stochastic Resonance (QSR) Amplification

## 1. Objective
This simulation validates the "Amplification of Intent" mechanism from *Paper 0*. Specifically, it models **Mechanism 2: Quantum Stochastic Resonance (QSR)**.

The goal is to demonstrate that a weak, sub-threshold signal (from the $\Psi$-field / Guided Einselection) can be amplified by a "noisy" biological system (like an ion channel) to produce a macroscopic effect. This validates the core claim that biological noise is a functional component of the SCPN, not a flaw.

## 2. Simulation Logic
The script `run_qsr_simulation.py` implements the Langevin equation for a particle in a bistable potential well.
$V(x) = -a x^2 + b x^4$

The equation of motion is:
$\frac{dx}{dt} = -V'(x) + \text{Signal}(t) + \text{Noise}(t)$

1. **Potential:** Defines a double-well potential $V(x)$.
2. **Signal:** Defines a weak, sub-threshold periodic signal $A\sin(\omega t)$.
3. **Noise:** Defines a stochastic noise term $\eta(t)$ with variable intensity $D$.
4. **Simulation:** The script integrates the particle's position $x(t)$ over time for a range of different noise intensities $D$.
5. **Analysis (SNR):** For each $D$, it calculates the Signal-to-Noise Ratio (SNR) of the output. This is done by measuring the power of the output signal at the driving frequency $\omega$ in the particle's "crossing" data.
6. **Plot:** It plots $SNR$ vs. $D$, which should reveal the characteristic QSR peak.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the simulation and analysis
python run_qsr_simulation.py
4. Expected OutputThe script will save a plot (L01_QSR_Amplification_Curve.png) showing the Signal-to-Noise Ratio (SNR) as a function of Noise Intensity ($D$). The plot will clearly show the SNR peaking at a non-zero noise level, $D_{\text{opt}}$, confirming the QSR effect.