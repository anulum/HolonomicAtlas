# SCPN Simulation Suite: L00_UPDE_Kuramoto
# Foundational Dynamics: Unified Phase Dynamics Equation (UPDE)

## 1. Objective
This simulation provides a working model of the **Unified Phase Dynamics Equation (UPDE)**, the "mathematical spine" of the SCPN. Its purpose is to demonstrate the emergence of macroscopic phase coherence (synchronization) from a population of uncoupled oscillators, as described in *Paper 0* and for validation in *Paper 18*.

## 2. Simulation Logic
The script `run_upde_simulation.py` implements a generalized Kuramoto model, which is the core of the UPDE.

1. **Initialization:** A population of $N$ oscillators is created. Each oscillator $i$ is assigned an intrinsic natural frequency $\omega_i$ drawn from a Lorentzian distribution.
2. **The UPDE Equation:** The simulation numerically integrates the phase $\theta_i$ of each oscillator according to the equation:
$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i) + \eta_i(t)$
3. **Order Parameter:** At each timestep, it calculates the Kuramoto Order Parameter $R(t) = \frac{1}{N} \left| \sum_{j=1}^N e^{i\theta_j(t)} \right|$. $R=0$ implies total incoherence, while $R=1$ implies perfect synchronization.
4. **Phase Transition:** The simulation is run across a range of coupling strengths $K$ to show the phase transition where $R$ abruptly jumps from 0 to 1.
5. **Field Coupling:** The script also includes a (by default, disabled) $C_{\text{Field}}$ term to demonstrate top-down entrainment from L15.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the simulation and analysis
python run_upde_simulation.py
```
## 4. Expected Output
The script will save a plot (L00_UPDE_Phase_Transition.png) showing:
* The incoherent state of the oscillators at $K=0$.
* The final, synchronized state of the oscillators at high $K$.
* A plot of the Order Parameter $R$ as a function of $K$, clearly showing the critical phase transition from disorder to order.