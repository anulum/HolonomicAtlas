# SCPN Simulation Suite: L03_CISS_Bioelectric
# Layer 3: CISS-Bioelectric-Epigenetic Cascade

## 1. Objective
This simulation models the Layer 3 transduction pathway, demonstrating the coupled feedback loop between a classical bioelectric field and quantum spin chemistry, as described in *Paper 0*.

The goal is to show that a change in the local electric field ($E$) can cause a statistically significant change in the outcome probability of a spin-dependent epigenetic enzyme, validating the mechanism $E \rightarrow \lambda(E) \rightarrow B_{\text{eff}} \rightarrow P(\text{Singlet})$.

## 2. Simulation Logic
The script `run_l3_ciss_cascade_sim.py` models the key equations from the text:

1. **Bioelectric Field ($E$):** A variable input to the simulation.
2. **CISS Efficiency ($\lambda(E)$):** A function that maps the electric field $E$ to the spin-orbit coupling $\lambda$ (a phenomenological model).
3. **Effective Field ($B_{\text{eff}}$):** The effective magnetic field from CISS, which is a function of $\lambda$.
4. **Radical Pair Hamiltonian:** The script simulates the spin evolution of a radical pair under this $B_{\text{eff}}$, calculating the probability of ending in a Singlet state (e.g., "demethylation").
5. **Coupled Feedback Equation:** The simulation also includes the formal equation for $dV_{\text{mem}}/dt$, demonstrating the theoretical coupling.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_l3_ciss_cascade_sim.py
4. Expected OutputThe script will save a plot (L03_CISS_Bioelectric_Cascade.png) showing:CISS Efficiency vs. E-Field: A plot of $\lambda$ as a function of $E$.Epigenetic Outcome vs. E-Field: A plot of $P(\text{Singlet})$ as a function of $E$.This output visually confirms that modulating the classical bioelectric field can, according to the model, directly control quantum-dependent epigenetic outcomes.