# SCPN Simulation Suite: L04_SOC_Quasicriticality
# Layer 4: Self-Organised Criticality (SOC) and Quasicriticality

## 1. Objective
This simulation validates the "Universal Dynamic Regime" of the SCPN. It is designed to prove two core claims from *Paper 0*:
1. That a simple, local homeostatic control rule, $\frac{d\sigma}{dt} = -\kappa(\sigma - 1)$, can robustly tune a network to the critical point ($\sigma=1$). This is the "Self-Organised Criticality (SOC)" mechanism.
2. That this critical state ($\sigma=1$) is, in fact, the "edge of chaos," and its definitive signature is the emergence of scale-free "neuronal avalanches" that follow a power-law distribution, $P(S) \sim S^{-\tau}$.

## 2. Simulation Logic
The script `run_soc_simulation.py` implements a simple branching process.
1. **Network Model:** A simple network where, at each step, $N$ active nodes activate $k$ new nodes. $k$ is drawn from $\text{Poisson}(\sigma)$.
2. **Branching Parameter ($\sigma$):** The simulation is initialized in a *supercritical* state ($\sigma > 1$).
3. **Homeostatic Controller:** At each timestep, $\sigma$ is updated using the discrete form of the SOC control equation: $\sigma(t+1) = \sigma(t) - \kappa(\sigma(t) - 1)$.
4. **Avalanche Recording:** The simulation runs, and $\sigma$ converges to 1. After it has stabilized, the simulation continues to run, and the size ($S$) of every "avalanche" (a cascade of activity) is recorded.
5. **Analysis:** The script generates two plots:
* **Convergence Plot:** Shows $\sigma(t)$ converging to 1 over time.
* **Power-Law Plot:** A log-log plot of the avalanche size distribution $P(S)$, demonstrating the linear signature of a power law.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the simulation and analysis
python run_soc_simulation.py
4. Expected OutputThe script will save two plots:L04_SOC_Convergence.png: Shows $\sigma(t)$ starting at a high value and converging to 1.L04_SOC_Power_Law.png: Shows the log-log plot of $P(S)$ vs. $S$, which should be a straight line, confirming the $P(S) \sim S^{-\tau}$ signature.