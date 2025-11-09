# SCPN Simulation Suite: L05_Geometric_Qualia_TDA
# Layer 5: Geometric Qualia Hypothesis (TDA) Validation

## 1. Objective
This simulation provides a conceptual validation of the "Geometric Qualia Hypothesis" methodology from *Paper 0*. It demonstrates how Topological Data Analysis (TDA) can be used to distinguish between the "shapes" of different neural state spaces, which the hypothesis claims correspond to subjective qualia.

## 2. Simulation Logic
The script `run_l5_tda_sim.py` performs the following:
1. **Generates Mock Data:** Creates two `numpy` point clouds:
* `data_calm`: $N$ points sampled from a single, well-defined circle (represents an integrated, simple state: $b_0=1, b_1=1$).
* `data_anxious`: $N$ points sampled from three distinct, small, noisy clusters (represents a fragmented state: $b_0=3, b_1=0$).
2. **Runs TDA Pipeline:** Uses the `ripser` library to perform a Vietoris-Rips filtration and compute the persistent homology (PH) for both datasets.
3. **Calculates Betti Numbers:** Extracts the Betti numbers ($b_k$) from the PH results.
4. **Plots Persistence Diagrams:** Uses the `persim` library to plot the "persistence diagrams" for both states, providing a clear visual representation of their different topological "fingerprints."

## 3. How to Run
This simulation requires TDA-specific libraries.

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn ripser persim

# Run the simulation and analysis
python run_l5_tda_sim.py
4. Expected Output
The script will print the computed Betti numbers for both states, showing that they are different. It will also save a plot (L05_TDA_Persistence_Diagrams.png) showing the two distinct persistence diagrams, visually confirming that the "shape of feeling" is a measurable quantity.
