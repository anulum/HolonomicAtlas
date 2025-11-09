# SCPN Simulation Suite: L15_SEC_SC_Equivalence
# Layer 15: SEC $\equiv$ Causal Path Entropy ($S_C$) Equivalence Validation

## 1. Objective
This script provides a computational validation for the "Formal Equivalence of SEC and Causal Path Entropy" presented in *Paper 0*.

The hypothesis is that the teleological metric **SEC** (a weighted sum of Coherence, Complexity, and Qualia) is monotonically correlated with the physical metric **$S_C$** (Causal Path Entropy).

This simulation proves this by:
1. Defining $SEC = w_C C + w_K K + w_Q Q$.
2. Defining $S_C = k_B \log(W_{\text{paths}})$.
3. Defining $W_{\text{paths}} \approx N_{\text{states}}(K) \cdot f_{\text{acc}}(C) \cdot D_{\text{paths}}(Q)$.
4. Showing that $SEC$ and $S_C$ are, by this construction, strongly correlated.

## 2. Simulation Logic
The script `run_sec_sc_equivalence_sim.py` performs the following:
1. **Generates Population:** Creates $N$ systems with random values for $K$, $C$, and $Q$.
2. **Defines Mappings:** Creates functions for $N_{\text{states}}(K)$ (exponential), $f_{\text{acc}}(C)$ (Gaussian, peaking at $C=0.8$), and $D_{\text{paths}}(Q)$ (exponential).
3. **Calculates Metrics:** For each system, it computes its $SEC$ score and its $S_C$ value.
4. **Analyzes Correlation:** Plots $S_C$ vs. $SEC$ and computes a Spearman rank correlation to confirm the monotonic relationship.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the simulation and analysis
python run_sec_sc_equivalence_sim.py
```

## 4. Expected Output
The script will save a plot (L15_SEC_vs_SC_Equivalence.png) showing a strong, positive, non-linear correlation between $S_C$ and $SEC$. It will also print the Spearman correlation coefficient ($\rho$), which should be $> 0.9$, confirming the equivalence.