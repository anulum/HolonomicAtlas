# SCPN Simulation Suite: L00_FIM_Pullback
# Foundational Physics: Operational Pullback Protocol (FIM) Validation

## 1. Objective
This script provides a computational validation for the **Operational Pullback Protocol** described in *Paper 0*. This is a purely mathematical proof-of-concept.

The goal is to demonstrate that the abstract Fisher Information Metric ($I_{ij}$) on the statistical manifold ($\Theta$) can be "pulled back" onto the physical spacetime manifold ($M$) to create a concrete, physical metric $g^F_{\mu\nu}$.

## 2. Simulation Logic
The script `run_fim_pullback_sim.py` implements the pullback formula:
$g^F_{\mu\nu} = \partial_\mu \theta^i \cdot I_{ij} \cdot \partial_\nu \theta^j$


In matrix form: $g^F = J^T \cdot I \cdot J$

1. **Define $I$:** A simple 2x2 FIM ($I_{ij}$) is defined.
2. **Define $J$:** A 2x2 Jacobian matrix ($J = \partial_\mu \theta^i$) is defined, representing the "section" $\theta(x)$.
3. **Compute $g^F$:** The script performs the matrix multiplication $J^T \cdot I \cdot J$.
4. **Assert:** The script prints the resulting $g^F_{\mu\nu}$ matrix, confirming that a non-trivial spacetime metric is generated. This metric is what would be used in the infoton's kinetic term.

## 3. How to Run
```bash
# Install dependencies
pip install numpy

# Run the validation script
python run_fim_pullback_sim.py
4. Expected OutputThe script will print the inputs ($I$ and $J$) and the final computed $g^F_{\mu\nu}$ matrix. It will then print a "VERDICT: CONFIRMED" message, validating the mathematical procedure.
