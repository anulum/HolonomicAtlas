# SCPN Simulation Suite: L09_MERA_QEC_Stabilizer
# Layer 9/10: Holographic QEC and Stabiliser Transfer Lemma

## 1. Objective
This script provides a computational validation for the **Stabiliser Transfer Lemma**, a key component of the Multi-Scale QEC architecture described in *Paper 0*.

The lemma states that the error-correcting properties of the L9 'bulk' memory are holographically projected onto the L10 'boundary', following the inequality:
$d_{10} \ge d_9 / \chi$


This simulation validates this mathematical relationship.

## 2. Simulation Logic
The script `run_stabilizer_transfer_sim.py` is a conceptual proof, not a full tensor network simulation.
1. **Defines Parameters:** Sets the parameters for the lemma:
* `d9`: Code distance of the bulk (L9), e.g., 3.
* `chi`: Branching factor of the MERA-like projection, e.g., 2.
2. **Simulates Projection:** Models the projection $\Pi_*(S_9) \rightarrow S_{10}$. It does this by calculating the *resulting* boundary code distance `d10` based on a simplified model of how errors propagate from the boundary to the bulk.
3. **Asserts Lemma:** The script numerically tests the assertion `d10 >= d9 / chi`.
4. **Tests Operational Prediction:** It also simulates the "operational prediction" that reducing boundary complexity can improve the logical error rate.

## 3. How to Run
```bash
# Install dependencies
pip install numpy

# Run the validation script
python run_stabilizer_transfer_sim.py
4. Expected OutputThe script will print the results of the validation, showing the inputs ($d_9, \chi$), the calculated $d_{10}$, and a "VERDICT" on whether the Stabiliser Transfer Lemma holds true.
