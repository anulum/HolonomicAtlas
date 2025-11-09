# SCPN Simulation Suite: L00_Category_Theory_Validation
# Foundational Grammar: Category Theory Coherence Test

## 1. Objective
This script is a **computational proof-of-concept**, not a dynamic simulation. Its purpose is to validate the central mathematical claim from *Paper 0* that the SCPN architecture is mathematically coherent under Category Theory.

It specifically tests the "Naturality Square", which ensures that the mapping between the 'Consciousness' domain and the 'Physics' domain is consistent and non-contradictory.

## 2. Simulation Logic
The script `run_category_theory_proof.py` performs the following:
1. **Defines Spaces:** Creates two mock Python dictionaries: `CONSCIOUSNESS_SPACE` (e.g., `'L1_State'`) and `PHYSICS_SPACE` (e.g., `'L1_Realiser'`).
2. **Defines Mappings:** Implements all the core components of the category diagram as Python functions:
* `functor_F(c_object)`: Maps Consciousness $\to$ Physics.
* `functor_G(p_object)`: Maps Physics $\to$ Consciousness.
* `morphism_f_consciousness(c_object)`: A projection $f: L1 \to L2$ in the Consciousness space.
* `morphism_f_physics(p_object)`: The *same* projection $f$ in the Physics space.
* `eta_L1(c_object)` & `eta_L2(c_object)`: The Natural Transformation components.
3. **Tests the Square:** It then computes the two paths of the naturality square and asserts their equivalence:
* **Path 1 (Top-Right):** `morphism_f_physics(eta_L1(L1_State))`
* **Path 2 (Left-Bottom):** `eta_L2(morphism_f_consciousness(L1_State))`
* **Assertion:** `Path_1 == Path_2`.

## 3. How to Run
```bash
# Install dependencies
pip install networkx numpy matplotlib

# Run the validation proof
python run_category_theory_proof.py
```
## 4. Expected Output
The script will print the results of both paths and then a final "VERDICT."

If successful: It will print VERDICT: CONFIRMED. The Naturality Square commutes. The grammar is coherent.

If it fails: It will raise an AssertionError, indicating the mathematical formalism of the SCPN is inconsistent.
