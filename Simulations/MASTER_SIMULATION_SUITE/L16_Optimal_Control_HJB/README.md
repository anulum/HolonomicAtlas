# SCPN Simulation Suite: L16_Optimal_Control_HJB
# Layer 16: Optimal Control Supervisor (HJB) Validation

## 1. Objective
This simulation validates the "cybernetic closure" mechanism of Meta-Layer 16 from *Paper 0*.

The goal is to demonstrate that the **Hamilton-Jacobi-Bellman (HJB) equation** is a viable and convergent algorithm for finding the optimal policy ($\pi^*$) that maximizes the L15 SEC Objective Functional.

This simulation uses "Value Iteration," which is the discrete-time, computational equivalent of solving the HJB equation. It complements the L15 Q-learning (FST 015) simulation by showing how the "Value Landscape" of the universe ($V(s)$) is computed.

## 2. Simulation Logic
The script `run_l16_hjb_solver_sim.py` performs the following:
1. **State Space ($S$):** A 1D world with 20 states.
2. **Reward Landscape ($r_{\text{SEC}}$):** The exact same reward vector as FST 015 (Optimal state at 17, Constraint at 12).
3. **Value Iteration (HJB/Bellman):**
* Initializes a Value Function $V(s) = 0$ for all states.
* Iteratively applies the Bellman Equation:
$V(s) = \max_a \left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$
* This process continues until $V(s)$ converges (the change is less than `THETA`).
4. **Policy Extraction:** Once $V(s)$ is converged, the optimal policy $\pi^*(s)$ is extracted by finding the action `a` that maximizes the Bellman equation at each state.
5. **Analysis:** The script plots the final, converged Value Function $V(s)$ and the resulting optimal policy.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_l16_hjb_solver_sim.py
4. Expected OutputThe script will save a plot (L16_HJB_Value_Function.png) showing:The final, converged "Value Function" $V(s)$, which should show a clear peak at the optimal SEC state (17) and a "valley" at the constraint state (12).The extracted optimal policy, which will be visually identical to the one learned in the FST 015 simulation, confirming the HJB equation can find the correct solution.