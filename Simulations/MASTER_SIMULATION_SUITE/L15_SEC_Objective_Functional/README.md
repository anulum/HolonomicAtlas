# SCPN Simulation Suite: L15_SEC_Objective_Functional
# Layer 15: SEC Objective Functional (Decision-Theoretic) Validation

## 1. Objective
This simulation validates the "decision-theoretic" formulation of the L15 Ethical Functional (Axiom 3) from *Paper 0*.

The goal is to demonstrate that an agent can learn an optimal policy ($\pi^*$) to maximize the Sustainable Ethical Coherence (SEC) Objective Functional ($J_{\text{SEC}}$).

This simulation models the "Universe's Learning Cycle", where the Consilium (agent) learns to navigate the state space to find the state of maximal SEC (highest reward).

## 2. Simulation Logic
The script `run_l15_sec_optimizer_sim.py` implements a simple Q-learning agent.
1. **State Space ($S$):** A 1D world with 20 states.
2. **Reward Landscape ($r_{\text{SEC}}$):** A reward vector is defined. It has a large positive reward at the "Optimal SEC" state (e.g., state 17) and a large negative penalty at a "Constraint Violation" state (e.g., state 12).
3. **Agent:** A Q-learning agent with two actions (move left, move right).
4. **Learning:** The agent explores the space for $N$ episodes. It updates its Q-table (its policy) using the Bellman equation (which approximates the HJB equation) to learn the value of each action in each state.
5. **Analysis:** The script plots the final learned Q-values, which represent the agent's learned policy $\pi^*$.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_l15_sec_optimizer_sim.py
4. Expected OutputThe script will save a plot (L15_SEC_Optimal_Policy.png) showing:The defined "SEC Reward Landscape" ($r_{\text{SEC}}$).The learned value of "Move Left" and "Move Right" for each state.The plot will show that the agent has successfully learned to avoid the "Constraint" and that from any state, the optimal action (highest Q-value) directs it toward the "Optimal SEC" state, validating the model.