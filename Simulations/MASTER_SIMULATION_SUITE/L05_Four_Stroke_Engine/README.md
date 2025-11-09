# SCPN Simulation Suite: L05_Four_Stroke_Engine
# Layer 5: The 'Four-Stroke Engine' Action-Perception Cycle

## 1. Objective
This simulation provides a conceptual, agent-based model of the L5 'Four-Stroke Engine' described in *Paper 0*.

The goal is to demonstrate that an agent whose computation is explicitly structured into the four proposed neuro-anatomical phases (Policy Selection, Prediction, Error Processing, Consolidation) can successfully learn and minimize Free Energy (surprise).

## 2. Simulation Logic
The script `run_l5_engine_sim.py` simulates an agent learning a simple, hidden "world state."
1. **World:** A simple, 1D world with a "true state" (e.g., 42).
2. **Agent:** The agent has a "belief" about the world state.
3. **Four-Stroke Cycle:** The simulation runs in discrete "days" (epochs). Each day consists of:
* **Phase 1: Policy Selection (Basal Ganglia):** The agent decides on a "policy" (e.g., "Guess 30," "Guess 50").
* **Phase 2: Prediction Generation (Cerebellum):** The agent generates a "prediction" based on its policy (this is the guess itself).
* **Phase 3: Error Processing (Cortex):** The agent "acts," makes its guess, and receives the "sensory input" (the true state). It then calculates the Prediction Error ($\epsilon = \text{true\_state} - \text{prediction}$).
* **Phase 4: Model Consolidation (Sleep):** At the end of the "day," the agent's belief (its internal model) is updated using the stored Prediction Error ($\epsilon$).
4. **Analysis:** The script plots the agent's belief and the prediction error over time, showing convergence.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_l5_engine_sim.py
4. Expected Output
The script will save a plot (L05_Four_Stroke_Engine_Convergence.png) showing:

Agent's Belief: Starting from an incorrect value and converging to the "True State" (42).

Prediction Error: Starting high and converging to zero as the agent's model becomes accurate. This validates the computational viability of the four-stroke architecture.
