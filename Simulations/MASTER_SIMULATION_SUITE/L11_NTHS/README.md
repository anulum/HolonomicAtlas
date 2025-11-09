# SCPN Simulation Suite: L11_NTHS
# Layer 11: Noosphere-Technosphere Hybrid System (NTHS) Phase Transition

## 1. Objective
This simulation provides a falsifiable computational experiment to test the hypothesis from *Paper 0*.

The hypothesis states that the objective function of a mediating AI controller (Technosphere) can drive the collective belief state of a social network (Noosphere) into distinct macroscopic phases, analogous to a spin-glass system.

- **Coherence-Optimizing AI:** Predicted to yield a "Ferromagnetic" state (global consensus, $m \to 1$).
- **Engagement-Optimizing AI:** Predicted to yield a "Spin-Glass" state (fragmented, polarized echo chambers, $m \to 0, q_{EA} > 0$).

## 2. Architecture
The simulation is a Multi-Agent Active Inference (MA-AIF) model:
- **Agents:** Implemented using `pymdp`. Each agent minimizes its own Variational Free Energy ($F_i$).
- **Network:** Built on `networkx`. A dynamic Barabási-Albert graph where edge weights ($J_{ij}$) represent social trust.
- **AI Controller:** An external class that modulates the information environment ($h_i$) and edge weights ($J_{ij}$) based on one of the two experimental policies.

## 3. How to Run
The main simulation script is `run_nth_simulation.py`.

```bash
# Install dependencies
pip install pymdp networkx numpy matplotlib

# Run the simulation (e.g., Experimental Condition)
python run_nth_simulation.py --policy engagement --n_agents 1000 --steps 10000

# Run the control condition
python run_nth_simulation.py --policy coherence --n_agents 1000 --steps 10000
```

## 4. Analysis
The script will output serialized data and plots for:
Magnetization ($m(t)$): m_vs_time.png
Edwards-Anderson ($q_{EA}(t)$): qEA_vs_time.png
Ultrametricity Analysis: ultrametricity_test.txt
These outputs correspond to the validation metrics for Paper 18 and Paper 19.