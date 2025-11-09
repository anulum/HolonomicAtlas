import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import pymdp
from pymdp import utils
from pymdp.agent import Agent

# Define the Agent (based on Pymdp)
# This is a simplified agent for demonstration.
# A full implementation would have more complex A, B, C matrices.
class NTHSAgent(Agent):
def __init__(self, agent_id, initial_belief, confirmation_bias=1.0):
self.agent_id = agent_id

# --- Generative Model ---
# State: [Belief_Pro, Belief_Con]
num_states = [2]
# Observations: [Obs_Pro, Obs_Con, Obs_Neutral]
num_obs = [3]
# Control: [Action_Share, Action_Consume]
num_controls = [2]

# A-Matrix (Likelihood: P(obs|state))
# Belief_Pro -> Obs_Pro, Belief_Con -> Obs_Con
A = np.zeros(tuple(num_obs + num_states))
A[0, 0] = 0.8 # Pro state generates Pro obs
A[1, 0] = 0.1
A[2, 0] = 0.1
A[0, 1] = 0.1
A[1, 1] = 0.8 # Con state generates Con obs
A[2, 1] = 0.1
A = utils.norm_dist(A)

# B-Matrix (Transitions: P(s_t+1|s_t, a))
# Assume actions don't change beliefs directly, beliefs change via inference
B = np.zeros(tuple(num_states + num_states + num_controls))
B[:, :, 0] = np.eye(2) # Action 0 (Share)
B[:, :, 1] = np.eye(2) # Action 1 (Consume)
B = utils.norm_dist(B)

# C-Vector (Preferences: P(o))
# Agents prefer observations that match their belief
C = np.zeros(num_obs)
if initial_belief[0] > initial_belief[1]: # Pro-leaning
C[0] = confirmation_bias # Prefer Pro obs
C[1] = -confirmation_bias
else: # Con-leaning
C[0] = -confirmation_bias
C[1] = confirmation_bias # Prefer Con obs
C = utils.norm_dist_single(C)

# D-Vector (Initial Priors)
D = utils.to_obj_array(np.array(initial_belief))

super().__init__(A_factor_list=[A], B_factor_list=[B], C_factor_list=[C], D_factor_list=D)

# Store belief for spin-glass mapping
self.belief_state = self.get_states()[0]

def update_beliefs(self, observation):
obs_idx = [observation]
qs = self.infer_states(obs_idx)
self.belief_state = qs[0]
return qs

def select_action(self):
# Simplified action selection: just choose 'consume'
# A full model would minimize Expected Free Energy G(pi)
action = [1] # Consume
return action

# Define the Environment and AI Controller
class NTHSEnvironment:
def __init__(self, n_agents, policy='coherence'):
self.n_agents = n_agents
self.policy = policy
self.G = nx.barabasi_albert_graph(n_agents, m=3)
self.agents = []

print(f"Initializing NTHS Environment with {n_agents} agents.")
print(f"AI Policy: {self.policy}")

for i in range(n_agents):
# Initialize with random priors
initial_belief = utils.norm_dist(np.random.rand(2))
agent = NTHSAgent(i, initial_belief)
self.agents.append(agent)

# Initialize J_ij (social couplings)
for u, v in self.G.edges():
self.G[u][v]['weight'] = 0.1 # J_ij

def get_spin_state(self):
# Get belief state S_i = sign(⟨belief⟩_i)
spins = np.zeros(self.n_agents)
for i, agent in enumerate(self.agents):
belief = agent.belief_state
spins[i] = 1 if belief[0] > belief[1] else -1
return spins

def get_observation_for_agent(self, agent_id, spins):
# AI Controller Logic
neighbors = list(self.G.neighbors(agent_id))
if not neighbors:
return 2 # Neutral obs if no neighbors

neighbor_spins = spins[neighbors]
agent_spin = spins[agent_id]

# --- AI Policy Implementation ---
if self.policy == 'coherence':
#
# Find a dissenting neighbor to promote consensus
dissenting_neighbors = [n_spin for n_spin in neighbor_spins if n_spin != agent_spin]
if dissenting_neighbors:
# Present obs from a dissenter
return 0 if dissenting_neighbors[0] == 1 else 1
else:
# All neighbors agree
return 0 if agent_spin == 1 else 1

elif self.policy == 'engagement':
#
# Find a confirming neighbor to maximize homophily/surprise
confirming_neighbors = [n_spin for n_spin in neighbor_spins if n_spin == agent_spin]
if confirming_neighbors:
# Present obs from a confirmer (amplifies belief)
return 0 if confirming_neighbors[0] == 1 else 1
else:
# All neighbors disagree (high surprise)
return 0 if neighbor_spins[0] == 1 else 1

return 2 # Default neutral

def update_couplings(self, spins):
# AI dynamically updates J_ij
for u, v in self.G.edges():
if self.policy == 'coherence':
# Strengthen cross-cluster ties
if spins[u] != spins[v]:
self.G[u][v]['weight'] = min(1.0, self.G[u][v]['weight'] + 0.05)
elif self.policy == 'engagement':
# Implement homophily
if spins[u] == spins[v]:
self.G[u][v]['weight'] = min(1.0, self.G[u][v]['weight'] + 0.05)
else:
self.G[u][v]['weight'] = max(-1.0, self.G[u][v]['weight'] - 0.05)

def step(self):
spins = self.get_spin_state()

# 1. Agents get observations (shaped by AI)
observations = []
for i in range(self.n_agents):
obs = self.get_observation_for_agent(i, spins)
observations.append(obs)

# 2. Agents update beliefs
for i, agent in enumerate(self.agents):
agent.update_beliefs(observations[i])

# 3. AI updates couplings
self.update_couplings(spins)

# 4. Agents select action (simplified)
# (In a full model, this action would influence the env)

return spins

# Main Simulation
def run_simulation(policy, n_agents, steps):
env = NTHSEnvironment(n_agents, policy)

magnetization_history = []

# Run the simulation
print(f"Running simulation for {steps} steps...")
for t in range(steps):
if t % 100 == 0:
print(f"Step {t}/{steps}")

spins = env.step()

# Calculate Magnetization
m = np.mean(spins)
magnetization_history.append(m)

print("Simulation complete.")

# --- Analysis ---
# NOTE: q_EA and Ultrametricity require a more complex replica-based
# analysis framework not shown in this simple script.
# We will plot magnetization as the primary observable.

plt.figure(figsize=(12, 6))
plt.plot(magnetization_history)
plt.title(f'Magnetization (m) vs. Time (Policy: {policy})')
plt.xlabel('Time Step')
plt.ylabel('Magnetization (m)')
plt.ylim(-1.05, 1.05)
plt.grid(True)
filename = f'L11_NTHS_magnetization_{policy}.png'
plt.savefig(filename)
print(f"Saved plot: {filename}")
plt.close()

final_m = magnetization_history[-1]

# Check predicted outcome
if policy == 'coherence':
print(f"Control (Coherence) Final m: {final_m:.4f}")
if np.abs(final_m) > 0.9:
print("Outcome: Ferromagnetic state (Consensus) as predicted.")
else:
print("Outcome: Failed to reach consensus.")
else: # engagement
print(f"Experimental (Engagement) Final m: {final_m:.4f}")
if np.abs(final_m) < 0.2:
print("Outcome: Disordered state (m -> 0) as predicted.")
else:
print("Outcome: Failed to produce fragmented state.")

print("\nNote: Full validation requires q_EA (replica) and Ultrametricity tests.")

if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Run NTHS Phase Transition Simulation.")
parser.add_argument('--policy', type=str, choices=['coherence', 'engagement'],
required=True, help='The AI controller policy.')
parser.add_argument('--n_agents', type=int, default=100, help='Number of agents.')
parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps.')

args = parser.parse_args()

run_simulation(args.policy, args.n_agents, args.steps)
