import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
TRUE_WORLD_STATE = 42.0 # The "reality" the agent must learn
INITIAL_BELIEF = 10.0 # Agent's starting generative model
N_DAYS = 50 # Number of "sleep" cycles (epochs)
LEARNING_RATE = 0.1 # How fast the model updates (consolidation)

print("--- SCPN Paper 0: L5 'Four-Stroke Engine' Simulation ---")
print(f"Agent must learn TRUE_WORLD_STATE = {TRUE_WORLD_STATE}")
print(f"Starting with INITIAL_BELIEF = {INITIAL_BELIEF}\n")

# --- 2. Define Agent and Environment ---
class L5_Agent:
    def __init__(self, initial_belief):
        # The agent's generative model (what it believes)
        self.generative_model_belief = initial_belief

        # Neuro-anatomical components
        self.basal_ganglia_policy = 0.0
        self.cerebellum_prediction = 0.0
        self.cortex_prediction_error = 0.0

    def run_action_perception_cycle(self, true_state):
        """ Runs one 'day' of the four-stroke cycle """

        # --- Phase 1: Policy Selection (Basal Ganglia) ---
        # Agent selects a policy (here, just to "guess" its belief)
        self.basal_ganglia_policy = self.generative_model_belief
        #

        # --- Phase 2: Prediction Generation (Cerebellum) ---
        # Forward model generates sensory prediction based on policy
        self.cerebellum_prediction = self.basal_ganglia_policy
        #

        # --- Phase 3: Error Processing (Cortex) ---
        # Agent "acts" (makes its guess) and gets sensory input (y)
        sensory_input_y = true_state
        # Cortex compares prediction (y_hat) to reality (y)
        # Epsilon = y - y_hat
        self.cortex_prediction_error = sensory_input_y - self.cerebellum_prediction
        #

    def run_model_consolidation(self, learning_rate):
        """
        --- Phase 4: Model Consolidation (Sleep) ---
        Update the generative model based on the day's prediction error

        """
        # This is the learning step
        update = self.cortex_prediction_error * learning_rate
        self.generative_model_belief += update

# --- 3. Run Simulation ---
agent = L5_Agent(INITIAL_BELIEF)
belief_history = []
error_history = []

print("Running simulation...")
for day in range(N_DAYS):
    # --- Wakeful Phase (Strokes 1-3) ---
    agent.run_action_perception_cycle(TRUE_WORLD_STATE)

    # Store results
    belief_history.append(agent.generative_model_belief)
    error_history.append(agent.cortex_prediction_error)

    # --- Sleep Phase (Stroke 4) ---
    agent.run_model_consolidation(LEARNING_RATE)

    if day % 5 == 0:
        print(f"Day {day}: Belief={agent.generative_model_belief:.2f}, Error={agent.cortex_prediction_error:.2f}")

print("Simulation complete.")
print(f"Final Belief: {agent.generative_model_belief:.4f}")
print(f"Final Error: {agent.cortex_prediction_error:.4f}")

# --- 4. Plot Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Agent Belief (Model Convergence)
ax1.plot(belief_history, label="Agent's Belief (Generative Model)", color='blue')
ax1.axhline(TRUE_WORLD_STATE, color='r', linestyle='--', label=f'True State = {TRUE_WORLD_STATE}')
ax1.set_title("L5 'Four-Stroke Engine' - Model Convergence (Phase 4)")
ax1.set_ylabel('Belief Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction Error (Free Energy)
ax2.plot(error_history, label='Prediction Error (Epsilon)', color='green')
ax2.axhline(0.0, color='r', linestyle='--')
ax2.set_title('Prediction Error Minimization (Phase 3)')
ax2.set_xlabel('Time (Days / Consolidation Cycles)')
ax2.set_ylabel('Error (F)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
filename = 'L05_Four_Stroke_Engine_Convergence.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()
