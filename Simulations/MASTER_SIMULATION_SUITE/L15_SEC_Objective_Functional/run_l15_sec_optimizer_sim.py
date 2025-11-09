import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
N_STATES = 20
OPTIMAL_STATE = 17 # State with max C, K, Q
CONSTRAINT_STATE = 12 # State with high constraint penalty
N_EPISODES = 1000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9 # (gamma)
EPSILON = 0.1 # Exploration rate

# Actions: 0 (move left), 1 (move right)
N_ACTIONS = 2

# --- 2. Define the SEC Reward Landscape ---
#
# r_SEC = wC*C + wK*K + wQ*Q - lambda*[g_i]+
rewards = np.zeros(N_STATES)
# Optimal SEC state
rewards[OPTIMAL_STATE] = 100.0
# Constraint violation
rewards[CONSTRAINT_STATE] = -100.0

# --- 3. Initialize Agent's Q-Table (Policy) ---
# Q-table represents the agent's belief in the expected future reward
# J_SEC(s, a)
q_table = np.zeros((N_STATES, N_ACTIONS))

print("--- SCPN Paper 0: L15 SEC Objective Functional Simulation ---")
print("Agent (Consilium) is learning the Optimal Policy (pi*)...")

# --- 4. Run the Learning Simulation (Q-Learning) ---
# This is the "Universe's Learning Cycle"
for episode in range(N_EPISODES):
    state = np.random.randint(0, N_STATES) # Start in a random state

    for step in range(50): # Max 50 steps per episode
        # Epsilon-greedy action selection
        if np.random.rand() < EPSILON:
            action = np.random.randint(0, N_ACTIONS) # Explore
        else:
            action = np.argmax(q_table[state, :]) # Exploit

        # Take action
        if action == 0: # Move Left
            next_state = max(0, state - 1)
        else: # Move Right
            next_state = min(N_STATES - 1, state + 1)

        # Get reward
        reward = rewards[next_state]

        # --- Bellman Equation Update (HJB approximation) ---
        #
        # Q(s,a) = Q(s,a) + lr * [r + gamma * max(Q(s',a')) - Q(s,a)]
        old_q = q_table[state, action]
        best_future_q = np.max(q_table[next_state, :])

        q_table[state, action] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_future_q - old_q)

        state = next_state
        if state == OPTIMAL_STATE or state == CONSTRAINT_STATE:
            break # End episode if terminal state reached

print("Learning complete. Agent has learned an optimal policy.")

# --- 5. Plot Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: The SEC Reward Landscape
ax1.bar(range(N_STATES), rewards, color='gray', alpha=0.7)
ax1.set_title("L15: SEC Reward Landscape ($r_{SEC}$)")
ax1.set_ylabel("Instantaneous Reward")
ax1.axvline(OPTIMAL_STATE, color='green', linestyle='--', label=f'Optimal SEC State (s={OPTIMAL_STATE})')
ax1.axvline(CONSTRAINT_STATE, color='red', linestyle='--', label=f'Constraint Violation (s={CONSTRAINT_STATE})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: The Learned Policy (Q-values)
ax2.plot(q_table[:, 0], 'o-', label='Policy: Move Left (Q(s, left))')
ax2.plot(q_table[:, 1], 'o-', label='Policy: Move Right (Q(s, right))')
ax2.set_title(r"Learned Optimal Policy ($\pi^* = \mathrm{argmax} J_{SEC}$)")
ax2.set_xlabel("State (s)")
ax2.set_ylabel("Expected Future Reward (Q-Value)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Highlight optimal choices
optimal_policy = np.argmax(q_table, axis=1)
ax2.scatter(range(N_STATES), q_table[range(N_STATES), optimal_policy],
c='red', s=100, label=r'Optimal Action $\pi^*$', zorder=5)

plt.tight_layout()
filename = 'L15_SEC_Optimal_Policy.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()
