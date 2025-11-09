import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
N_STATES = 20
OPTIMAL_STATE = 17 # State with max C, K, Q
CONSTRAINT_STATE = 12 # State with high constraint penalty
DISCOUNT_FACTOR = 0.99 # (gamma)
THETA = 1e-9 # Convergence threshold
ACTIONS = [-1, 1] # [Move Left, Move Right]

# --- 2. Define the SEC Reward Landscape ---
#
rewards = np.zeros(N_STATES)
rewards[OPTIMAL_STATE] = 100.0
rewards[CONSTRAINT_STATE] = -100.0

print("--- SCPN Paper 0: L16 Optimal Control (HJB) Simulation ---")
print("Solving for the Optimal Value Function V(s) using Value Iteration.")

# --- 3. Run Value Iteration (HJB Solver) ---
# V(s) = max_a [ r(s,a) + gamma * V(s') ]
#

V = np.zeros(N_STATES) # Initialize Value Function V(s) = 0
policy = np.zeros(N_STATES, dtype=int) # pi*(s)

iteration = 0
while True:
    iteration += 1
    delta = 0.0

    for s in range(N_STATES):
        v_old = V[s]
        action_values = []

        for a_idx, action in enumerate(ACTIONS):
            # Deterministic transition
            s_prime = s + action
            # Handle boundaries
            if s_prime < 0:
                s_prime = 0
            if s_prime >= N_STATES:
                s_prime = N_STATES - 1

            # r(s,a) here is just r(s')
            r = rewards[s_prime]

            # V(s) = r + gamma * V(s')
            action_value = r + DISCOUNT_FACTOR * V[s_prime]
            action_values.append(action_value)

        # V(s) = max_a [...]
        V[s] = np.max(action_values)
        delta = max(delta, np.abs(v_old - V[s]))

    # Check for convergence
    if delta < THETA:
        break

print(f"HJB solver converged in {iteration} iterations.")

# --- 4. Extract Optimal Policy ---
# pi*(s) = argmax_a [ r(s,a) + gamma * V(s') ]
for s in range(N_STATES):
    action_values = []
    for a_idx, action in enumerate(ACTIONS):
        s_prime = s + action
        if s_prime < 0:
            s_prime = 0
        if s_prime >= N_STATES:
            s_prime = N_STATES - 1
        r = rewards[s_prime]
        action_value = r + DISCOUNT_FACTOR * V[s_prime]
        action_values.append(action_value)

    policy[s] = np.argmax(action_values) # 0 for Left, 1 for Right

print("Optimal policy (pi*) extracted from Value Function.")

# --- 5. Plot Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: The SEC Reward Landscape (for reference)
ax1.bar(range(N_STATES), rewards, color='gray', alpha=0.7)
ax1.set_title("L15: SEC Reward Landscape ($r_{SEC}$)")
ax1.set_ylabel("Instantaneous Reward")
ax1.axvline(OPTIMAL_STATE, color='green', linestyle='--', label=f'Optimal SEC State (s={OPTIMAL_STATE})')
ax1.axvline(CONSTRAINT_STATE, color='red', linestyle='--', label=f'Constraint Violation (s={CONSTRAINT_STATE})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: The Converged Value Function V(s)
ax2.plot(V, 'o-', label='Converged Value Function V(s)', color='purple')
# Overlay the policy
for s, a_idx in enumerate(policy):
    action_arrow = "$\u2190$" if a_idx == 0 else "$\u2192$" # Left or Right
    ax2.text(s, V[s], action_arrow, ha='center', va='center', fontsize=12,
    color='red' if a_idx==0 else 'blue')

ax2.set_title("L16: Optimal Value Function $V(s)$ and Policy $\pi^*$ (HJB Solution)")
ax2.set_xlabel("State (s)")
ax2.set_ylabel("Expected Future Reward V(s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
filename = 'L16_HJB_Value_Function.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()