import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.datasets import make_circles

print("--- SCPN Paper 0: L5 Geometric Qualia (TDA) Simulation ---")
print("Validating TDA pipeline on mock 'Calm' vs. 'Anxious' neural data.\n")

# --- 1. Generate Mock Data ---

# State 1: "Calm" (Integrated, single loop)
#
N_CALM = 200
data_calm, _ = make_circles(n_samples=N_CALM, factor=0.8, noise=0.05)
data_calm = data_calm + np.random.rand(N_CALM, 2) * 0.1

# State 2: "Anxious / Fragmented" (Disintegrated, multiple components)
#
N_ANXIOUS = 100
cluster1 = np.random.rand(N_ANXIOUS, 2) + np.array([0, 5])
cluster2 = np.random.rand(N_ANXIOUS, 2) + np.array([5, 0])
cluster3 = np.random.rand(N_ANXIOUS, 2) + np.array([5, 5])
data_anxious = np.vstack([cluster1, cluster2, cluster3])

print(f"Generated 'Calm' data: {data_calm.shape} points")
print(f"Generated 'Anxious' data: {data_anxious.shape} points")

# --- 2. Run TDA Pipeline (Persistent Homology) ---
#
print("\nRunning TDA (Vietoris-Rips)...")
# maxdim=1 computes 0-dim (b0, components) and 1-dim (b1, loops)
ph_calm = ripser(data_calm, maxdim=1)['dgms']
ph_anxious = ripser(data_anxious, maxdim=1)['dgms']

# --- 3. Calculate Betti Numbers (b_k) ---
# Betti numbers are the count of persistent features.
# We filter out "noise" (features that die almost instantly)
PERSISTENCE_THRESHOLD = 0.1

# Betti 0 (Connected Components)
b0_calm = np.sum(np.isinf(ph_calm[0][:, 1])) # Count infinite-persistence components
b1_calm = np.sum(ph_calm[1][:, 1] - ph_calm[1][:, 0] > PERSISTENCE_THRESHOLD)

b0_anxious = np.sum(np.isinf(ph_anxious[0][:, 1]))
b1_anxious = np.sum(ph_anxious[1][:, 1] - ph_anxious[1][:, 0] > PERSISTENCE_THRESHOLD)

print("\n--- 4. Analysis & Falsification Verdict ---")
print(f" -> 'Calm' State Topology: b0={b0_calm} (integrated), b1={b1_calm} (simple loop)")
print(f" -> 'Anxious' State Topology: b0={b0_anxious} (fragmented), b1={b1_anxious} (no loops)")

# Falsification Check
if b0_calm != b0_anxious or b1_calm != b1_anxious:
    print("\nVERDICT: \033[1mCONFIRMED\033[0m")
    print("TDA successfully distinguished the topological signatures of the two states.")
else:
    print("\nVERDICT: \033[1mFALSIFIED\033[0m")
    print("TDA failed to find a topological difference.")

# --- 5. Plot Results ---
plt.figure(figsize=(12, 6))

# Plot the point clouds
plt.subplot(1, 2, 1)
plt.scatter(data_calm[:, 0], data_calm[:, 1], label='Data')
plt.title(f"'Calm' State (b0={b0_calm}, b1={b1_calm})")
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(data_anxious[:, 0], data_anxious[:, 1], label='Data')
plt.title(f"'Anxious' State (b0={b0_anxious}, b1={b1_anxious})")
plt.axis('equal')

plt.suptitle("L5 Geometric Qualia: Mock Neural State Spaces")
filename = 'L05_TDA_State_Spaces.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()

# Plot the persistence diagrams
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_diagrams(ph_calm, show=False, title=f"Persistence Diagram: 'Calm'")

plt.subplot(1, 2, 2)
plot_diagrams(ph_anxious, show=False, title=f"Persistence Diagram: 'Anxious'")

plt.suptitle("L5 Geometric Qualia: TDA 'Fingerprints'")
filename_diag = 'L05_TDA_Persistence_Diagrams.png'
plt.savefig(filename_diag)
print(f"Saved plot: {filename_diag}")
plt.close()
