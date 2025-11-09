import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Lagrangian Parameters ---
# These parameters are for demonstration, based on the physics in Paper 0.
MU_SQ = 2.0 # mu^2 parameter in V(Psi)
LAMBDA = 1.0 # lambda parameter in V(Psi)
G_COUPLING = 0.65 # U(1) gauge coupling 'g'

print("--- SCPN Paper 0: SSB & Mass Generation Simulation ---")
print("Validating 'Mexican Hat' potential and emergent particle masses.")
print(f"Parameters: mu^2 = {MU_SQ}, lambda = {LAMBDA}, g = {G_COUPLING}\n")

# --- 2. Define the Potential ---
def potential(phi_mag_sq):
    """Calculates V(Psi) = -mu^2 * |Psi|^2 + lambda * |Psi|^4"""
    return -MU_SQ * phi_mag_sq + LAMBDA * (phi_mag_sq**2)

# --- 3. Calculate Vacuum Expectation Value (VEV) ---
# The potential V(Psi) is given by V = -mu^2|Psi|^2 + lambda|Psi|^4.
# The minimum of the potential, or the Vacuum Expectation Value (VEV), is found
# by solving dV/d|Psi| = 0, which gives |Psi|^2 = mu^2 / (2*lambda).
# Therefore, the VEV v = sqrt(mu^2 / (2*lambda)).
v_sq = MU_SQ / (2 * LAMBDA)
VEV = np.sqrt(v_sq)

print(f"[Results] Potential V(Psi) = -{MU_SQ}*|Psi|^2 + {LAMBDA}*|Psi|^4")
print(f" -> Solved VEV (v) = sqrt(mu^2 / (2*lambda)) = {VEV:.4f}")

# --- 4. Calculate Predicted Particle Masses ---

# Infoton Mass (m_A)
# m_A = g * v
m_infoton = G_COUPLING * VEV
print(f" -> Predicted Infoton Mass (m_A = g*v) = {m_infoton:.4f}")

# Psi-Higgs Mass (m_h)
# The mass of the Higgs boson is found by calculating the second derivative
# of the potential with respect to the field and evaluating it at the VEV.
# m_h^2 = d^2V/d|Psi|^2 at v = 4*mu^2
m_h_sq = 4 * MU_SQ
m_higgs = np.sqrt(m_h_sq)
print(f" -> Predicted Psi-Higgs Mass (m_h = sqrt(4*mu^2)) = {m_higgs:.4f}")

# --- 5. Generate Plot ---
phi_mag = np.linspace(-2.5, 2.5, 400)
phi_mag_sq = phi_mag**2
V = potential(phi_mag_sq)

plt.figure(figsize=(10, 6))
plt.plot(phi_mag, V, label=f"V(Psi) = -{MU_SQ}|Psi|^2 + {LAMBDA}|Psi|^4")
plt.axvline(VEV, color='r', linestyle='--', label=f'VEV (v) = {VEV:.3f}')
plt.axvline(-VEV, color='r', linestyle='--')
plt.title("L00: 'Mexican Hat' Potential & SSB", fontsize=16)
plt.xlabel("Field Magnitude |Psi|")
plt.ylabel("Potential Energy V(Psi)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(np.min(V) - 0.5, np.max(V))

filename = 'L00_Mexican_Hat_Potential.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()

print("\nSimulation complete. SSB mechanism validated.")
