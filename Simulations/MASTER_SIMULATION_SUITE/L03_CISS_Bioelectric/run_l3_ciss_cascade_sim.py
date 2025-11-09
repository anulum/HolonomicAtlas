import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters ---
E_field_range = np.linspace(0, 10.0, 50) # Range of Bioelectric Field (arbitrary units)
lambda_base = 0.5 # Base Spin-Orbit Coupling
lambda_sensitivity = 0.1 # How much E-field affects lambda
B_eff_scaling = 5.0 # How much lambda affects B_eff
J_coupling = 1e-9 # (J) Exchange interaction in Radical Pair (T)
A_hyperfine = 1e-9 # (A) Hyperfine coupling (T)

print("--- SCPN Paper 0: L3 CISS-Bioelectric Cascade Simulation ---")
print("Simulating E-Field -> lambda -> B_eff -> P(Singlet) cascade.")

# --- 2. Define Coupled Functions ---

def get_lambda(E):
    """ Phenomenological model for E-field modulating CISS efficiency """
    # E-field alters conformation, tuning spin-orbit coupling lambda
    # dV_mem/dt = -I_ion(V_mem, B_eff(lambda(E)))
    return lambda_base + lambda_sensitivity * E**2

def get_Beff(lambda_val):
    """ CISS B_eff as a function of spin-orbit coupling """
    # H_total = ... + (lambda/L^2)(sigma·L)
    return B_eff_scaling * lambda_val

def get_singlet_probability(Beff, J, A):
    """
    Simplified Radical Pair spin evolution
    H_RP = ... + B_local·(g1*S1 + g2*S2)
    This is a proxy calculation.
    """
    # Total magnetic field (B_eff from CISS, B_local from V_mem)
    B_total = Beff + A # Combine effective fields

    # Singlet-Triplet energy splitting
    delta_E_ST = np.sqrt(J**2 + B_total**2)

    # Probability of S-T conversion (proxy)
    P_ST_conversion = (B_total**2) / (delta_E_ST**2 + (1e-18))

    # Assume starts in Singlet, P(Singlet) is 1 - P(conversion)
    P_Singlet = 1.0 - P_ST_conversion
    return max(0, min(1, P_Singlet)) # Clamp probability [0, 1]

# --- 3. Run Simulation Cascade ---
lambda_values = []
Beff_values = []
P_singlet_values = []

print("Running cascade simulation...")
for E in E_field_range:
    # 1. E-field modulates CISS efficiency
    lambda_val = get_lambda(E)

    # 2. CISS efficiency determines B_eff
    Beff = get_Beff(lambda_val)

    # 3. B_eff biases Radical Pair and thus Epigenetic Outcome
    P_singlet = get_singlet_probability(Beff, J_coupling, A_hyperfine)

    lambda_values.append(lambda_val)
    Beff_values.append(Beff)
    P_singlet_values.append(P_singlet)

print("Simulation complete.")

# --- 4. Plot Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: E-Field -> CISS Efficiency
ax1.plot(E_field_range, lambda_values, 'b-o', label='lambda(E)')
ax1.set_title('L3 CISS-Bioelectric Coupling (Mechanism)')
ax1.set_xlabel('Bioelectric Field Strength (E) [Arbitrary Units]')
ax1.set_ylabel('CISS Spin-Orbit Coupling (lambda) [Arbitrary Units]')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: CISS -> Epigenetic Outcome
ax2.plot(E_field_range, P_singlet_values, 'r-s', label='P(Singlet) ~ P(Demethylation)')
ax2.set_title('L3 Transduction Cascade (Outcome)')
ax2.set_xlabel('Bioelectric Field Strength (E) [Arbitrary Units]')
ax2.set_ylabel('Probability of Singlet Outcome P(S)')
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
filename = 'L03_CISS_Bioelectric_Cascade.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()
