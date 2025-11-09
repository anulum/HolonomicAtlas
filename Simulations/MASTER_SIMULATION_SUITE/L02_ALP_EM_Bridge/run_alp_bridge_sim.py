import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Simulation Parameters ---
# Using the illustrative parameters from the Paper 0 figure
#
G_AGG = 1.0 # g_agg (ALP-photon coupling)
B_T = 1.0 # B_T (Transverse Magnetic Field)
L_COH = 10.0 # L (Coherence Length)
M_A = 0.8 # m_a (ALP intrinsic mass)
OMEGA_PL = 0.5 # omega_pl (Plasma frequency)

# Energy range to plot
omega_min = 0.01
omega_max = 5.0
N_POINTS = 1000

print("--- SCPN Paper 0: L2 ALP-EM Bridge Simulation ---")
print("Simulating ALP-Photon Conversion Probability (Primakoff Effect)")
print(f"Parameters: g_agg={G_AGG}, B_T={B_T}, L={L_COH}, m_a={M_A}, omega_pl={OMEGA_PL}\n")

# --- 2. Calculate Effective Mass ---
# m_eff^2 = m_a^2 - omega_pl^2
m_eff_sq = M_A**2 - OMEGA_PL**2
print(f"Calculated m_eff^2 = {m_eff_sq:.3f}")

# --- 3. Define Conversion Probability Function ---
# P(omega) ~ (g*B*w / m_eff^2)^2 * sin^2(m_eff^2*L / 4*w)
#
def calculate_conversion_probability(omega, g, B, L, m_eff_sq):
    if omega == 0:
        return 0

    # Amplitude term
    amplitude = (g * B * omega) / (m_eff_sq + 1e-9)

    # Phase term
    phase = (m_eff_sq * L) / (4 * omega)

    # Probability
    P = (amplitude**2) * (np.sin(phase)**2)
    return P

# --- 4. Run Simulation ---
print("Calculating P(omega) across energy range...")
omegas = np.linspace(omega_min, omega_max, N_POINTS)
probabilities = [calculate_conversion_probability(w, G_AGG, B_T, L_COH, m_eff_sq) for w in omegas]

print("Simulation complete.")

# --- 5. Plot Results ---
plt.figure(figsize=(12, 7))
plt.plot(omegas, probabilities, label=r'P(a $\leftrightarrow \gamma$)', color='cyan')
plt.title(r'ALP-Photon Conversion Probability vs. Photon Energy ($\omega$)')
plt.xlabel(r'Photon Energy ($\omega$) [arbitrary units]')
plt.ylabel(r'Conversion Probability P(a $\leftrightarrow \gamma$) [arbitrary units]')
plt.grid(True, which="both", ls="--", alpha=0.3)

# Add annotations from the manuscript figure
formula_tex = r'$P(\omega) \simeq \left(\frac{g_{a\gamma\gamma} B_T \omega}{m_{\mathrm{eff}}^2}\right)^2 \sin^2\left(\frac{m_{\mathrm{eff}}^2 L}{4\omega}\right)$'
mass_tex = r'$m_{\mathrm{eff}}^2 = m_a^2 - \omega_{pl}^2$'
plt.text(3, 5.0, formula_tex, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.text(3, 4.0, mass_tex, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.ylim(bottom=0)
plt.legend()

filename = 'L02_ALP_Photon_Conversion.png'
plt.savefig(filename)
print(f"\nSaved plot: {filename}")
plt.close()
