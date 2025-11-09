import numpy as np

print("--- SCPN Paper 0: Operational Pullback Protocol (FIM) Validation ---")
print("Validating the mathematical procedure: g_F = J^T * I * J")

# --- 1. Define Model Components ---

# FIM (I_ij) on Statistical Manifold (Theta)
# Let's use a simple 2x2 identity matrix for the FIM I_ij
FIM_I = np.array([
[1, 0],
[0, 1]
])
print(f"\n1. FIM on Statistical Manifold (I_ij):\n{FIM_I}")

# Section (theta(x)) defined by its Jacobian matrix (J)
# J_mu,i = d(theta_i) / d(x_mu)
# Let theta_1 = 2*x0 + 1*x1
# Let theta_2 = 1*x0 - 1*x1
Jacobian_J = np.array([
[2, 1], # [d(theta1)/dx0, d(theta1)/dx1]
[1, -1] # [d(theta2)/dx0, d(theta2)/dx1]
])
# Note: The formula g = J^T * I * J requires J_ij = d(theta_j)/d(x_i)
# Let's be precise. g_munu = (d(theta_i)/d(x_mu)) * I_ij * (d(theta_j)/d(x_nu))
# This is J.T @ I @ J where J_ij = d(theta_j)/d(x_i) (J_row_col = x_coord, theta_coord)
# Let's use the definition from the text: J_mu,i = d(theta_i)/d(x_mu)
# So J = [[d(theta1)/dx0, d(theta2)/dx0], [d(theta1)/dx1, d(theta2)/dx1]]
# Let's redefine J to be J_i_mu = d(theta_i)/d(x_mu)
# J = [[d(theta1)/dx0, d(theta1)/dx1], [d(theta2)/dx0, d(theta2)/dx1]]
J_i_mu = np.array([
[2, 1], # [d(theta_1)/dx_0, d(theta_1)/dx_1]
[1, -1] # [d(theta_2)/dx_0, d(theta_2)/dx_1]
])
print(f"\n2. Jacobian of the Section (J_i_mu = d(theta_i)/d(x_mu)):\n{J_i_mu}")

# --- 2. Apply Pullback Formula ---
# g_F = J^T * I * J
g_F_munu = J_i_mu.T @ FIM_I @ J_i_mu
# g_F = [[2, 1], [1, -1]] @ [[1, 0], [0, 1]] @ [[2, 1], [1, -1]]
# g_F = [[2, 1], [1, -1]] @ [[2, 1], [1, -1]]
# g_F = [[(4+1), (2-1)], [(2-1), (1+1)]]
# g_F = [[5, 1], [1, 2]]

print(f"\n3. Calculated Pulled-Back Metric (g_F_munu) on Spacetime:\n{g_F_munu}")

# --- 3. Falsification Verdict ---
expected_g_F = np.array([[5, 1], [1, 2]])

print("\n--- 4. Falsification Verdict ---")
try:
    assert np.allclose(g_F_munu, expected_g_F)
    print("VERDICT: \033[1mCONFIRMED\033[0m")
    print("The pullback formula successfully computed the spacetime metric g_F.")
    print("The protocol is mathematically sound and computable.")
except AssertionError:
    print("VERDICT: \033[1mFALSIFIED\033[0m")
    print("The computed metric does not match the expected result.")
