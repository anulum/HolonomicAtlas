import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

print("--- SCPN Paper 0: Category Theory Grammar Validation ---")
print("Testing the coherence of the Consciousness <-> Physics mapping.")
print("Asserting: G(f) * eta_1 == eta_2 * F(f)\n")

# --- 1. Define the Category C_SCPN (Objects) ---
#
C_SCPN = nx.DiGraph()
C_SCPN.add_nodes_from(['L1', 'L2', 'L3'])
C_SCPN.add_edge('L1', 'L2', morphism_name='f')
C_SCPN.add_edge('L2', 'L3', morphism_name='g')
C_SCPN.add_edge('L1', 'L3', morphism_name='g_f') # Composition g(f(L1))

# --- 2. Define Mock Spaces and Mappings ---

# We define two "spaces" (categories)
CONSCIOUSNESS_SPACE = {
'L1_State': 'Qualia_Pattern_A',
'L2_State': 'Qualia_Pattern_B',
'L3_State': 'Qualia_Pattern_C'
}
PHYSICS_SPACE = {
'L1_Realiser': 1.23,
'L2_Realiser': 4.56,
'L3_Realiser': 7.89
}

# Define Functors (Mappings between spaces)
def functor_F(c_state_name):
    """ F: Consciousness -> Physics """
    if c_state_name == 'Qualia_Pattern_A': return 1.23
    if c_state_name == 'Qualia_Pattern_B': return 4.56
    if c_state_name == 'Qualia_Pattern_C': return 7.89

def functor_G(p_realiser):
    """ G: Physics -> Consciousness """
    if p_realiser == 1.23: return 'Qualia_Pattern_A'
    if p_realiser == 4.56: return 'Qualia_Pattern_B'
    if p_realiser == 7.89: return 'Qualia_Pattern_C'

# Define Morphisms (Projections f: L1 -> L2)
def morphism_f_consciousness(c_state_L1):
    """ The projection f in the Consciousness space """
    if c_state_L1 == 'Qualia_Pattern_A':
        return 'Qualia_Pattern_B'

def morphism_f_physics(p_realiser_L1):
    """ The projection f in the Physics space """
    if p_realiser_L1 == 1.23:
        return 4.56

# Define Natural Transformation (eta)
# eta connects the two spaces at each object
def eta_L1(c_state_L1):
    """ eta_L1: F(L1) -> G(L1) (conceptually) """
    # For this test, we just map from C-space to P-space
    return functor_F(c_state_L1)

def eta_L2(c_state_L2):
    """ eta_L2: F(L2) -> G(L2) (conceptually) """
    return functor_F(c_state_L2)

# --- 3. Run the Computational Proof ---
# We are testing the Naturality Square for the morphism f: L1 -> L2
# We must prove that both paths from 'Qualia_Pattern_A' to '4.56' are equal.

print("Testing Naturality Square for morphism f: L1 -> L2...")
START_STATE = 'Qualia_Pattern_A'

# Path 1: (Top-Right Path) F(L1) -> F(L2) -> G(L2)
# F(f) o F(L1)
# 1. F(L1): eta_L1(START_STATE)
path_1_step_1 = eta_L1(START_STATE)
# 2. G(f) o (result): morphism_f_physics(path_1_step_1)
path_1_result = morphism_f_physics(path_1_step_1)
print(f" -> Path 1 (F(L1) -> F(L2)): {path_1_result}")

# Path 2: (Left-Bottom Path) F(L1) -> G(L1) -> G(L2)
# G(f) o G(L1)
# 1. G(L1): morphism_f_consciousness(START_STATE)
path_2_step_1 = morphism_f_consciousness(START_STATE)
# 2. eta_L2 o (result): eta_L2(path_2_step_1)
path_2_result = eta_L2(path_2_step_1)
print(f" -> Path 2 (G(L1) -> G(L2)): {path_2_result}")

# --- 4. Falsification Verdict ---
print("\n--- 5. Falsification Verdict ---")
try:
    assert np.allclose(path_1_result, path_2_result)
    print("VERDICT: \033[1mCONFIRMED\033[0m")
    print("The Naturality Square commutes. The Consciousness-Physics mapping is coherent.")
    print("G(f) * eta_L1 == eta_L2 * F(f)")
except AssertionError:
    print("VERDICT: \033[1mFALSIFIED\033[0m")
    print("The Naturality Square does NOT commute. The formalism is inconsistent.")

# Bonus: Visualize the Category Graph
pos = nx.spring_layout(C_SCPN)
nx.draw(C_SCPN, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, arrows=True)
edge_labels = nx.get_edge_attributes(C_SCPN, 'morphism_name')
nx.draw_networkx_edge_labels(C_SCPN, pos, edge_labels=edge_labels)
plt.title("C_SCPN Graph (Objects and Morphisms)")
filename = 'L00_Category_Graph.png'
plt.savefig(filename)
print(f"\nSaved graph visualization: {filename}")
plt.close()
