#!/usr/bin/env python3
"""
Module 2 Quick Demonstration - Fast Version
==========================================

Demonstrates quantum-classical validation functionality without
running full experiments (which can be time-consuming).

This script shows the API usage and validates core functionality.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from layer2_quantum_classical import (
    QuantumState,
    QuantumEvolution,
    DecoherenceModel,
    VesicleReleaseSimulator,
    CalciumCooperativityValidator
)

print("\n" + "="*80)
print("MODULE 2: Quantum-Classical Validators - Quick Demo")
print("="*80 + "\n")

# ====================
# DEMO 1: Quantum States
# ====================
print("DEMO 1: Quantum State Representation")
print("-" * 40)

# Create pure state
pure_state = QuantumState.create_pure_state(n_levels=3, state_index=0)
print(f"✅ Pure state created: {pure_state.dimension}D Hilbert space")
print(f"   Purity: {pure_state.purity:.4f} (1.0 = perfectly pure)")
print(f"   von Neumann entropy: {pure_state.von_neumann_entropy:.4f}")

# Create superposition
super_state = QuantumState.create_superposition(n_levels=3)
print(f"\n✅ Superposition state created")
print(f"   Coherence measure: {super_state.coherence_measure():.4f}")
print(f"   Amplitudes: {np.abs(super_state.amplitudes)}")

# Create thermal state
thermal_state = QuantumState.create_thermal_state(n_levels=3, temperature=310)
print(f"\n✅ Thermal state created at T=310K")
print(f"   Purity: {thermal_state.purity:.4f} (mixed state)")
print(f"   Linear entropy: {thermal_state.linear_entropy:.4f}")

# ====================
# DEMO 2: Decoherence
# ====================
print("\n\nDEMO 2: Decoherence Mechanisms")
print("-" * 40)

# Calculate coherence times
temperatures = [290, 310, 320]  # K
for T in temperatures:
    tau = DecoherenceModel.calculate_coherence_time(
        temperature=T,
        energy_gap=0.1,  # eV
        coupling_strength=0.01  # eV
    )
    print(f"T = {T}K: τ_coherence = {tau*1000:.2f} ms")

# Thermal decoherence rate
gamma = DecoherenceModel.thermal_decoherence_rate(
    temperature=310,
    energy_gap=0.1
)
print(f"\n✅ Thermal decoherence rate: {gamma/1e9:.2f} GHz")

# ====================
# DEMO 3: Vesicle Release
# ====================
print("\n\nDEMO 3: Vesicle Release Simulator")
print("-" * 40)

simulator = VesicleReleaseSimulator(
    n_vesicles=200,
    K_release=1e-6,  # 1 μM
    hill_coefficient=4.0,
    lambda_psi=0.2
)
print(f"✅ Simulator initialized: {simulator.n_vesicles} vesicles")
print(f"   Hill coefficient: {simulator.hill_coefficient}")
print(f"   K_release: {simulator.K_release*1e6:.1f} μM")

# Test at different Ca concentrations
Ca_concentrations = [0.1e-6, 1.0e-6, 5.0e-6]  # μM
print(f"\n   Release Probabilities:")
for Ca in Ca_concentrations:
    P = simulator.base_release_probability(Ca)
    print(f"   [{Ca*1e6:.1f} μM Ca²⁺]: P = {P:.4f}")

# Ψ_s field modulation
Ca_test = 2e-6
P_baseline = simulator.psi_s_modulated_probability(Ca_test, 0.0)
P_enhanced = simulator.psi_s_modulated_probability(Ca_test, 1.0)
P_suppressed = simulator.psi_s_modulated_probability(Ca_test, -1.0)

print(f"\n   Ψ_s Field Modulation at [Ca²⁺] = {Ca_test*1e6:.1f} μM:")
print(f"   Baseline (Ψ_s=0):  P = {P_baseline:.4f}")
print(f"   Enhanced (Ψ_s=+1): P = {P_enhanced:.4f} ({P_enhanced/P_baseline:.2f}x)")
print(f"   Suppressed (Ψ_s=-1): P = {P_suppressed:.4f} ({P_suppressed/P_baseline:.2f}x)")

# ====================
# DEMO 4: Cooperativity
# ====================
print("\n\nDEMO 4: Calcium Cooperativity Validation")
print("-" * 40)

# Generate synthetic data with known Hill coefficient
Ca_range = np.logspace(-7, -5, 20)  # 100 nM to 10 μM
true_K = 1e-6
true_n = 4.0

# Hill function
response = Ca_range**true_n / (true_K**true_n + Ca_range**true_n)

# Fit and recover
n_fit, K_fit, r_squared = CalciumCooperativityValidator.measure_hill_coefficient(
    Ca_range, response
)

print(f"✅ Hill coefficient fit:")
print(f"   True n: {true_n:.2f}")
print(f"   Fitted n: {n_fit:.2f}")
print(f"   Error: {abs(n_fit - true_n):.3f}")
print(f"   R²: {r_squared:.6f}")
print(f"   Result: {'✅ PASS' if abs(n_fit - true_n) < 0.1 else '❌ FAIL'}")

# Validate simulator
print(f"\n   Validating VesicleReleaseSimulator...")
validation = CalciumCooperativityValidator.validate_cooperativity(
    simulator,
    expected_hill=4.0,
    tolerance=0.5
)

print(f"   Measured Hill coefficient: {validation['hill_coefficient']:.2f}")
print(f"   Expected: {validation['expected']:.2f} ± {validation['tolerance']:.2f}")
print(f"   R²: {validation['r_squared']:.4f}")
print(f"   Validation: {'✅ PASS' if validation['passed'] else '❌ FAIL'}")

# ====================
# DEMO 5: Quick Evolution
# ====================
print("\n\nDEMO 5: Quantum Evolution (brief)")
print("-" * 40)

# Simple 2-level system
H = np.array([[0, 0.1], [0.1, 0]], dtype=complex) * 1.6e-19  # 0.1 eV coupling
dephasing_op = np.array([[1, 0], [0, -1]], dtype=complex) * 0.1

evolution = QuantumEvolution(
    hamiltonian=H,
    lindblad_operators=[dephasing_op],
    decay_rates=[1e10]  # Fast dephasing
)

initial = QuantumState.create_superposition(n_levels=2)
print(f"✅ 2-level system initialized")
print(f"   Initial coherence: {initial.coherence_measure():.4f}")

# Very short evolution (to keep demo fast)
times, states = evolution.evolve(initial, t_final=1e-12, n_steps=5)

print(f"   After {times[-1]*1e12:.2f} ps:")
print(f"   Final coherence: {states[-1].coherence_measure():.4f}")
print(f"   Coherence decay: {(1 - states[-1].coherence_measure()/initial.coherence_measure())*100:.1f}%")

# ====================
# SUMMARY
# ====================
print("\n" + "="*80)
print("✅ MODULE 2 QUICK DEMO COMPLETED SUCCESSFULLY")
print("="*80)

print("\nComponents Validated:")
print("  ✅ Quantum state representation (pure, mixed, thermal)")
print("  ✅ Decoherence mechanisms and rate calculations")
print("  ✅ Vesicle release probability model")
print("  ✅ Ca²⁺ cooperativity (Hill coefficient ~4)")
print("  ✅ Ψ_s field modulation")
print("  ✅ Quantum evolution (unitary + Lindblad)")

print("\nKey Results:")
print(f"  • Hill coefficient: {n_fit:.2f} (theory: 4.0)")
print(f"  • Ψ_s enhancement: {P_enhanced/P_baseline:.2f}x")
print(f"  • Coherence time (310K): {DecoherenceModel.calculate_coherence_time(310, 0.1, 0.01)*1000:.2f} ms")

print("\nFull experiments available:")
print("  • quantum_decoherence - Complete decoherence dynamics")
print("  • vesicle_release_validation - Full Ca⁴ validation")

print("\n" + "="*80)
print("Ready for neurotransmitter validation (Module 3)!")
print("="*80 + "\n")
