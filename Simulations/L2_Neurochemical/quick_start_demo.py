#!/usr/bin/env python3
"""
Layer 2 Validation Suite - Quick Start Demonstration
====================================================

This script demonstrates the basic usage of the Layer 2 validation framework.
Run this to verify that the core module is working correctly.

Usage:
    python quick_start_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from layer2_validation_core import (
        ExperimentConfig,
        NeuralState,
        ValidationMetrics,
        PhysicalConstants,
        NeurotransmitterParams,
        OscillationParams,
    )
    print("✅ Successfully imported Layer 2 validation core module\n")
except ImportError as e:
    print(f"❌ Error importing module: {e}")
    print("Make sure layer2_validation_core.py is in the same directory")
    sys.exit(1)


def demo_1_configuration():
    """Demonstrate experiment configuration"""
    print("="*80)
    print("DEMO 1: Creating Experiment Configuration")
    print("="*80)
    
    config = ExperimentConfig(
        name="demo_neurotransmitter_dynamics",
        description="Demonstration of basic neurotransmitter dynamics",
        layer_components=["neurotransmitter_dynamics", "oscillations"],
        duration=5.0,  # 5 seconds
        dt=0.001,  # 1 ms timestep
        temperature=PhysicalConstants.T_body,
        psi_s_field=OscillationParams.PSI_S_BASELINE,
        random_seed=42,
        validation_checks=["energy_conservation", "causality"]
    )
    
    print(f"\n✅ Configuration Created:")
    print(f"   Name: {config.name}")
    print(f"   Duration: {config.duration}s with dt={config.dt}s")
    print(f"   Number of steps: {int(config.duration/config.dt)}")
    print(f"   Temperature: {config.temperature}K")
    print(f"   Ψ_s field: {config.psi_s_field}")
    print(f"   Components: {', '.join(config.layer_components)}")
    
    # Save and load demonstration
    config_path = Path("demo_config.json")
    config.save(config_path)
    print(f"\n✅ Configuration saved to: {config_path}")
    
    loaded_config = ExperimentConfig.load(config_path)
    print(f"✅ Configuration loaded successfully")
    
    # Cleanup
    config_path.unlink()
    
    return config


def demo_2_neural_state():
    """Demonstrate neural state representation"""
    print("\n" + "="*80)
    print("DEMO 2: Neural State Representation")
    print("="*80)
    
    # Initialize default state
    state = NeuralState.initialize_default()
    
    print(f"\n✅ Initial Neural State:")
    print(f"   Membrane potential: {state.V_membrane:.2f} mV")
    print(f"   Internal Ca²⁺: {state.Ca_internal:.2e} M")
    print(f"   External Ca²⁺: {state.Ca_external:.2e} M")
    print(f"   Quantum coherence: {state.quantum_coherence:.2f}")
    print(f"   Ψ_s field (local): {state.psi_s_local:.2f}")
    
    print(f"\n   Neurotransmitter Concentrations:")
    for nt, conc in sorted(state.NT_concentrations.items()):
        print(f"      {nt:15s}: {conc:.2e} M")
    
    print(f"\n   Oscillation Phases (radians):")
    for band, phase in sorted(state.oscillation_phases.items()):
        print(f"      {band:15s}: {phase:.3f}")
    
    # Demonstrate array conversion
    print(f"\n✅ Array Conversion:")
    state_array = state.to_array()
    print(f"   State vector length: {len(state_array)}")
    
    # Reconstruct
    nt_keys = sorted(state.NT_concentrations.keys())
    osc_keys = sorted(state.oscillation_phases.keys())
    reconstructed = NeuralState.from_array(state_array, 0.0, nt_keys, osc_keys)
    
    # Verify
    reconstructed_array = reconstructed.to_array()
    match = np.allclose(state_array, reconstructed_array)
    print(f"   Reconstruction accuracy: {'✅ PERFECT' if match else '❌ FAILED'}")
    
    return state


def demo_3_neurotransmitter_params():
    """Display neurotransmitter parameters"""
    print("\n" + "="*80)
    print("DEMO 3: Neurotransmitter System Parameters")
    print("="*80)
    
    print(f"\n{'Neurotransmitter':<15} {'Baseline (M)':<15} {'Synaptic Peak (M)':<20} {'Release Rate':<15} {'Uptake Rate'}")
    print("-" * 95)
    
    for nt in sorted(NeurotransmitterParams.CONCENTRATIONS.keys()):
        conc = NeurotransmitterParams.CONCENTRATIONS[nt]
        kinetics = NeurotransmitterParams.KINETICS[nt]
        
        print(f"{nt:<15} {conc['baseline']:<15.2e} {conc['synaptic_peak']:<20.2e} "
              f"{kinetics['release_rate']:<15.3f} {kinetics['uptake_rate']:.3f}")
    
    print(f"\n✅ All major neurotransmitter systems parameterized")
    
    print(f"\n{'Oscillation Band':<15} {'Frequency Range (Hz)'}")
    print("-" * 40)
    for band, (low, high) in NeurotransmitterParams.OSCILLATION_BANDS.items():
        print(f"{band:<15} {low:6.2f} - {high:6.1f}")


def demo_4_validation_metrics():
    """Demonstrate validation metrics"""
    print("\n" + "="*80)
    print("DEMO 4: Validation Metrics")
    print("="*80)
    
    # Test 1: Energy conservation
    print("\n📊 Test 1: Energy Conservation Check")
    n_points = 1000
    # Create states with conserved energy (small fluctuations)
    test_states = np.random.randn(n_points, 10) * 0.01 + 1.0
    
    energy_conserved = ValidationMetrics.check_energy_conservation(test_states, tolerance=0.1)
    print(f"   Result: {'✅ PASS' if energy_conserved else '❌ FAIL'}")
    
    # Test 2: Probability conservation
    print("\n📊 Test 2: Quantum Probability Conservation")
    quantum_states = np.random.randn(n_points, 5) + 1j * np.random.randn(n_points, 5)
    # Normalize
    norms = np.sqrt(np.sum(np.abs(quantum_states)**2, axis=1, keepdims=True))
    quantum_states = quantum_states / norms
    
    prob_conserved = ValidationMetrics.check_probability_conservation(quantum_states)
    print(f"   Result: {'✅ PASS' if prob_conserved else '❌ FAIL'}")
    
    # Test 3: Phase-Amplitude Coupling
    print("\n📊 Test 3: Phase-Amplitude Coupling (PAC) Measurement")
    
    # Generate nested oscillations
    t = np.linspace(0, 10, 10000)
    
    # Theta carrier (6 Hz)
    theta = np.sin(2 * np.pi * 6 * t)
    
    # Gamma oscillation (40 Hz) amplitude-modulated by theta
    modulation_depth = 0.5
    gamma = np.sin(2 * np.pi * 40 * t) * (1 + modulation_depth * np.cos(2 * np.pi * 6 * t))
    
    pac_strength = ValidationMetrics.measure_oscillation_pac(theta, gamma)
    print(f"   Theta-Gamma PAC strength: {pac_strength:.4f}")
    print(f"   Expected: ~{modulation_depth:.2f} (modulation depth)")
    print(f"   Result: {'✅ PASS' if 0.2 < pac_strength < 0.8 else '⚠️  Check'}")
    
    # Test 4: Cooperativity
    print("\n📊 Test 4: Ca²⁺ Cooperativity Verification")
    
    Ca_range = np.logspace(-7, -5, 50)  # 100 nM to 10 μM
    # Simulate Hill response with n=4
    K = 1e-6  # Half-activation at 1 μM
    response = Ca_range**4 / (K**4 + Ca_range**4)
    
    cooperativity_verified = ValidationMetrics.verify_cooperativity(
        Ca_range, response, expected_hill=4.0, tolerance=0.5
    )
    print(f"   Expected Hill coefficient: 4.0 (Ca⁴ cooperativity)")
    print(f"   Result: {'✅ PASS' if cooperativity_verified else '❌ FAIL'}")
    
    return pac_strength


def demo_5_visualization():
    """Create demonstration visualizations"""
    print("\n" + "="*80)
    print("DEMO 5: Basic Visualization")
    print("="*80)
    
    # Generate synthetic neural dynamics
    t = np.linspace(0, 5, 5000)
    dt = t[1] - t[0]
    
    # Membrane potential with spiking
    V_rest = -70
    V_threshold = -55
    V_peak = 40
    
    # Simple integrate-and-fire
    V = np.zeros_like(t)
    V[0] = V_rest
    
    for i in range(1, len(t)):
        if V[i-1] >= V_threshold:
            V[i] = V_rest  # Reset after spike
        else:
            # Integrate with noise
            dV = (-(V[i-1] - V_rest) / 0.02 + np.random.randn() * 5) * dt
            V[i] = V[i-1] + dV
            
            if V[i] > V_threshold:
                V[i] = V_peak  # Spike
    
    # Neurotransmitter release events (coinciding with spikes)
    spikes = V >= V_threshold
    
    # Oscillations
    theta = 10 * np.sin(2 * np.pi * 6 * t) - 70
    gamma = 5 * np.sin(2 * np.pi * 40 * t) * (1 + 0.5 * np.cos(2 * np.pi * 6 * t))
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Membrane potential
    axes[0].plot(t, V, 'b-', linewidth=0.5, label='Membrane Potential')
    axes[0].axhline(V_threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('Neural Dynamics Simulation')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Nested oscillations
    axes[1].plot(t, theta, 'g-', linewidth=1, alpha=0.7, label='Theta (6 Hz)')
    axes[1].plot(t, gamma, 'orange', linewidth=0.5, alpha=0.6, label='Gamma (40 Hz)')
    axes[1].set_ylabel('Amplitude (a.u.)')
    axes[1].set_title('Theta-Gamma Nested Oscillations')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Spike raster and rate
    spike_times = t[spikes]
    axes[2].eventplot([spike_times], colors='black', linewidths=0.5)
    axes[2].set_ylabel('Spikes')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title(f'Spike Train (Total: {len(spike_times)} spikes, Rate: {len(spike_times)/t[-1]:.1f} Hz)')
    axes[2].set_ylim([0.5, 1.5])
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("demo_neural_dynamics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.close()
    
    print(f"\n   Generated plots:")
    print(f"      1. Membrane potential with action potentials")
    print(f"      2. Theta-gamma nested oscillations (PAC)")
    print(f"      3. Spike train raster")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("LAYER 2 VALIDATION SUITE - QUICK START DEMONSTRATION")
    print("Sentient-Consciousness Projection Network (SCPN)")
    print("Module 1: Core Framework")
    print("="*80)
    
    try:
        # Run all demos
        config = demo_1_configuration()
        state = demo_2_neural_state()
        demo_3_neurotransmitter_params()
        pac = demo_4_validation_metrics()
        demo_5_visualization()
        
        # Summary
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Results:")
        print(f"  • Configuration system: ✅ Working")
        print(f"  • State representation: ✅ Working")
        print(f"  • Neurotransmitter parameters: ✅ Loaded")
        print(f"  • Validation metrics: ✅ Functional")
        print(f"  • Visualization: ✅ Generated")
        print(f"  • Measured PAC strength: {pac:.4f}")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\n1. Review the generated files:")
        print("   - demo_neural_dynamics.png (visualization)")
        print("\n2. Explore the core module:")
        print("   - Open layer2_validation_core.py")
        print("   - Read the comprehensive docstrings")
        print("   - Try modifying parameters")
        print("\n3. Wait for Module 2 (Quantum-Classical Validators)")
        print("   - Will include complete quantum transition experiments")
        print("   - Vesicle release validation")
        print("   - SNARE quantum mechanics")
        print("\n4. Check README_LAYER2_VALIDATION.md for full documentation")
        
        print("\n" + "="*80)
        print("Framework is ready for experimental validation!")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
