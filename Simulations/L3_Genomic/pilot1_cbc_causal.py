"""
Pilot 1: CBC Causal Test
==========================

Validates temporal precedence of CBC cascade through synchronized multi-modal measurements.

Critical Prediction: t_spin < t_field < t_channel < t_voltage < t_chromatin

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import sys
sys.path.append('..')

from core.cbc_cascade import CBCCascade, CBCParameters, create_pulse_profile
from core.ciss_mechanism import CISSModel, CISSParameters


@dataclass
class CBCCausalTestParameters:
    """Parameters for CBC causal test"""
    
    # Test conditions
    chirality: str = 'L-DNA'  # 'L-DNA' or 'D-DNA'
    magnetic_field: float = 50e-6  # External B-field (T)
    orientation_angle: float = 0.0  # Field orientation (rad)
    
    # Pulse parameters
    pulse_start: float = 0.1  # s
    pulse_width: float = 0.05  # s
    pulse_amplitude: float = 1e-5  # A/m²
    
    # Measurement parameters
    detection_threshold: float = 0.1  # Fraction of max signal
    
    # Simulation
    duration: float = 1.0  # s
    dt: float = 1e-6  # s (1 μs resolution)


class CBCCausalTest:
    """
    Complete CBC Causal Test Protocol
    
    Implements synchronized multi-modal measurements:
    1. ESR/NV magnetometry → Spin current (S)
    2. Patch-clamp → Channel gating (M1)
    3. Voltage imaging → Membrane voltage (M2)
    4. ATAC-seq → Chromatin accessibility (Y)
    """
    
    def __init__(self, params: Optional[CBCCausalTestParameters] = None):
        self.params = params or CBCCausalTestParameters()
        self.results = {}
    
    def run_test(self) -> Dict:
        """
        Execute complete CBC causal test
        
        Returns:
            Dictionary with all measurements and validation results
        """
        p = self.params
        
        print(f"Running CBC Causal Test")
        print(f"  Chirality: {p.chirality}")
        print(f"  External B-field: {p.magnetic_field*1e6:.1f} μT")
        print(f"  Orientation: {np.degrees(p.orientation_angle):.1f}°")
        print()
        
        # 1. Setup CBC cascade
        chi = +1.0 if p.chirality == 'L-DNA' else -1.0
        cbc_params = CBCParameters(chi=chi)
        cascade = CBCCascade(cbc_params)
        
        # 2. Setup electron transfer pulse
        pulse_profile = create_pulse_profile(
            t_start=p.pulse_start,
            t_width=p.pulse_width,
            amplitude=p.pulse_amplitude
        )
        
        # 3. Run simulation with full stage recording
        print("Simulating CBC cascade...")
        results = cascade.simulate(
            duration=p.duration,
            dt=p.dt,
            j_et_profile=pulse_profile,
            record_stages=True
        )
        
        # 4. Detect onset times
        print("Detecting stage onset times...")
        precedence = cascade.validate_temporal_precedence(
            threshold=p.detection_threshold
        )
        
        # 5. Compile results
        self.results = {
            'simulation': results,
            'precedence': precedence,
            'parameters': p,
            'cascade_params': cbc_params
        }
        
        # 6. Print validation results
        self._print_results()
        
        return self.results
    
    def _print_results(self):
        """Print test results"""
        prec = self.results['precedence']
        
        print("\n" + "="*60)
        print("CBC CAUSAL TEST RESULTS")
        print("="*60)
        
        print("\nDetected Onset Times:")
        print(f"  t_spin:      {prec['t_spin']*1e3:8.3f} ms")
        print(f"  t_field:     {prec['t_field']*1e3:8.3f} ms")
        print(f"  t_channel:   {prec['t_channel']*1e3:8.3f} ms")
        print(f"  t_voltage:   {prec['t_voltage']*1e3:8.3f} ms")
        print(f"  t_chromatin: {prec['t_chromatin']*1e3:8.3f} ms")
        
        print("\nTemporal Precedence:")
        if prec['temporal_precedence_valid']:
            print("  ✓ VALIDATED: All stages in correct temporal order")
        else:
            print("  ✗ FALSIFIED: Temporal ordering violated")
        
        print("\nTime Intervals (Δt between stages):")
        times = [
            prec['t_spin'],
            prec['t_field'],
            prec['t_channel'],
            prec['t_voltage'],
            prec['t_chromatin']
        ]
        
        for i in range(len(times)-1):
            if not np.isnan(times[i]) and not np.isnan(times[i+1]):
                dt = (times[i+1] - times[i]) * 1e3
                print(f"  Δt_{i+1}: {dt:8.3f} ms")
        
        print("\n" + "="*60)
    
    def run_chirality_control(self) -> Dict:
        """
        Control experiment: Test both L-DNA and D-DNA
        
        Critical Prediction: Sign reversal in ΔA
        
        Returns:
            Dictionary comparing L-DNA vs D-DNA results
        """
        print("\nRunning Chirality Control Test...")
        print("="*60)
        
        # Test L-DNA
        print("\n1. Testing L-DNA...")
        self.params.chirality = 'L-DNA'
        results_l = self.run_test()
        
        # Test D-DNA
        print("\n2. Testing D-DNA...")
        self.params.chirality = 'D-DNA'
        results_d = self.run_test()
        
        # Compare
        sim_l = results_l['simulation']
        sim_d = results_d['simulation']
        
        delta_a_l = sim_l['chromatin_accessibility'][-1] - sim_l['chromatin_accessibility'][0]
        delta_a_d = sim_d['chromatin_accessibility'][-1] - sim_d['chromatin_accessibility'][0]
        
        sign_reversed = np.sign(delta_a_l) == -np.sign(delta_a_d)
        
        print("\n" + "="*60)
        print("CHIRALITY CONTROL RESULTS")
        print("="*60)
        print(f"\nΔA (L-DNA): {delta_a_l:+.4f}")
        print(f"ΔA (D-DNA): {delta_a_d:+.4f}")
        print(f"\nSign Reversal: {'✓ VALIDATED' if sign_reversed else '✗ FALSIFIED'}")
        print("="*60)
        
        return {
            'l_dna_results': results_l,
            'd_dna_results': results_d,
            'delta_a_l': delta_a_l,
            'delta_a_d': delta_a_d,
            'sign_reversed': sign_reversed
        }
    
    def plot_cascade_dynamics(self, save_path: Optional[str] = None):
        """
        Plot CBC cascade temporal dynamics
        
        Args:
            save_path: Path to save figure (optional)
        """
        if not self.results:
            raise ValueError("Must run test first")
        
        sim = self.results['simulation']
        prec = self.results['precedence']
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        
        t = sim['time'] * 1e3  # Convert to ms
        
        # Stage 1: Spin current
        axes[0].plot(t, sim['p_ciss'], 'b-', linewidth=2)
        axes[0].axvline(prec['t_spin']*1e3, color='r', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('P_CISS', fontsize=12)
        axes[0].set_title('Stage 1: CISS Spin Generation', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Stage 2: Effective field
        axes[1].plot(t, sim['b_eff']*1e6, 'g-', linewidth=2)
        axes[1].axvline(prec['t_field']*1e3, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('B_eff (μT)', fontsize=12)
        axes[1].set_title('Stage 2: Effective Magnetic Field', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Stage 3: Channel opening
        axes[2].plot(t, sim['p_open'], 'm-', linewidth=2)
        axes[2].axvline(prec['t_channel']*1e3, color='r', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('P_open', fontsize=12)
        axes[2].set_title('Stage 3: Ion Channel Modulation', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Stage 4: Membrane voltage
        axes[3].plot(t, sim['v_mem']*1e3, 'c-', linewidth=2)
        axes[3].axvline(prec['t_voltage']*1e3, color='r', linestyle='--', alpha=0.5)
        axes[3].set_ylabel('V_mem (mV)', fontsize=12)
        axes[3].set_title('Stage 4a: Membrane Voltage', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # Stage 5: Chromatin accessibility
        axes[4].plot(t, sim['chromatin_accessibility'], 'orange', linewidth=2)
        axes[4].axvline(prec['t_chromatin']*1e3, color='r', linestyle='--', alpha=0.5)
        axes[4].set_ylabel('Accessibility', fontsize=12)
        axes[4].set_title('Stage 4b: Chromatin Remodeling', fontsize=14, fontweight='bold')
        axes[4].set_xlabel('Time (ms)', fontsize=12)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        else:
            plt.show()
        
        return fig


def run_full_pilot1():
    """
    Run complete Pilot 1 experimental protocol
    """
    print("="*70)
    print(" PILOT 1: CBC CAUSAL TEST - COMPLETE EXPERIMENTAL PROTOCOL ")
    print("="*70)
    
    # Initialize test
    test = CBCCausalTest()
    
    # 1. Basic causal test
    print("\n### EXPERIMENT 1: Basic CBC Causal Test ###\n")
    results = test.run_test()
    
    # 2. Chirality control
    print("\n\n### EXPERIMENT 2: Chirality Control ###\n")
    chirality_results = test.run_chirality_control()
    
    # 3. Visualize
    print("\n\n### Generating Visualization ###\n")
    test.plot_cascade_dynamics(save_path='pilot1_cbc_cascade.png')
    
    # 4. Summary
    print("\n\n" + "="*70)
    print(" PILOT 1 COMPLETE ")
    print("="*70)
    print("\nKey Findings:")
    
    if results['precedence']['temporal_precedence_valid']:
        print("  ✓ Temporal precedence VALIDATED")
    else:
        print("  ✗ Temporal precedence FALSIFIED")
    
    if chirality_results['sign_reversed']:
        print("  ✓ Chirality dependence VALIDATED")
    else:
        print("  ✗ Chirality dependence FALSIFIED")
    
    print("\nNext Steps:")
    print("  1. Experimental validation with real biological samples")
    print("  2. Pilot 2: V2M transduction validation")
    print("  3. Pilot 3: Quantum coherence detection")
    print("="*70)
    
    return {
        'causal_test': results,
        'chirality_control': chirality_results
    }


if __name__ == "__main__":
    # Run complete Pilot 1 protocol
    results = run_full_pilot1()
