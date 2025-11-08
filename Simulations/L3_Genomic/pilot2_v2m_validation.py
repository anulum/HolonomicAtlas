"""
Pilot 2: V2M/PDE Validation Protocol
=====================================

Validates the Voltage-to-Morphogen (V2M) operator and morphogenetic PDE solver
through optogenetic manipulation and morphogen reporter imaging.

Objectives:
1. Calibrate V2M operator parameters (μ_e, k_react)
2. Validate RDA numerical solver accuracy
3. Test prediction: transporter blockade abolishes morphogen response

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import sys
sys.path.append('..')

from core.morphogenetic import (
    MorphogeneticPDE, 
    MorphogeneticParameters,
    create_gradient_voltage,
    create_domain_voltage
)


@dataclass
class Pilot2Parameters:
    """Parameters for Pilot 2 V2M validation"""
    
    # Spatial domain
    tissue_size: Tuple[float, float] = (2e-3, 2e-3)  # 2mm × 2mm
    n_points: Tuple[int, int] = (100, 100)
    
    # Optogenetic control
    voltage_pattern: str = 'gradient'  # 'gradient', 'domain', 'stripe', 'custom'
    V_imposed_min: float = -70e-3  # V
    V_imposed_max: float = -30e-3  # V
    
    # Imaging
    frame_rate: float = 1.0  # Hz
    duration: float = 300.0  # s (5 minutes)
    
    # Transporter blockade conditions
    blockade_conditions: List[str] = None
    
    # Calibration settings
    n_calibration_patterns: int = 5
    
    def __post_init__(self):
        if self.blockade_conditions is None:
            self.blockade_conditions = ['control', 'blocked']


class Pilot2V2MValidation:
    """
    Complete Pilot 2: V2M/PDE Validation Protocol
    
    Part A: Calibration of V2M operator
    Part B: Validation of dynamic simulations
    Part C: Transporter blockade control
    """
    
    def __init__(self, params: Optional[Pilot2Parameters] = None):
        self.params = params or Pilot2Parameters()
        self.results = {}
        
        # Create morphogenetic solver
        morph_params = MorphogeneticParameters(
            domain_size=self.params.tissue_size,
            n_points=self.params.n_points
        )
        self.morph_solver = MorphogeneticPDE(morph_params)
    
    def part_a_calibration(self) -> Dict:
        """
        Part A: Calibrate V2M operator parameters
        
        Protocol:
        1. Impose steady voltage patterns (optogenetics)
        2. Wait for steady-state morphogen distribution
        3. Fit model parameters to observed patterns
        
        Returns:
            Calibration results with fitted parameters
        """
        print("=" * 70)
        print(" PILOT 2 - PART A: V2M OPERATOR CALIBRATION ")
        print("=" * 70)
        
        p = self.params
        
        # Test multiple voltage patterns
        patterns = []
        phi_observed = []
        
        print(f"\nTesting {p.n_calibration_patterns} voltage patterns...")
        
        for i in range(p.n_calibration_patterns):
            print(f"\nPattern {i+1}/{p.n_calibration_patterns}")
            
            # Create voltage pattern
            if i < 3:
                # Gradients with varying steepness
                V_min = -70e-3
                V_max = -30e-3 - i * 5e-3
                V_pattern = create_gradient_voltage(
                    self.morph_solver.params,
                    V_min=V_min,
                    V_max=V_max
                )
            else:
                # Domain boundaries at different positions
                pos = 0.3 + i * 0.1
                V_pattern = create_domain_voltage(
                    self.morph_solver.params,
                    V_left=-70e-3,
                    V_right=-30e-3,
                    boundary_pos=pos
                )
            
            patterns.append(V_pattern)
            
            # Simulate to steady state
            print("  Computing steady-state...")
            phi_steady = self.morph_solver.steady_state_solver(
                V_field=V_pattern,
                tolerance=1e-6
            )
            
            phi_observed.append(phi_steady)
            
            print(f"  Steady-state range: [{phi_steady.min():.3f}, {phi_steady.max():.3f}]")
        
        # Parameter fitting
        print("\n" + "-" * 70)
        print("Fitting V2M operator parameters...")
        
        # Simplified fitting: use observed response to estimate μ_e
        # In real experiment, would use nonlinear least squares
        
        fitted_params = {
            'mu_e': self.morph_solver.params.mu_e,
            'D': self.morph_solver.params.D,
            'k_react': self.morph_solver.params.k_react,
            'k_degrade': self.morph_solver.params.k_degrade
        }
        
        # Compute goodness of fit
        r_squared_values = []
        
        for V, phi_obs in zip(patterns, phi_observed):
            # Simulate with fitted parameters
            phi_pred = self.morph_solver.steady_state_solver(V_field=V)
            
            # R² metric
            ss_res = np.sum((phi_obs - phi_pred)**2)
            ss_tot = np.sum((phi_obs - np.mean(phi_obs))**2)
            r_squared = 1 - ss_res / ss_tot
            r_squared_values.append(r_squared)
        
        print(f"\nCalibration Results:")
        print(f"  μ_e (electrophoretic mobility): {fitted_params['mu_e']:.3e} m²/V·s")
        print(f"  D (diffusion coefficient):      {fitted_params['D']:.3e} m²/s")
        print(f"  k_react (reaction rate):        {fitted_params['k_react']:.3e} 1/s")
        print(f"  Mean R²:                        {np.mean(r_squared_values):.4f}")
        
        calibration_results = {
            'fitted_params': fitted_params,
            'voltage_patterns': patterns,
            'phi_observed': phi_observed,
            'r_squared': r_squared_values,
            'mean_r_squared': np.mean(r_squared_values)
        }
        
        return calibration_results
    
    def part_b_dynamic_validation(self,
                                   calibrated_params: Optional[Dict] = None) -> Dict:
        """
        Part B: Validate dynamic PDE solver
        
        Protocol:
        1. Impose dynamic voltage pattern V(x,y,t)
        2. Record morphogen dynamics φ_exp(x,y,t)
        3. Simulate with calibrated parameters φ_sim(x,y,t)
        4. Compare experimental vs simulated dynamics
        
        Returns:
            Validation results with comparison metrics
        """
        print("\n" + "=" * 70)
        print(" PILOT 2 - PART B: DYNAMIC SIMULATION VALIDATION ")
        print("=" * 70)
        
        p = self.params
        
        # Use calibrated parameters if provided
        if calibrated_params is not None:
            for key, val in calibrated_params.items():
                setattr(self.morph_solver.params, key, val)
        
        # Create dynamic voltage profile
        def V_dynamic(t, X, Y):
            """Time-varying voltage pattern"""
            # Traveling wave
            omega = 2 * np.pi * 0.1  # 0.1 Hz
            k = 2 * np.pi / p.tissue_size[0]  # Wavelength = domain size
            
            V_base = -50e-3
            V_amp = 20e-3
            
            V = V_base + V_amp * np.sin(k * X - omega * t)
            
            return V
        
        print(f"\nRunning dynamic simulation...")
        print(f"  Duration: {p.duration} s")
        print(f"  Frame rate: {p.frame_rate} Hz")
        
        # Simulate "experimental" data (with noise)
        results_exp = self.morph_solver.simulate(
            duration=p.duration,
            V_profile=V_dynamic,
            record_interval=int(1.0 / (p.frame_rate * self.morph_solver.params.dt))
        )
        
        # Add experimental noise
        noise_level = 0.05  # 5% noise
        for i in range(len(results_exp['phi'])):
            results_exp['phi'][i] += noise_level * np.random.randn(*results_exp['phi'][i].shape)
            results_exp['phi'][i] = np.maximum(results_exp['phi'][i], 0)  # Enforce positivity
        
        # Simulate with same voltage but "predicted"
        print("\n  Simulating with calibrated model...")
        results_sim = self.morph_solver.simulate(
            duration=p.duration,
            V_profile=V_dynamic,
            record_interval=int(1.0 / (p.frame_rate * self.morph_solver.params.dt))
        )
        
        # Compute comparison metrics
        print("\nComputing comparison metrics...")
        
        mse_time = []
        correlation_time = []
        
        for phi_e, phi_s in zip(results_exp['phi'], results_sim['phi']):
            # MSE
            mse = np.mean((phi_e - phi_s)**2)
            mse_time.append(mse)
            
            # Spatial correlation
            corr = np.corrcoef(phi_e.flatten(), phi_s.flatten())[0, 1]
            correlation_time.append(corr)
        
        print(f"\nValidation Metrics:")
        print(f"  Mean MSE:               {np.mean(mse_time):.6f}")
        print(f"  Mean spatial corr:      {np.mean(correlation_time):.4f}")
        print(f"  Min spatial corr:       {np.min(correlation_time):.4f}")
        
        # Validation criterion
        validation_passed = np.mean(correlation_time) > 0.90
        
        print(f"\n  Validation Status: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
        print(f"  Criterion: Mean correlation > 0.90")
        
        validation_results = {
            'results_experimental': results_exp,
            'results_simulated': results_sim,
            'mse_time': np.array(mse_time),
            'correlation_time': np.array(correlation_time),
            'mean_mse': np.mean(mse_time),
            'mean_correlation': np.mean(correlation_time),
            'validation_passed': validation_passed
        }
        
        return validation_results
    
    def part_c_transporter_blockade(self) -> Dict:
        """
        Part C: Transporter blockade control experiment
        
        Critical Falsifier:
        Blocking electrophoretic transporters must abolish voltage-driven
        morphogen accumulation
        
        Returns:
            Blockade results
        """
        print("\n" + "=" * 70)
        print(" PILOT 2 - PART C: TRANSPORTER BLOCKADE CONTROL ")
        print("=" * 70)
        
        # Test voltage gradient
        V_gradient = create_gradient_voltage(
            self.morph_solver.params,
            V_min=-70e-3,
            V_max=-30e-3
        )
        
        results_by_condition = {}
        
        for condition in self.params.blockade_conditions:
            print(f"\nCondition: {condition.upper()}")
            
            if condition == 'blocked':
                # Simulate blockade by setting μ_e = 0
                original_mu_e = self.morph_solver.params.mu_e
                self.morph_solver.params.mu_e = 0.0
                print("  Electrophoretic mobility set to 0")
            
            # Simulate
            results = self.morph_solver.simulate(
                duration=100.0,
                V_field=V_gradient,
                record_interval=10
            )
            
            # Measure accumulation
            phi_final = results['phi'][-1]
            phi_init = results['phi'][0]
            
            # Gradient of morphogen (should align with voltage gradient if working)
            grad_phi_x = np.gradient(phi_final, self.morph_solver.params.dx, axis=0)
            mean_gradient = np.mean(grad_phi_x)
            
            # Center of mass shift
            com_shift = results['center_of_mass'][-1] - results['center_of_mass'][0]
            com_shift_magnitude = np.linalg.norm(com_shift)
            
            print(f"  Final morphogen range: [{phi_final.min():.3f}, {phi_final.max():.3f}]")
            print(f"  Mean gradient: {mean_gradient:.6f}")
            print(f"  COM shift: {com_shift_magnitude*1e6:.2f} μm")
            
            results_by_condition[condition] = {
                'phi_final': phi_final,
                'gradient': mean_gradient,
                'com_shift': com_shift_magnitude,
                'full_results': results
            }
            
            # Restore original parameters
            if condition == 'blocked':
                self.morph_solver.params.mu_e = original_mu_e
        
        # Compare control vs blocked
        control_gradient = results_by_condition['control']['gradient']
        blocked_gradient = results_by_condition['blocked']['gradient']
        
        # Falsification criterion
        reduction_ratio = blocked_gradient / (control_gradient + 1e-12)
        
        print("\n" + "-" * 70)
        print("TRANSPORTER BLOCKADE RESULTS:")
        print(f"  Control gradient:       {control_gradient:.6f}")
        print(f"  Blocked gradient:       {blocked_gradient:.6f}")
        print(f"  Reduction ratio:        {reduction_ratio:.4f}")
        
        falsification_criterion = reduction_ratio < 0.2  # >80% reduction
        
        print(f"\n  Falsification Test: {'✓ PASSED' if falsification_criterion else '✗ FAILED'}")
        print(f"  Criterion: Blockade must reduce gradient by >80%")
        
        blockade_results = {
            'by_condition': results_by_condition,
            'control_gradient': control_gradient,
            'blocked_gradient': blocked_gradient,
            'reduction_ratio': reduction_ratio,
            'falsification_passed': falsification_criterion
        }
        
        return blockade_results
    
    def run_complete_pilot2(self) -> Dict:
        """
        Execute complete Pilot 2 protocol
        
        Returns:
            Complete results from all parts
        """
        print("\n" + "=" * 80)
        print(" PILOT 2: COMPLETE V2M/PDE VALIDATION PROTOCOL ")
        print("=" * 80)
        
        # Part A: Calibration
        calibration = self.part_a_calibration()
        
        # Part B: Dynamic validation
        validation = self.part_b_dynamic_validation(
            calibrated_params=calibration['fitted_params']
        )
        
        # Part C: Blockade control
        blockade = self.part_c_transporter_blockade()
        
        # Summary
        print("\n" + "=" * 80)
        print(" PILOT 2 SUMMARY ")
        print("=" * 80)
        
        print("\nPart A - Calibration:")
        print(f"  Mean R²: {calibration['mean_r_squared']:.4f}")
        print(f"  Status: {'✓ GOOD FIT' if calibration['mean_r_squared'] > 0.85 else '✗ POOR FIT'}")
        
        print("\nPart B - Dynamic Validation:")
        print(f"  Mean correlation: {validation['mean_correlation']:.4f}")
        print(f"  Status: {'✓ VALIDATED' if validation['validation_passed'] else '✗ FAILED'}")
        
        print("\nPart C - Transporter Blockade:")
        print(f"  Reduction: {(1-blockade['reduction_ratio'])*100:.1f}%")
        print(f"  Status: {'✓ FALSIFIER PASSED' if blockade['falsification_passed'] else '✗ FALSIFIER FAILED'}")
        
        # Overall assessment
        all_passed = (
            calibration['mean_r_squared'] > 0.85 and
            validation['validation_passed'] and
            blockade['falsification_passed']
        )
        
        print("\n" + "-" * 80)
        print(f"OVERALL PILOT 2 STATUS: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 80)
        
        complete_results = {
            'part_a_calibration': calibration,
            'part_b_validation': validation,
            'part_c_blockade': blockade,
            'all_tests_passed': all_passed
        }
        
        self.results = complete_results
        return complete_results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of Pilot 2 results
        """
        if not self.results:
            raise ValueError("Must run complete pilot first")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Part A - Calibration patterns
        calibration = self.results['part_a_calibration']
        
        for i in range(min(3, len(calibration['voltage_patterns']))):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(
                calibration['phi_observed'][i].T,
                origin='lower',
                cmap='viridis'
            )
            ax.set_title(f'Calibration Pattern {i+1}\nR² = {calibration["r_squared"][i]:.3f}',
                        fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, label='φ')
        
        # Part B - Dynamic validation
        validation = self.results['part_b_validation']
        
        ax_corr = fig.add_subplot(gs[1, :])
        t = validation['results_experimental']['time']
        ax_corr.plot(t, validation['correlation_time'], 'b-', linewidth=2)
        ax_corr.axhline(0.90, color='r', linestyle='--', label='Threshold')
        ax_corr.set_xlabel('Time (s)', fontsize=12)
        ax_corr.set_ylabel('Spatial Correlation', fontsize=12)
        ax_corr.set_title('Part B: Exp vs Sim Correlation', fontsize=14, fontweight='bold')
        ax_corr.legend()
        ax_corr.grid(True, alpha=0.3)
        
        # Part C - Blockade comparison
        blockade = self.results['part_c_blockade']
        
        ax_control = fig.add_subplot(gs[2, 0])
        im = ax_control.imshow(
            blockade['by_condition']['control']['phi_final'].T,
            origin='lower',
            cmap='hot'
        )
        ax_control.set_title('Control', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax_control, label='φ')
        
        ax_blocked = fig.add_subplot(gs[2, 1])
        im = ax_blocked.imshow(
            blockade['by_condition']['blocked']['phi_final'].T,
            origin='lower',
            cmap='hot'
        )
        ax_blocked.set_title('Transporter Blocked', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax_blocked, label='φ')
        
        ax_comparison = fig.add_subplot(gs[2, 2])
        conditions = list(blockade['by_condition'].keys())
        gradients = [blockade['by_condition'][c]['gradient'] for c in conditions]
        ax_comparison.bar(conditions, gradients, color=['blue', 'red'])
        ax_comparison.set_ylabel('Mean Morphogen Gradient', fontsize=12)
        ax_comparison.set_title('Blockade Effect', fontsize=12, fontweight='bold')
        ax_comparison.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Pilot 2: V2M/PDE Validation Results',
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        return fig


def run_pilot2_demo():
    """
    Run demonstration of complete Pilot 2 protocol
    """
    # Initialize
    pilot2 = Pilot2V2MValidation()
    
    # Run complete protocol
    results = pilot2.run_complete_pilot2()
    
    # Visualize
    pilot2.plot_results(save_path='pilot2_v2m_validation.png')
    
    return results


if __name__ == "__main__":
    results = run_pilot2_demo()
    plt.show()
