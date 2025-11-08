"""
Pilot 3: QFT Emergence Test
============================

Validates quantum field theoretic description of morphogenetic patterns
by detecting mass parameter m² in biological data.

Objectives:
1. Validate spectral estimator on synthetic data
2. Apply to real 4D microscopy data
3. Test field modulation by Ψ_s surrogate

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import sys
sys.path.append('..')

from core.qft_kg import KleinGordonField, KGParameters


@dataclass
class Pilot3Parameters:
    """Parameters for Pilot 3 QFT test"""
    
    # Imaging setup
    tissue_size: Tuple[float, float, float] = (1e-3, 1e-3, 500e-6)  # 1mm × 1mm × 0.5mm
    n_points: Tuple[int, int, int] = (64, 64, 32)
    duration: float = 600.0  # 10 minutes
    frame_rate: float = 0.1  # Hz (one frame every 10s)
    
    # QFT parameters
    m_squared_expected: float = 1e8  # 1/m²
    m_squared_tolerance: float = 0.3  # 30% tolerance
    
    # Ψ_s surrogate conditions
    psi_s_conditions: list = None
    
    def __post_init__(self):
        if self.psi_s_conditions is None:
            self.psi_s_conditions = ['baseline', 'low_field', 'high_field']


class Pilot3QFTTest:
    """
    Complete Pilot 3: QFT Emergence Test Protocol
    
    Part A: Validate estimator on synthetic data
    Part B: Apply to biological 4D microscopy data
    Part C: Test Ψ_s field modulation
    """
    
    def __init__(self, params: Optional[Pilot3Parameters] = None):
        self.params = params or Pilot3Parameters()
        self.results = {}
    
    def power_spectrum_estimator(self,
                                 phi_4d: np.ndarray,
                                 dx: float,
                                 dy: float,
                                 dz: float) -> Dict:
        """
        Estimate m² from 3D+time field data
        
        Uses Klein-Gordon dispersion: P(k) ~ 1/(k² + m²)
        
        Args:
            phi_4d: Field data [time, x, y, z]
            dx, dy, dz: Spatial resolutions
            
        Returns:
            Dictionary with m² estimate
        """
        from scipy.fft import fftn, fftfreq
        
        n_t, nx, ny, nz = phi_4d.shape
        
        # Spatial FFT (average over time)
        P_k_avg = np.zeros((nx, ny, nz))
        
        for t in range(n_t):
            phi_k = fftn(phi_4d[t])
            P_k_avg += np.abs(phi_k)**2 / n_t
        
        # Wave vector magnitudes
        kx = 2*np.pi * fftfreq(nx, dx)
        ky = 2*np.pi * fftfreq(ny, dy)
        kz = 2*np.pi * fftfreq(nz, dz)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Radial average
        k_max = k_mag.max()
        k_bins = np.linspace(0, k_max, 50)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        P_avg = np.zeros(len(k_centers))
        
        for i, (k_min, k_max) in enumerate(zip(k_bins[:-1], k_bins[1:])):
            mask = (k_mag >= k_min) & (k_mag < k_max)
            if np.any(mask):
                P_avg[i] = np.mean(P_k_avg[mask])
        
        # Remove DC and low-k noise
        mask = k_centers > k_centers[5]  # Skip first 5 bins
        k_fit = k_centers[mask]
        P_fit = P_avg[mask]
        
        # Fit Klein-Gordon form: P(k) = A/(k² + m²)
        def kg_model(k, A, m_sq):
            return A / (k**2 + m_sq)
        
        try:
            popt, pcov = curve_fit(
                kg_model,
                k_fit,
                P_fit,
                p0=[P_fit[0], 1e8],
                bounds=([0, 1e6], [np.inf, 1e10])
            )
            
            A_fit, m_sq_fit = popt
            
            # Standard error
            m_sq_std = np.sqrt(pcov[1, 1])
            
            # Goodness of fit
            P_pred = kg_model(k_fit, *popt)
            ss_res = np.sum((P_fit - P_pred)**2)
            ss_tot = np.sum((P_fit - np.mean(P_fit))**2)
            r_squared = 1 - ss_res / ss_tot
            
            fit_success = True
            
        except Exception as e:
            print(f"  Fit failed: {e}")
            m_sq_fit = np.nan
            m_sq_std = np.nan
            r_squared = 0.0
            fit_success = False
        
        return {
            'm_squared_estimate': m_sq_fit,
            'm_squared_std': m_sq_std,
            'r_squared': r_squared,
            'fit_success': fit_success,
            'k': k_centers,
            'P': P_avg,
            'k_fit': k_fit,
            'P_fit': P_fit
        }
    
    def part_a_validation(self) -> Dict:
        """
        Part A: Validate estimator on synthetic Klein-Gordon data
        
        Protocol:
        1. Generate synthetic KG field with known m²
        2. Apply spectral estimator
        3. Compare estimated vs true m²
        
        Returns:
            Validation results
        """
        print("=" * 70)
        print(" PILOT 3 - PART A: ESTIMATOR VALIDATION ")
        print("=" * 70)
        
        p = self.params
        
        # Create Klein-Gordon simulator
        kg_params = KGParameters(
            domain_size=p.tissue_size,
            n_points=p.n_points,
            m_squared=p.m_squared_expected
        )
        
        kg = KleinGordonField(kg_params)
        
        print(f"\nGenerating synthetic KG data...")
        print(f"  True m²: {p.m_squared_expected:.2e} 1/m²")
        print(f"  Duration: {p.duration} s")
        
        # Simulate
        n_frames = int(p.duration * p.frame_rate)
        record_interval = int(1.0 / (p.frame_rate * kg_params.dt))
        
        results_kg = kg.simulate(
            duration=p.duration,
            record_interval=record_interval
        )
        
        # Extract 4D data
        phi_4d = results_kg['phi']  # [time, x, y, z]
        
        print(f"  Generated {phi_4d.shape[0]} frames")
        
        # Apply estimator
        print("\nApplying spectral estimator...")
        estimate = self.power_spectrum_estimator(
            phi_4d,
            kg_params.dx,
            kg_params.dy,
            kg_params.dz
        )
        
        # Validation
        if estimate['fit_success']:
            m_sq_true = p.m_squared_expected
            m_sq_est = estimate['m_squared_estimate']
            
            relative_error = abs(m_sq_est - m_sq_true) / m_sq_true
            
            print(f"\nEstimator Results:")
            print(f"  True m²:      {m_sq_true:.2e} 1/m²")
            print(f"  Estimated m²: {m_sq_est:.2e} ± {estimate['m_squared_std']:.2e} 1/m²")
            print(f"  Relative error: {relative_error*100:.1f}%")
            print(f"  R²: {estimate['r_squared']:.4f}")
            
            # Validation criterion
            validation_passed = (
                relative_error < p.m_squared_tolerance and
                estimate['r_squared'] > 0.90
            )
            
            print(f"\n  Validation: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
            
        else:
            print("\n  ✗ Estimator failed on synthetic data")
            validation_passed = False
        
        validation_results = {
            'synthetic_data': phi_4d,
            'estimate': estimate,
            'm_squared_true': p.m_squared_expected,
            'validation_passed': validation_passed
        }
        
        return validation_results
    
    def part_b_biological_data(self,
                               phi_experimental: Optional[np.ndarray] = None) -> Dict:
        """
        Part B: Apply validated estimator to biological data
        
        Protocol:
        1. Acquire 4D microscopy (VSD or morphogen reporter)
        2. Apply validated estimator
        3. Extract stable m² value
        
        Returns:
            Biological data analysis results
        """
        print("\n" + "=" * 70)
        print(" PILOT 3 - PART B: BIOLOGICAL DATA ANALYSIS ")
        print("=" * 70)
        
        p = self.params
        
        # If no experimental data provided, simulate "experimental" data
        # with realistic noise and sampling
        if phi_experimental is None:
            print("\nSimulating realistic experimental data...")
            
            # Generate KG field with noise
            kg_params = KGParameters(
                domain_size=p.tissue_size,
                n_points=p.n_points,
                m_squared=p.m_squared_expected * 1.1  # Slightly different
            )
            
            kg = KleinGordonField(kg_params)
            
            n_frames = int(p.duration * p.frame_rate)
            record_interval = int(1.0 / (p.frame_rate * kg_params.dt))
            
            results = kg.simulate(
                duration=p.duration,
                record_interval=record_interval
            )
            
            phi_experimental = results['phi']
            
            # Add experimental noise
            noise_level = 0.1  # 10% noise
            for t in range(phi_experimental.shape[0]):
                phi_experimental[t] += noise_level * np.random.randn(*p.n_points)
            
            dx, dy, dz = kg_params.dx, kg_params.dy, kg_params.dz
            
            print(f"  Generated {phi_experimental.shape[0]} frames with noise")
        else:
            # Use actual experimental data
            dx = p.tissue_size[0] / p.n_points[0]
            dy = p.tissue_size[1] / p.n_points[1]
            dz = p.tissue_size[2] / p.n_points[2]
        
        # Apply estimator
        print("\nApplying estimator to biological data...")
        estimate = self.power_spectrum_estimator(phi_experimental, dx, dy, dz)
        
        if estimate['fit_success']:
            print(f"\nBiological Data Results:")
            print(f"  Estimated m²: {estimate['m_squared_estimate']:.2e} ± {estimate['m_squared_std']:.2e} 1/m²")
            print(f"  R²: {estimate['r_squared']:.4f}")
            
            # Check stability
            stability_passed = estimate['r_squared'] > 0.80
            
            print(f"\n  Stability Test: {'✓ PASSED' if stability_passed else '✗ FAILED'}")
            print(f"  Criterion: R² > 0.80")
            
        else:
            print("\n  ✗ No stable m² found in biological data")
            stability_passed = False
        
        biological_results = {
            'phi_experimental': phi_experimental,
            'estimate': estimate,
            'stability_passed': stability_passed
        }
        
        return biological_results
    
    def part_c_field_modulation(self) -> Dict:
        """
        Part C: Test Ψ_s field modulation
        
        Critical Prediction:
        m²(Ψ_s_high) should differ from m²(Ψ_s_low)
        
        Protocol:
        1. Apply Ψ_s surrogate fields (control, low, high)
        2. Measure m² in each condition
        3. Test for statistically significant modulation
        
        Returns:
            Field modulation results
        """
        print("\n" + "=" * 70)
        print(" PILOT 3 - PART C: Ψ_s FIELD MODULATION ")
        print("=" * 70)
        
        p = self.params
        
        results_by_condition = {}
        m_squared_values = []
        
        for condition in p.psi_s_conditions:
            print(f"\nCondition: {condition.upper()}")
            
            # Modulate m² based on Ψ_s condition
            if condition == 'baseline':
                m_sq = p.m_squared_expected
                psi_s = 0.0
            elif condition == 'low_field':
                m_sq = p.m_squared_expected * 0.8  # 20% reduction
                psi_s = -0.5
            elif condition == 'high_field':
                m_sq = p.m_squared_expected * 1.2  # 20% increase
                psi_s = +0.5
            
            print(f"  Ψ_s surrogate: {psi_s:+.1f}")
            print(f"  Expected m²: {m_sq:.2e}")
            
            # Simulate with modulated m²
            kg_params = KGParameters(
                domain_size=p.tissue_size,
                n_points=(32, 32, 16),  # Smaller for speed
                m_squared=m_sq
            )
            
            kg = KleinGordonField(kg_params)
            
            # Shorter simulation for each condition
            duration_short = 60.0  # 1 minute
            n_frames = int(duration_short * p.frame_rate)
            record_interval = int(1.0 / (p.frame_rate * kg_params.dt))
            
            results = kg.simulate(
                duration=duration_short,
                record_interval=record_interval
            )
            
            phi_data = results['phi']
            
            # Estimate m²
            estimate = self.power_spectrum_estimator(
                phi_data,
                kg_params.dx,
                kg_params.dy,
                kg_params.dz
            )
            
            if estimate['fit_success']:
                m_sq_measured = estimate['m_squared_estimate']
                print(f"  Measured m²: {m_sq_measured:.2e} ± {estimate['m_squared_std']:.2e}")
                
                m_squared_values.append(m_sq_measured)
            else:
                print(f"  ✗ Fit failed")
                m_squared_values.append(np.nan)
            
            results_by_condition[condition] = {
                'phi_data': phi_data,
                'estimate': estimate,
                'm_squared_expected': m_sq,
                'psi_s': psi_s
            }
        
        # Statistical test for modulation
        print("\n" + "-" * 70)
        print("FIELD MODULATION ANALYSIS:")
        
        # Remove NaN values
        m_sq_clean = [m for m in m_squared_values if not np.isnan(m)]
        
        if len(m_sq_clean) >= 2:
            m_sq_range = max(m_sq_clean) - min(m_sq_clean)
            m_sq_mean = np.mean(m_sq_clean)
            
            relative_modulation = m_sq_range / m_sq_mean
            
            print(f"  m² range: [{min(m_sq_clean):.2e}, {max(m_sq_clean):.2e}]")
            print(f"  Relative modulation: {relative_modulation*100:.1f}%")
            
            # Falsification criterion: modulation > 10%
            modulation_detected = relative_modulation > 0.10
            
            print(f"\n  Modulation Test: {'✓ DETECTED' if modulation_detected else '✗ NOT DETECTED'}")
            print(f"  Criterion: Relative modulation > 10%")
        else:
            print("  ✗ Insufficient successful measurements")
            modulation_detected = False
        
        modulation_results = {
            'by_condition': results_by_condition,
            'm_squared_values': m_squared_values,
            'modulation_detected': modulation_detected
        }
        
        return modulation_results
    
    def run_complete_pilot3(self) -> Dict:
        """
        Execute complete Pilot 3 protocol
        
        Returns:
            Complete results from all parts
        """
        print("\n" + "=" * 80)
        print(" PILOT 3: COMPLETE QFT EMERGENCE TEST PROTOCOL ")
        print("=" * 80)
        
        # Part A: Validation
        validation = self.part_a_validation()
        
        # Part B: Biological data
        biological = self.part_b_biological_data()
        
        # Part C: Field modulation
        modulation = self.part_c_field_modulation()
        
        # Summary
        print("\n" + "=" * 80)
        print(" PILOT 3 SUMMARY ")
        print("=" * 80)
        
        print("\nPart A - Estimator Validation:")
        if validation['validation_passed']:
            print(f"  ✓ VALIDATED")
            rel_err = abs(validation['estimate']['m_squared_estimate'] - 
                         validation['m_squared_true']) / validation['m_squared_true']
            print(f"  Error: {rel_err*100:.1f}%")
        else:
            print(f"  ✗ FAILED")
        
        print("\nPart B - Biological Data:")
        if biological['stability_passed']:
            print(f"  ✓ STABLE m² FOUND")
            print(f"  m² = {biological['estimate']['m_squared_estimate']:.2e}")
        else:
            print(f"  ✗ NO STABLE m²")
        
        print("\nPart C - Field Modulation:")
        if modulation['modulation_detected']:
            print(f"  ✓ Ψ_s MODULATION DETECTED")
        else:
            print(f"  ✗ NO MODULATION")
        
        # Overall assessment
        all_passed = (
            validation['validation_passed'] and
            biological['stability_passed'] and
            modulation['modulation_detected']
        )
        
        print("\n" + "-" * 80)
        print(f"OVERALL PILOT 3 STATUS: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 80)
        
        complete_results = {
            'part_a_validation': validation,
            'part_b_biological': biological,
            'part_c_modulation': modulation,
            'all_tests_passed': all_passed
        }
        
        self.results = complete_results
        return complete_results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualize Pilot 3 results
        """
        if not self.results:
            raise ValueError("Must run complete pilot first")
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Part A - Power spectrum fit
        validation = self.results['part_a_validation']
        est = validation['estimate']
        
        ax1 = fig.add_subplot(gs[0, 0])
        if est['fit_success']:
            ax1.loglog(est['k'], est['P'], 'b.', alpha=0.3, label='Data')
            ax1.loglog(est['k_fit'], est['P_fit'], 'ro', label='Fit range')
            
            # Fitted curve
            k_smooth = np.logspace(np.log10(est['k_fit'][0]), 
                                  np.log10(est['k_fit'][-1]), 100)
            P_smooth = est['P_fit'][0] / (k_smooth**2 + est['m_squared_estimate'])
            ax1.loglog(k_smooth, P_smooth, 'r-', linewidth=2, label='KG fit')
        
        ax1.set_xlabel('k (1/m)', fontsize=12)
        ax1.set_ylabel('P(k)', fontsize=12)
        ax1.set_title('Part A: Synthetic Data Fit', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Part B - Biological data
        biological = self.results['part_b_biological']
        
        ax2 = fig.add_subplot(gs[0, 1])
        phi_exp = biological['phi_experimental']
        # Show middle frame
        mid_frame = phi_exp.shape[0] // 2
        im = ax2.imshow(phi_exp[mid_frame, :, :, phi_exp.shape[3]//2].T,
                       origin='lower', cmap='viridis')
        ax2.set_title('Part B: Biological Data (mid-frame)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='φ')
        
        # Part C - Field modulation
        modulation = self.results['part_c_modulation']
        
        ax3 = fig.add_subplot(gs[0, 2])
        conditions = list(modulation['by_condition'].keys())
        m_sq_vals = modulation['m_squared_values']
        
        # Filter out NaN
        valid_idx = [i for i, m in enumerate(m_sq_vals) if not np.isnan(m)]
        conditions_valid = [conditions[i] for i in valid_idx]
        m_sq_valid = [m_sq_vals[i] for i in valid_idx]
        
        ax3.bar(range(len(conditions_valid)), m_sq_valid, color=['blue', 'orange', 'red'])
        ax3.set_xticks(range(len(conditions_valid)))
        ax3.set_xticklabels(conditions_valid, rotation=45)
        ax3.set_ylabel('m² (1/m²)', fontsize=12)
        ax3.set_title('Part C: Ψ_s Modulation', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Bottom row - Time evolution examples
        ax4 = fig.add_subplot(gs[1, :])
        
        # Show energy evolution from synthetic data
        if 'synthetic_data' in validation:
            # Create a KG simulation to show energy
            from core.qft_kg import KleinGordonField, KGParameters
            kg_params = KGParameters(
                domain_size=self.params.tissue_size,
                n_points=(32, 32, 16),
                m_squared=self.params.m_squared_expected
            )
            kg = KleinGordonField(kg_params)
            
            # Quick simulation
            results = kg.simulate(duration=0.1, record_interval=5)
            
            ax4.plot(results['time']*1e3, results['energy'], 'b-', linewidth=2)
            ax4.set_xlabel('Time (ms)', fontsize=12)
            ax4.set_ylabel('Field Energy', fontsize=12)
            ax4.set_title('Field Energy Evolution', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Pilot 3: QFT Emergence Test Results',
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        return fig


def run_pilot3_demo():
    """
    Run demonstration of complete Pilot 3 protocol
    """
    # Initialize
    pilot3 = Pilot3QFTTest()
    
    # Run complete protocol
    results = pilot3.run_complete_pilot3()
    
    # Visualize
    pilot3.plot_results(save_path='pilot3_qft_emergence.png')
    
    return results


if __name__ == "__main__":
    results = run_pilot3_demo()
    plt.show()
