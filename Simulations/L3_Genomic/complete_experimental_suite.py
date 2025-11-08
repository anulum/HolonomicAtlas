"""
Layer 3 Complete Experimental Suite - Integration Module
==========================================================

Master integration script that executes all three pilot studies and
provides comprehensive validation of Layer 3 mechanisms.

This module orchestrates:
- Pilot 1: CBC Causal Test
- Pilot 2: V2M/PDE Validation  
- Pilot 3: QFT Emergence Test

Plus complete inter-layer information flow analysis.

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import time
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import all experimental modules
try:
    from experiments.pilot1_cbc_causal import run_full_pilot1
    from experiments.pilot2_v2m_validation import run_pilot2_demo
    from experiments.pilot3_qft_emergence import run_pilot3_demo
except ImportError:
    print("Warning: Some experimental modules not found. Running in standalone mode.")


@dataclass
class ExperimentalManifest:
    """Complete experimental run manifest"""
    
    timestamp: str
    git_hash: str
    python_version: str
    numpy_version: str
    scipy_version: str
    
    pilot1_status: str
    pilot2_status: str
    pilot3_status: str
    
    overall_status: str
    
    pilot1_results: Dict
    pilot2_results: Dict
    pilot3_results: Dict
    
    execution_time: float  # seconds
    
    def to_json(self, filepath: str):
        """Save manifest to JSON"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        print(f"\nManifest saved to: {filepath}")


class Layer3ExperimentalSuite:
    """
    Complete Layer 3 Experimental Suite
    
    Orchestrates all three pilot studies and provides
    comprehensive validation framework.
    """
    
    def __init__(self):
        self.results = {}
        self.manifest = None
        
    def run_all_pilots(self, 
                       run_pilot1: bool = True,
                       run_pilot2: bool = True,
                       run_pilot3: bool = True) -> Dict:
        """
        Execute all pilot studies
        
        Args:
            run_pilot1: Execute Pilot 1 (CBC Causal Test)
            run_pilot2: Execute Pilot 2 (V2M Validation)
            run_pilot3: Execute Pilot 3 (QFT Emergence)
            
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        print("=" * 90)
        print(" LAYER 3 EXPERIMENTAL SUITE - COMPLETE VALIDATION PROTOCOL ")
        print("=" * 90)
        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nThis suite will execute:")
        if run_pilot1:
            print("  • Pilot 1: CBC Causal Test (temporal precedence & chirality)")
        if run_pilot2:
            print("  • Pilot 2: V2M/PDE Validation (morphogenetic fields)")
        if run_pilot3:
            print("  • Pilot 3: QFT Emergence Test (quantum signatures)")
        print("\n" + "=" * 90)
        
        results = {}
        
        # ========================================
        # PILOT 1: CBC CAUSAL TEST
        # ========================================
        if run_pilot1:
            try:
                print("\n\n")
                print("█" * 90)
                print("█" + " " * 88 + "█")
                print("█" + " " * 25 + "PILOT 1: CBC CAUSAL TEST" + " " * 39 + "█")
                print("█" + " " * 88 + "█")
                print("█" * 90)
                print()
                
                pilot1_results = run_full_pilot1()
                
                # Extract key metrics
                causal_test = pilot1_results['causal_test']
                chirality = pilot1_results['chirality_control']
                
                pilot1_passed = (
                    causal_test['precedence']['temporal_precedence_valid'] and
                    chirality['sign_reversed']
                )
                
                results['pilot1'] = {
                    'status': 'PASSED' if pilot1_passed else 'FAILED',
                    'temporal_precedence': causal_test['precedence']['temporal_precedence_valid'],
                    'chirality_reversal': chirality['sign_reversed'],
                    'full_results': pilot1_results
                }
                
                print(f"\n{'✓' if pilot1_passed else '✗'} PILOT 1: {results['pilot1']['status']}")
                
            except Exception as e:
                print(f"\n✗ PILOT 1 FAILED WITH ERROR: {e}")
                results['pilot1'] = {'status': 'ERROR', 'error': str(e)}
        
        # ========================================
        # PILOT 2: V2M/PDE VALIDATION
        # ========================================
        if run_pilot2:
            try:
                print("\n\n")
                print("█" * 90)
                print("█" + " " * 88 + "█")
                print("█" + " " * 22 + "PILOT 2: V2M/PDE VALIDATION" + " " * 39 + "█")
                print("█" + " " * 88 + "█")
                print("█" * 90)
                print()
                
                pilot2_results = run_pilot2_demo()
                
                pilot2_passed = pilot2_results['all_tests_passed']
                
                results['pilot2'] = {
                    'status': 'PASSED' if pilot2_passed else 'FAILED',
                    'calibration_r2': pilot2_results['part_a_calibration']['mean_r_squared'],
                    'validation_corr': pilot2_results['part_b_validation']['mean_correlation'],
                    'blockade_reduction': 1 - pilot2_results['part_c_blockade']['reduction_ratio'],
                    'full_results': pilot2_results
                }
                
                print(f"\n{'✓' if pilot2_passed else '✗'} PILOT 2: {results['pilot2']['status']}")
                
            except Exception as e:
                print(f"\n✗ PILOT 2 FAILED WITH ERROR: {e}")
                results['pilot2'] = {'status': 'ERROR', 'error': str(e)}
        
        # ========================================
        # PILOT 3: QFT EMERGENCE TEST
        # ========================================
        if run_pilot3:
            try:
                print("\n\n")
                print("█" * 90)
                print("█" + " " * 88 + "█")
                print("█" + " " * 21 + "PILOT 3: QFT EMERGENCE TEST" + " " * 40 + "█")
                print("█" + " " * 88 + "█")
                print("█" * 90)
                print()
                
                pilot3_results = run_pilot3_demo()
                
                pilot3_passed = pilot3_results['all_tests_passed']
                
                results['pilot3'] = {
                    'status': 'PASSED' if pilot3_passed else 'FAILED',
                    'estimator_validated': pilot3_results['part_a_validation']['validation_passed'],
                    'stable_m_squared': pilot3_results['part_b_biological']['stability_passed'],
                    'field_modulation': pilot3_results['part_c_modulation']['modulation_detected'],
                    'full_results': pilot3_results
                }
                
                print(f"\n{'✓' if pilot3_passed else '✗'} PILOT 3: {results['pilot3']['status']}")
                
            except Exception as e:
                print(f"\n✗ PILOT 3 FAILED WITH ERROR: {e}")
                results['pilot3'] = {'status': 'ERROR', 'error': str(e)}
        
        # ========================================
        # FINAL SUMMARY
        # ========================================
        execution_time = time.time() - start_time
        
        print("\n\n")
        print("=" * 90)
        print(" EXPERIMENTAL SUITE COMPLETE - FINAL SUMMARY ")
        print("=" * 90)
        
        # Count results
        total_tests = sum([run_pilot1, run_pilot2, run_pilot3])
        passed_tests = sum([
            results.get('pilot1', {}).get('status') == 'PASSED' if run_pilot1 else False,
            results.get('pilot2', {}).get('status') == 'PASSED' if run_pilot2 else False,
            results.get('pilot3', {}).get('status') == 'PASSED' if run_pilot3 else False
        ])
        
        print(f"\nTests Passed: {passed_tests}/{total_tests}")
        print(f"Execution Time: {execution_time:.1f} seconds")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if run_pilot1:
            print(f"\nPilot 1 (CBC): {results['pilot1']['status']}")
        if run_pilot2:
            print(f"Pilot 2 (V2M): {results['pilot2']['status']}")
        if run_pilot3:
            print(f"Pilot 3 (QFT): {results['pilot3']['status']}")
        
        # Overall assessment
        all_passed = (passed_tests == total_tests)
        
        print("\n" + "=" * 90)
        if all_passed:
            print("✓✓✓ ALL EXPERIMENTAL VALIDATIONS PASSED ✓✓✓")
            print("\nLayer 3 mechanisms are experimentally validated:")
            print("  ✓ CBC cascade shows correct temporal precedence")
            print("  ✓ Chirality dependence confirmed")
            print("  ✓ V2M operator accurately predicts morphogen dynamics")
            print("  ✓ Transporter blockade eliminates voltage-driven accumulation")
            print("  ✓ Klein-Gordon mass parameter detected in biological data")
            print("  ✓ Ψ_s field modulation affects morphogenetic patterns")
            print("\nThe SCPN Layer 3 framework is empirically supported!")
        else:
            print("⚠ SOME VALIDATIONS FAILED ⚠")
            print("\nReview failed tests for theoretical or methodological issues.")
        
        print("=" * 90)
        
        # Store results
        self.results = results
        
        # Create manifest
        self.manifest = ExperimentalManifest(
            timestamp=datetime.now().isoformat(),
            git_hash="N/A",  # Would be populated from git in production
            python_version=sys.version.split()[0],
            numpy_version=np.__version__,
            scipy_version="N/A",  # Would import scipy
            pilot1_status=results.get('pilot1', {}).get('status', 'NOT_RUN'),
            pilot2_status=results.get('pilot2', {}).get('status', 'NOT_RUN'),
            pilot3_status=results.get('pilot3', {}).get('status', 'NOT_RUN'),
            overall_status='PASSED' if all_passed else 'FAILED',
            pilot1_results=results.get('pilot1', {}),
            pilot2_results=results.get('pilot2', {}),
            pilot3_results=results.get('pilot3', {}),
            execution_time=execution_time
        )
        
        return results
    
    def generate_comprehensive_report(self, 
                                     save_path: str = 'layer3_complete_report.png'):
        """
        Generate comprehensive visualization of all results
        """
        if not self.results:
            raise ValueError("Must run experiments first")
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Title
        fig.suptitle('Layer 3 Complete Experimental Validation Suite\nSCPN Framework',
                    fontsize=18, fontweight='bold', y=0.98)
        
        # ========================================
        # ROW 1: PILOT 1 - CBC CAUSAL TEST
        # ========================================
        if 'pilot1' in self.results and self.results['pilot1']['status'] != 'ERROR':
            pilot1 = self.results['pilot1']['full_results']
            
            # Temporal precedence
            ax1 = fig.add_subplot(gs[0, 0])
            prec = pilot1['causal_test']['precedence']
            times = [prec['t_spin'], prec['t_field'], prec['t_channel'], 
                    prec['t_voltage'], prec['t_chromatin']]
            times = [t*1e3 for t in times if not np.isnan(t)]  # ms
            stages = ['Spin', 'Field', 'Channel', 'Voltage', 'Chromatin'][:len(times)]
            
            colors = ['blue', 'green', 'purple', 'orange', 'red'][:len(times)]
            ax1.bar(stages, times, color=colors, alpha=0.7)
            ax1.set_ylabel('Onset Time (ms)', fontsize=10)
            ax1.set_title('Pilot 1A: Temporal Precedence', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Chirality reversal
            ax2 = fig.add_subplot(gs[0, 1])
            chirality = pilot1['chirality_control']
            delta_a = [chirality['delta_a_l'], chirality['delta_a_d']]
            ax2.bar(['L-DNA', 'D-DNA'], delta_a, color=['blue', 'red'])
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax2.set_ylabel('ΔChromatin Accessibility', fontsize=10)
            ax2.set_title('Pilot 1B: Chirality Dependence', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Status
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.text(0.5, 0.6, 'PILOT 1', ha='center', fontsize=16, fontweight='bold')
            status = self.results['pilot1']['status']
            color = 'green' if status == 'PASSED' else 'red'
            ax3.text(0.5, 0.4, status, ha='center', fontsize=20, 
                    fontweight='bold', color=color)
            ax3.text(0.5, 0.2, f"Precedence: {'✓' if self.results['pilot1']['temporal_precedence'] else '✗'}",
                    ha='center', fontsize=12)
            ax3.text(0.5, 0.1, f"Chirality: {'✓' if self.results['pilot1']['chirality_reversal'] else '✗'}",
                    ha='center', fontsize=12)
            ax3.axis('off')
        
        # ========================================
        # ROW 2: PILOT 2 - V2M VALIDATION
        # ========================================
        if 'pilot2' in self.results and self.results['pilot2']['status'] != 'ERROR':
            pilot2 = self.results['pilot2']['full_results']
            
            # Calibration
            ax4 = fig.add_subplot(gs[1, 0])
            calib = pilot2['part_a_calibration']
            r2_values = calib['r_squared']
            patterns = [f'P{i+1}' for i in range(len(r2_values))]
            ax4.bar(patterns, r2_values, color='purple', alpha=0.7)
            ax4.axhline(0.85, color='r', linestyle='--', label='Threshold')
            ax4.set_ylabel('R²', fontsize=10)
            ax4.set_title('Pilot 2A: Calibration Fit', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Dynamic validation
            ax5 = fig.add_subplot(gs[1, 1])
            valid = pilot2['part_b_validation']
            t = valid['results_experimental']['time'][:100]  # First 100 points
            corr = valid['correlation_time'][:100]
            ax5.plot(t, corr, 'b-', linewidth=2)
            ax5.axhline(0.90, color='r', linestyle='--', label='Threshold')
            ax5.set_xlabel('Time (s)', fontsize=10)
            ax5.set_ylabel('Spatial Correlation', fontsize=10)
            ax5.set_title('Pilot 2B: Dynamic Validation', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Blockade
            ax6 = fig.add_subplot(gs[1, 2])
            blockade = pilot2['part_c_blockade']
            conditions = list(blockade['by_condition'].keys())
            gradients = [blockade['by_condition'][c]['gradient'] for c in conditions]
            ax6.bar(conditions, gradients, color=['blue', 'red'])
            ax6.set_ylabel('Morphogen Gradient', fontsize=10)
            ax6.set_title('Pilot 2C: Transporter Blockade', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
        
        # ========================================
        # ROW 3: PILOT 3 - QFT EMERGENCE
        # ========================================
        if 'pilot3' in self.results and self.results['pilot3']['status'] != 'ERROR':
            pilot3 = self.results['pilot3']['full_results']
            
            # Estimator validation
            ax7 = fig.add_subplot(gs[2, 0])
            valid = pilot3['part_a_validation']
            if valid['estimate']['fit_success']:
                m_true = valid['m_squared_true']
                m_est = valid['estimate']['m_squared_estimate']
                m_std = valid['estimate']['m_squared_std']
                
                ax7.bar(['True', 'Estimated'], [m_true, m_est], 
                       color=['green', 'blue'], yerr=[0, m_std])
                ax7.set_ylabel('m² (1/m²)', fontsize=10)
                ax7.set_title('Pilot 3A: m² Estimation', fontsize=12, fontweight='bold')
                ax7.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                ax7.grid(True, alpha=0.3, axis='y')
            
            # Power spectrum
            ax8 = fig.add_subplot(gs[2, 1])
            est = valid['estimate']
            if est['fit_success']:
                ax8.loglog(est['k'], est['P'], 'b.', alpha=0.3, label='Data')
                ax8.loglog(est['k_fit'], est['P_fit'], 'ro', label='Fit range')
                ax8.set_xlabel('k (1/m)', fontsize=10)
                ax8.set_ylabel('P(k)', fontsize=10)
                ax8.set_title('Pilot 3B: Power Spectrum', fontsize=12, fontweight='bold')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
            
            # Field modulation
            ax9 = fig.add_subplot(gs[2, 2])
            modul = pilot3['part_c_modulation']
            conditions = list(modul['by_condition'].keys())
            m_sq_vals = modul['m_squared_values']
            
            # Filter NaN
            valid_idx = [i for i, m in enumerate(m_sq_vals) if not np.isnan(m)]
            conditions_valid = [conditions[i] for i in valid_idx]
            m_sq_valid = [m_sq_vals[i] for i in valid_idx]
            
            colors = ['gray', 'blue', 'red'][:len(conditions_valid)]
            ax9.bar(range(len(conditions_valid)), m_sq_valid, 
                   tick_label=conditions_valid, color=colors)
            ax9.set_ylabel('m² (1/m²)', fontsize=10)
            ax9.set_title('Pilot 3C: Ψ_s Modulation', fontsize=12, fontweight='bold')
            ax9.tick_params(axis='x', rotation=45)
            ax9.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax9.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive report saved to: {save_path}")
        
        return fig
    
    def save_manifest(self, filepath: str = 'layer3_manifest.json'):
        """Save experimental manifest to JSON"""
        if self.manifest is None:
            raise ValueError("Must run experiments first")
        
        self.manifest.to_json(filepath)


def run_complete_suite(save_results: bool = True):
    """
    Convenience function to run complete experimental suite
    
    Args:
        save_results: Save manifest and report to files
        
    Returns:
        ExperimentalSuite instance with all results
    """
    suite = Layer3ExperimentalSuite()
    
    # Run all pilots
    results = suite.run_all_pilots(
        run_pilot1=True,
        run_pilot2=True,
        run_pilot3=True
    )
    
    # Generate report
    if save_results:
        suite.generate_comprehensive_report('layer3_complete_validation.png')
        suite.save_manifest('layer3_experimental_manifest.json')
    
    return suite


if __name__ == "__main__":
    print("\n" + "="*90)
    print(" LAYER 3 EXPERIMENTAL SUITE - MASTER EXECUTION ")
    print("="*90 + "\n")
    
    suite = run_complete_suite(save_results=True)
    
    print("\n" + "="*90)
    print(" SUITE EXECUTION COMPLETE ")
    print("="*90)
    print("\nGenerated files:")
    print("  • layer3_complete_validation.png - Comprehensive visual report")
    print("  • layer3_experimental_manifest.json - Complete results manifest")
    print("\nAll experimental protocols have been executed.")
    print("Review results for validation status of Layer 3 mechanisms.")
    print("="*90 + "\n")
    
    plt.show()
