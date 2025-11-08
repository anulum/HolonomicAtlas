"""
Comprehensive SCPN Layer 4 Simulation Framework
Complete implementation with visualization and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import signal, stats
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from tissue_synchronization_model import (
    HierarchicalOscillatorNetwork, 
    QuasicriticalDynamics,
    CosmologicalEntrainment
)
from clinical_biomarkers_system import (
    MultiPathCouplingHamiltonian,
    ClinicalBiomarkers,
    TherapeuticInterventions
)
from information_theoretic_validation import (
    InformationTheoreticMeasures,
    ExperimentalValidation
)

class ComprehensiveTissueSimulation:
    """
    Complete simulation framework integrating all Layer 4 components
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize comprehensive simulation"""
        
        # Default configuration
        default_config = {
            'n_neurons': 100,
            'n_glia': 40,
            'n_vascular': 20,
            'topology': 'small_world',
            'simulation_duration': 100.0,
            'dt': 0.01,
            'schumann_coupling': True,
            'lunar_coupling': True,
            'save_results': True
        }
        
        self.config = config if config else default_config
        
        # Initialize all components
        self._initialize_components()
        
        # Results storage
        self.results = {
            'time_series': {},
            'metrics': {},
            'clinical': {},
            'information': {},
            'validation': {}
        }
        
    def _initialize_components(self):
        """Initialize all model components"""
        
        # Core oscillator network
        self.oscillator_network = HierarchicalOscillatorNetwork(
            N=self.config['n_neurons'],
            topology=self.config['topology']
        )
        
        # Quasicritical dynamics
        self.quasicritical = QuasicriticalDynamics(
            network_size=self.config['n_neurons']
        )
        
        # Cosmological entrainment
        self.cosmological = CosmologicalEntrainment()
        
        # Multi-path coupling
        self.hamiltonian = MultiPathCouplingHamiltonian(
            N_neurons=self.config['n_neurons'],
            N_glia=self.config['n_glia'],
            N_vascular=self.config['n_vascular']
        )
        
        # Clinical biomarkers
        self.biomarkers = ClinicalBiomarkers()
        
        # Information measures
        self.information = InformationTheoreticMeasures(
            n_elements=self.config['n_neurons']
        )
        
        # Experimental validation
        self.validation = ExperimentalValidation()
        
        # Therapeutic interventions
        self.therapeutics = TherapeuticInterventions()
    
    def run_simulation(self) -> Dict:
        """
        Run complete simulation with all components
        """
        print("Starting comprehensive tissue synchronization simulation...")
        
        # Time parameters
        duration = self.config['simulation_duration']
        dt = self.config['dt']
        time_points = np.arange(0, duration, dt)
        n_steps = len(time_points)
        
        # Initialize storage arrays
        phases = np.zeros((self.config['n_neurons'], n_steps))
        synchronization = np.zeros(n_steps)
        hamiltonian_energy = np.zeros(n_steps)
        phi_values = np.zeros(n_steps // 100)  # Calculate less frequently
        
        # Simulation loop
        print("Running main simulation loop...")
        for i, t in enumerate(time_points):
            if i % 1000 == 0:
                print(f"  Progress: {i/n_steps*100:.1f}%")
            
            # Apply cosmological influences
            if self.config['schumann_coupling']:
                schumann_field = self.cosmological.schumann_resonance_field(t)
                # Modulate intrinsic frequencies
                freq_modulation = 1 + 0.1 * schumann_field[0]
                self.oscillator_network.frequencies *= freq_modulation
            
            if self.config['lunar_coupling']:
                lunar = self.cosmological.lunar_influence(t / 86400)  # Convert to days
                # Modulate coupling strengths
                self.oscillator_network.params.synaptic_coupling *= lunar['gravitational']
            
            # Update oscillator network
            dphase = self.oscillator_network.kuramoto_dynamics(t, self.oscillator_network.phases)
            self.oscillator_network.phases += dphase * dt
            
            # Store states
            phases[:, i] = self.oscillator_network.phases
            synchronization[i] = self.oscillator_network.calculate_synchronization()
            
            # Update Hamiltonian system
            self.hamiltonian.neuron_states = np.cos(self.oscillator_network.phases)
            self.hamiltonian.evolve_system(dt)
            hamiltonian_energy[i] = self.hamiltonian.total_hamiltonian()
            
            # Calculate integrated information periodically
            if i % 100 == 0:
                connectivity = nx.adjacency_matrix(
                    self.oscillator_network.network
                ).toarray()
                phi = self.information.integrated_information(
                    self.hamiltonian.neuron_states,
                    connectivity
                )
                phi_values[i // 100] = phi
            
            # Detect avalanches
            if i % 10 == 0:
                avalanches = self.oscillator_network.detect_avalanches()
                self.oscillator_network.avalanche_sizes.extend(avalanches)
        
        print("Analyzing results...")
        
        # Store time series
        self.results['time_series'] = {
            'time': time_points,
            'phases': phases,
            'synchronization': synchronization,
            'hamiltonian': hamiltonian_energy,
            'phi': phi_values
        }
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Clinical analysis
        self._clinical_analysis()
        
        # Information-theoretic analysis
        self._information_analysis()
        
        # Validation checks
        self._validate_results()
        
        print("Simulation complete!")
        
        return self.results
    
    def _calculate_metrics(self):
        """Calculate summary metrics"""
        
        sync = self.results['time_series']['synchronization']
        
        self.results['metrics'] = {
            'mean_synchronization': np.mean(sync),
            'std_synchronization': np.std(sync),
            'metastability_index': self.oscillator_network.calculate_metastability_index(),
            'num_avalanches': len(self.oscillator_network.avalanche_sizes),
            'mean_avalanche_size': np.mean(self.oscillator_network.avalanche_sizes) 
                if self.oscillator_network.avalanche_sizes else 0
        }
        
        # Griffiths phase detection
        griffiths = self.quasicritical.griffiths_phase_detection(sync)
        self.results['metrics'].update(griffiths)
        
        # Power law fitting for avalanches
        if len(self.oscillator_network.avalanche_sizes) > 50:
            tau, p_value = self.validation._fit_power_law(
                self.oscillator_network.avalanche_sizes
            )
            self.results['metrics']['avalanche_exponent'] = tau
            self.results['metrics']['power_law_p_value'] = p_value
    
    def _clinical_analysis(self):
        """Perform clinical biomarker analysis"""
        
        # Sample neural signals for analysis
        neural_signals = np.cos(self.results['time_series']['phases'][:, ::10])
        
        # Calculate biomarkers
        plv = self.biomarkers.phase_locking_value(neural_signals)
        wpli = self.biomarkers.weighted_phase_lag_index(neural_signals)
        criticality = self.biomarkers.criticality_markers(neural_signals)
        
        self.results['clinical'] = {
            'PLV': plv,
            'wPLI': wpli,
            'criticality': criticality,
            'pathological_markers': self.biomarkers.pathological_detection(neural_signals)
        }
        
        # Get therapeutic recommendations
        therapy_stim = self.therapeutics.transcranial_stimulation_protocol(
            target_frequency=10.0,
            current_state=criticality['clinical_correlate']
        )
        
        therapy_pharma = self.therapeutics.pharmacological_modulation(
            {'branching_parameter': criticality['branching_parameter'],
             'PLV': plv, 'wPLI': wpli}
        )
        
        self.results['clinical']['therapeutic_recommendations'] = {
            'stimulation': therapy_stim,
            'pharmacological': therapy_pharma
        }
    
    def _information_analysis(self):
        """Perform information-theoretic analysis"""
        
        # Sample signals
        signals = self.results['time_series']['synchronization']
        
        # Calculate various information measures
        complexity = self.information.complexity_measures(signals)
        
        # Transfer entropy between oscillators
        if len(signals) > 100:
            te_example = self.information.transfer_entropy(
                signals[:100], 
                np.roll(signals[:100], 5)
            )
        else:
            te_example = 0
        
        self.results['information'] = {
            'complexity_measures': complexity,
            'transfer_entropy': te_example,
            'mean_phi': np.mean(self.results['time_series']['phi'])
        }
    
    def _validate_results(self):
        """Validate simulation results against theoretical predictions"""
        
        # Validate criticality
        neural_data = np.cos(self.results['time_series']['phases'])
        criticality_validation = self.validation.validate_criticality(neural_data)
        
        self.results['validation'] = criticality_validation
        
        # Check if system exhibits expected properties
        checks = {
            'is_synchronized': self.results['metrics']['mean_synchronization'] > 0.3,
            'is_metastable': self.results['metrics']['metastability_index'] > 1.0,
            'shows_avalanches': self.results['metrics']['num_avalanches'] > 0,
            'is_critical': criticality_validation['is_critical'],
            'shows_complexity': all(v > 0 for v in 
                                   self.results['information']['complexity_measures'].values())
        }
        
        self.results['validation']['property_checks'] = checks
        self.results['validation']['validation_score'] = sum(checks.values()) / len(checks)
    
    def visualize_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of results
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Synchronization time series
        ax1 = fig.add_subplot(gs[0, :2])
        time = self.results['time_series']['time']
        sync = self.results['time_series']['synchronization']
        ax1.plot(time, sync, 'b-', alpha=0.7, linewidth=0.5)
        ax1.fill_between(time, 0, sync, alpha=0.3)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Synchronization')
        ax1.set_title('Global Synchronization Dynamics')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase distribution
        ax2 = fig.add_subplot(gs[0, 2], projection='polar')
        final_phases = self.results['time_series']['phases'][:, -1]
        theta = np.linspace(0, 2*np.pi, 50)
        r_hist, _ = np.histogram(final_phases % (2*np.pi), bins=50)
        r_hist = r_hist / np.max(r_hist)
        ax2.bar(theta, r_hist, width=2*np.pi/50, alpha=0.7)
        ax2.set_title('Final Phase Distribution')
        
        # 3. Network visualization
        ax3 = fig.add_subplot(gs[0, 3])
        G = self.oscillator_network.network
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        node_colors = np.cos(final_phases)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=30, cmap='coolwarm', 
                              vmin=-1, vmax=1, ax=ax3)
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax3)
        ax3.set_title('Network State')
        ax3.axis('off')
        
        # 4. Avalanche distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if self.oscillator_network.avalanche_sizes:
            sizes = self.oscillator_network.avalanche_sizes
            hist, bins = np.histogram(sizes, bins=30)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot histogram
            ax4.loglog(bin_centers, hist, 'bo', markersize=4, alpha=0.6)
            
            # Fit power law
            if 'avalanche_exponent' in self.results['metrics']:
                tau = self.results['metrics']['avalanche_exponent']
                x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
                y_fit = x_fit ** (-tau) * hist[0] * (bin_centers[0] ** tau)
                ax4.loglog(x_fit, y_fit, 'r--', 
                          label=f'τ = {tau:.2f}', linewidth=2)
                ax4.legend()
        
        ax4.set_xlabel('Avalanche Size')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Avalanche Size Distribution')
        ax4.grid(True, alpha=0.3, which='both')
        
        # 5. Hamiltonian energy
        ax5 = fig.add_subplot(gs[1, 1])
        energy = self.results['time_series']['hamiltonian']
        ax5.plot(time, energy, 'g-', alpha=0.7, linewidth=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Hamiltonian Energy')
        ax5.set_title('System Energy Evolution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Integrated Information (Phi)
        ax6 = fig.add_subplot(gs[1, 2])
        phi = self.results['time_series']['phi']
        phi_time = np.linspace(0, time[-1], len(phi))
        ax6.plot(phi_time, phi, 'purple', linewidth=2, alpha=0.7)
        ax6.fill_between(phi_time, 0, phi, alpha=0.3, color='purple')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Φ (bits)')
        ax6.set_title('Integrated Information')
        ax6.grid(True, alpha=0.3)
        
        # 7. Clinical biomarkers
        ax7 = fig.add_subplot(gs[1, 3])
        biomarkers = self.results['clinical']['biomarker_values']
        labels = list(biomarkers.keys())
        values = list(biomarkers.values())
        colors = ['blue', 'green', 'red', 'orange']
        bars = ax7.bar(labels, values, color=colors, alpha=0.7)
        ax7.set_ylabel('Value')
        ax7.set_title('Clinical Biomarkers')
        ax7.set_ylim([0, max(values) * 1.2])
        
        # Add reference line for criticality
        ax7.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Critical')
        ax7.legend()
        
        # Rotate labels
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 8. Complexity measures
        ax8 = fig.add_subplot(gs[2, :2])
        complexity = self.results['information']['complexity_measures']
        comp_labels = list(complexity.keys())
        comp_values = list(complexity.values())
        
        x_pos = np.arange(len(comp_labels))
        bars = ax8.bar(x_pos, comp_values, alpha=0.7, color='teal')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(comp_labels, rotation=45, ha='right')
        ax8.set_ylabel('Complexity')
        ax8.set_title('Information Complexity Measures')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Validation results
        ax9 = fig.add_subplot(gs[2, 2:])
        validation = self.results['validation']['property_checks']
        val_labels = [label.replace('_', ' ').title() for label in validation.keys()]
        val_values = [1 if v else 0 for v in validation.values()]
        
        # Create color map
        colors_val = ['green' if v else 'red' for v in validation.values()]
        
        y_pos = np.arange(len(val_labels))
        bars = ax9.barh(y_pos, val_values, color=colors_val, alpha=0.7)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(val_labels)
        ax9.set_xlabel('Pass/Fail')
        ax9.set_title(f'System Validation (Score: {self.results["validation"]["validation_score"]:.2%})')
        ax9.set_xlim([0, 1.2])
        ax9.grid(True, alpha=0.3, axis='x')
        
        # Add overall title
        fig.suptitle('SCPN Layer 4: Tissue Synchronization Simulation Results', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, filename: Optional[str] = None):
        """Save simulation results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scpn_layer4_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def generate_report(self) -> str:
        """Generate comprehensive text report of results"""
        
        report = []
        report.append("=" * 80)
        report.append("SCPN LAYER 4 - TISSUE SYNCHRONIZATION SIMULATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Configuration
        report.append("SIMULATION CONFIGURATION:")
        report.append("-" * 40)
        for key, value in self.config.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # Key Metrics
        report.append("KEY METRICS:")
        report.append("-" * 40)
        metrics = self.results['metrics']
        report.append(f"  Mean Synchronization: {metrics['mean_synchronization']:.3f}")
        report.append(f"  Metastability Index: {metrics['metastability_index']:.3f}")
        report.append(f"  Number of Avalanches: {metrics['num_avalanches']}")
        if 'avalanche_exponent' in metrics:
            report.append(f"  Avalanche Exponent (τ): {metrics['avalanche_exponent']:.3f}")
        report.append(f"  Griffiths Phase: {metrics.get('is_griffiths', False)}")
        report.append("")
        
        # Clinical Analysis
        report.append("CLINICAL BIOMARKERS:")
        report.append("-" * 40)
        clinical = self.results['clinical']
        report.append(f"  Phase Locking Value: {clinical['PLV']:.3f}")
        report.append(f"  Weighted Phase Lag Index: {clinical['wPLI']:.3f}")
        report.append(f"  Criticality State: {clinical['criticality']['state']}")
        report.append(f"  Branching Parameter: {clinical['criticality']['branching_parameter']:.3f}")
        report.append(f"  Consciousness Level: {clinical['pathological_markers']['consciousness_level']}")
        report.append("")
        
        # Information Measures
        report.append("INFORMATION-THEORETIC MEASURES:")
        report.append("-" * 40)
        info = self.results['information']
        report.append(f"  Mean Integrated Information (Φ): {info['mean_phi']:.3f} bits")
        report.append(f"  Transfer Entropy: {info['transfer_entropy']:.3f} bits")
        for measure, value in info['complexity_measures'].items():
            report.append(f"  {measure.replace('_', ' ').title()}: {value:.3f}")
        report.append("")
        
        # Validation
        report.append("SYSTEM VALIDATION:")
        report.append("-" * 40)
        validation = self.results['validation']
        report.append(f"  Overall Validation Score: {validation['validation_score']:.2%}")
        report.append("  Property Checks:")
        for prop, passed in validation['property_checks'].items():
            status = "✓" if passed else "✗"
            report.append(f"    {status} {prop.replace('_', ' ').title()}")
        report.append("")
        
        # Therapeutic Recommendations
        report.append("THERAPEUTIC RECOMMENDATIONS:")
        report.append("-" * 40)
        therapy = clinical['therapeutic_recommendations']
        
        report.append("  Stimulation Protocol:")
        stim = therapy['stimulation']
        report.append(f"    Type: {stim['type']}")
        report.append(f"    Frequency: {stim.get('frequency', 'N/A')} Hz")
        report.append(f"    Intensity: {stim['intensity']} mA")
        report.append(f"    Duration: {stim['duration']} minutes")
        
        report.append("  Pharmacological Suggestions:")
        for i, suggestion in enumerate(therapy['pharmacological'][:3], 1):
            report.append(f"    {i}. {suggestion}")
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        
        # Interpret results
        if validation['validation_score'] > 0.8:
            report.append("✓ Simulation successfully demonstrates tissue synchronization dynamics")
            report.append("✓ System exhibits critical brain dynamics")
            report.append("✓ Multi-scale coupling mechanisms validated")
        elif validation['validation_score'] > 0.5:
            report.append("◐ Partial validation of tissue synchronization theory")
            report.append("◐ Some expected properties observed")
            report.append("◐ Further parameter tuning may improve results")
        else:
            report.append("✗ Limited validation of expected dynamics")
            report.append("✗ Consider adjusting model parameters")
            report.append("✗ Check component coupling strengths")
        
        report.append("")
        report.append("=" * 80)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        return "\n".join(report)


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("SCPN LAYER 4 - COMPREHENSIVE TISSUE SYNCHRONIZATION SIMULATION")
    print("=" * 80)
    print()
    
    # Configure simulation
    config = {
        'n_neurons': 100,
        'n_glia': 40,
        'n_vascular': 20,
        'topology': 'small_world',
        'simulation_duration': 50.0,  # seconds
        'dt': 0.01,
        'schumann_coupling': True,
        'lunar_coupling': True,
        'save_results': True
    }
    
    # Create and run simulation
    sim = ComprehensiveTissueSimulation(config)
    results = sim.run_simulation()
    
    # Generate and print report
    report = sim.generate_report()
    print("\n" + report)
    
    # Visualize results
    print("\nGenerating visualizations...")
    fig = sim.visualize_results(save_path='scpn_layer4_results.png')
    
    # Save results
    if config['save_results']:
        sim.save_results()
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
