"""
Comprehensive SCPN Layer 4 Simulation Framework
Complete implementation with visualization and analysis for tissue synchronization.
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
    A complete simulation framework for SCPN Layer 4 tissue synchronization.

    This class integrates all Layer 4 components, including the oscillator network,
    quasicritical dynamics, cosmological entrainment, multi-path coupling, clinical
    biomarkers, and information-theoretic measures. It provides a unified environment
    to run, analyze, and visualize complex tissue synchronization simulations.

    Attributes:
        config (Dict): A dictionary containing the simulation configuration.
        oscillator_network (HierarchicalOscillatorNetwork): The core oscillator network model.
        quasicritical (QuasicriticalDynamics): The quasicritical dynamics module.
        cosmological (CosmologicalEntrainment): The cosmological entrainment module.
        hamiltonian (MultiPathCouplingHamiltonian): The multi-path coupling Hamiltonian.
        biomarkers (ClinicalBiomarkers): The clinical biomarkers analysis module.
        information (InformationTheoreticMeasures): The information-theoretic measures module.
        validation (ExperimentalValidation): The experimental validation module.
        therapeutics (TherapeuticInterventions): The therapeutic interventions module.
        results (Dict): A dictionary to store all simulation results.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the comprehensive simulation framework.

        Args:
            config (Optional[Dict]): A dictionary of simulation parameters. If None,
                default parameters are used.
        """
        
        default_config = {
            'n_neurons': 100, 'n_glia': 40, 'n_vascular': 20,
            'topology': 'small_world', 'simulation_duration': 100.0,
            'dt': 0.01, 'schumann_coupling': True, 'lunar_coupling': True,
            'save_results': True
        }
        
        self.config = config if config else default_config
        self._initialize_components()
        self.results = {
            'time_series': {}, 'metrics': {}, 'clinical': {},
            'information': {}, 'validation': {}
        }
        
    def _initialize_components(self):
        """Initializes all model components based on the configuration."""
        self.oscillator_network = HierarchicalOscillatorNetwork(N=self.config['n_neurons'], topology=self.config['topology'])
        self.quasicritical = QuasicriticalDynamics(network_size=self.config['n_neurons'])
        self.cosmological = CosmologicalEntrainment()
        self.hamiltonian = MultiPathCouplingHamiltonian(N_neurons=self.config['n_neurons'], N_glia=self.config['n_glia'], N_vascular=self.config['n_vascular'])
        self.biomarkers = ClinicalBiomarkers()
        self.information = InformationTheoreticMeasures(n_elements=self.config['n_neurons'])
        self.validation = ExperimentalValidation()
        self.therapeutics = TherapeuticInterventions()
    
    def run_simulation(self) -> Dict:
        """
        Runs the complete tissue synchronization simulation.

        This method executes the main simulation loop, updating all components at each
        time step, and then calls the analysis and validation methods upon completion.

        Returns:
            Dict: A dictionary containing all simulation results.
        """
        print("Starting comprehensive tissue synchronization simulation...")
        duration, dt = self.config['simulation_duration'], self.config['dt']
        time_points = np.arange(0, duration, dt)
        n_steps = len(time_points)
        
        phases = np.zeros((self.config['n_neurons'], n_steps))
        synchronization = np.zeros(n_steps)
        hamiltonian_energy = np.zeros(n_steps)
        phi_values = np.zeros(n_steps // 100)
        
        print("Running main simulation loop...")
        for i, t in enumerate(time_points):
            if i % 1000 == 0: print(f"  Progress: {i/n_steps*100:.1f}%")
            
            if self.config['schumann_coupling']:
                schumann_field = self.cosmological.schumann_resonance_field(t)
                self.oscillator_network.frequencies *= 1 + 0.1 * schumann_field[0]
            
            if self.config['lunar_coupling']:
                lunar = self.cosmological.lunar_influence(t / 86400)
                self.oscillator_network.params.synaptic_coupling *= lunar['gravitational']
            
            dphase = self.oscillator_network.kuramoto_dynamics(t, self.oscillator_network.phases)
            self.oscillator_network.phases += dphase * dt
            
            phases[:, i] = self.oscillator_network.phases
            synchronization[i] = self.oscillator_network.calculate_synchronization()
            
            self.hamiltonian.neuron_states = np.cos(self.oscillator_network.phases)
            self.hamiltonian.evolve_system(dt)
            hamiltonian_energy[i] = self.hamiltonian.total_hamiltonian()
            
            if i % 100 == 0:
                connectivity = nx.adjacency_matrix(self.oscillator_network.network).toarray()
                phi = self.information.integrated_information(self.hamiltonian.neuron_states, connectivity)
                phi_values[i // 100] = phi
            
            if i % 10 == 0:
                self.oscillator_network.avalanche_sizes.extend(self.oscillator_network.detect_avalanches())
        
        print("Analyzing results...")
        self.results['time_series'] = {'time': time_points, 'phases': phases, 'synchronization': synchronization, 'hamiltonian': hamiltonian_energy, 'phi': phi_values}
        self._calculate_metrics()
        self._clinical_analysis()
        self._information_analysis()
        self._validate_results()
        
        print("Simulation complete!")
        return self.results
    
    def _calculate_metrics(self):
        """Calculates summary metrics from the simulation time series."""
        sync = self.results['time_series']['synchronization']
        self.results['metrics'] = {
            'mean_synchronization': np.mean(sync), 'std_synchronization': np.std(sync),
            'metastability_index': self.oscillator_network.calculate_metastability_index(),
            'num_avalanches': len(self.oscillator_network.avalanche_sizes),
            'mean_avalanche_size': np.mean(self.oscillator_network.avalanche_sizes) if self.oscillator_network.avalanche_sizes else 0
        }
        self.results['metrics'].update(self.quasicritical.griffiths_phase_detection(sync))
        if len(self.oscillator_network.avalanche_sizes) > 50:
            tau, p_value = self.validation._fit_power_law(self.oscillator_network.avalanche_sizes)
            self.results['metrics']['avalanche_exponent'] = tau
            self.results['metrics']['power_law_p_value'] = p_value
    
    def _clinical_analysis(self):
        """Performs clinical biomarker analysis on the simulation results."""
        neural_signals = np.cos(self.results['time_series']['phases'][:, ::10])
        plv = self.biomarkers.phase_locking_value(neural_signals)
        wpli = self.biomarkers.weighted_phase_lag_index(neural_signals)
        criticality = self.biomarkers.criticality_markers(neural_signals)
        
        self.results['clinical'] = {'PLV': plv, 'wPLI': wpli, 'criticality': criticality, 'pathological_markers': self.biomarkers.pathological_detection(neural_signals)}
        
        therapy_stim = self.therapeutics.transcranial_stimulation_protocol(target_frequency=10.0, current_state=criticality['clinical_correlate'])
        therapy_pharma = self.therapeutics.pharmacological_modulation({'branching_parameter': criticality['branching_parameter'], 'PLV': plv, 'wPLI': wpli})
        self.results['clinical']['therapeutic_recommendations'] = {'stimulation': therapy_stim, 'pharmacological': therapy_pharma}
    
    def _information_analysis(self):
        """Performs information-theoretic analysis on the simulation results."""
        signals = self.results['time_series']['synchronization']
        complexity = self.information.complexity_measures(signals)
        te_example = self.information.transfer_entropy(signals[:100], np.roll(signals[:100], 5)) if len(signals) > 100 else 0
        self.results['information'] = {'complexity_measures': complexity, 'transfer_entropy': te_example, 'mean_phi': np.mean(self.results['time_series']['phi'])}
    
    def _validate_results(self):
        """Validates the simulation results against theoretical predictions."""
        neural_data = np.cos(self.results['time_series']['phases'])
        criticality_validation = self.validation.validate_criticality(neural_data)
        self.results['validation'] = criticality_validation
        
        checks = {
            'is_synchronized': self.results['metrics']['mean_synchronization'] > 0.3,
            'is_metastable': self.results['metrics']['metastability_index'] > 1.0,
            'shows_avalanches': self.results['metrics']['num_avalanches'] > 0,
            'is_critical': criticality_validation['is_critical'],
            'shows_complexity': all(v > 0 for v in self.results['information']['complexity_measures'].values())
        }
        self.results['validation']['property_checks'] = checks
        self.results['validation']['validation_score'] = sum(checks.values()) / len(checks)
    
    def visualize_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Creates a comprehensive visualization of the simulation results.

        Args:
            save_path (Optional[str]): The path to save the figure. If None, the
                figure is not saved.

        Returns:
            plt.Figure: The matplotlib Figure object containing the plots.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        time, sync = self.results['time_series']['time'], self.results['time_series']['synchronization']
        ax1.plot(time, sync, 'b-', alpha=0.7, linewidth=0.5); ax1.fill_between(time, 0, sync, alpha=0.3)
        ax1.set(xlabel='Time (s)', ylabel='Synchronization', title='Global Synchronization Dynamics', ylim=[0, 1]); ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2], projection='polar')
        final_phases = self.results['time_series']['phases'][:, -1]
        theta = np.linspace(0, 2*np.pi, 50)
        r_hist, _ = np.histogram(final_phases % (2*np.pi), bins=50)
        ax2.bar(theta, r_hist / np.max(r_hist), width=2*np.pi/50, alpha=0.7); ax2.set_title('Final Phase Distribution')
        
        ax3 = fig.add_subplot(gs[0, 3])
        G = self.oscillator_network.network
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color=np.cos(final_phases), node_size=30, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3)
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax3)
        ax3.set(title='Network State'); ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 0])
        if self.oscillator_network.avalanche_sizes:
            sizes = self.oscillator_network.avalanche_sizes
            hist, bins = np.histogram(sizes, bins=30)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax4.loglog(bin_centers, hist, 'bo', markersize=4, alpha=0.6)
            if 'avalanche_exponent' in self.results['metrics']:
                tau = self.results['metrics']['avalanche_exponent']
                x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
                y_fit = x_fit ** (-tau) * hist[0] * (bin_centers[0] ** tau)
                ax4.loglog(x_fit, y_fit, 'r--', label=f'τ = {tau:.2f}', linewidth=2); ax4.legend()
        ax4.set(xlabel='Avalanche Size', ylabel='Frequency', title='Avalanche Size Distribution'); ax4.grid(True, alpha=0.3, which='both')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time, self.results['time_series']['hamiltonian'], 'g-', alpha=0.7, linewidth=0.5)
        ax5.set(xlabel='Time (s)', ylabel='Hamiltonian Energy', title='System Energy Evolution'); ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        phi = self.results['time_series']['phi']
        ax6.plot(np.linspace(0, time[-1], len(phi)), phi, 'purple', linewidth=2, alpha=0.7)
        ax6.fill_between(np.linspace(0, time[-1], len(phi)), 0, phi, alpha=0.3, color='purple')
        ax6.set(xlabel='Time (s)', ylabel='Φ (bits)', title='Integrated Information'); ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[1, 3])
        biomarkers = self.results['clinical']['biomarker_values']
        ax7.bar(biomarkers.keys(), biomarkers.values(), color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        ax7.set(ylabel='Value', title='Clinical Biomarkers', ylim=[0, max(biomarkers.values()) * 1.2])
        ax7.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Critical'); ax7.legend()
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax8 = fig.add_subplot(gs[2, :2])
        complexity = self.results['information']['complexity_measures']
        x_pos = np.arange(len(complexity))
        ax8.bar(x_pos, complexity.values(), alpha=0.7, color='teal')
        ax8.set_xticks(x_pos); ax8.set_xticklabels(complexity.keys(), rotation=45, ha='right')
        ax8.set(ylabel='Complexity', title='Information Complexity Measures'); ax8.grid(True, alpha=0.3, axis='y')
        
        ax9 = fig.add_subplot(gs[2, 2:])
        validation = self.results['validation']['property_checks']
        val_labels = [label.replace('_', ' ').title() for label in validation.keys()]
        colors_val = ['green' if v else 'red' for v in validation.values()]
        y_pos = np.arange(len(val_labels))
        ax9.barh(y_pos, [1 if v else 0 for v in validation.values()], color=colors_val, alpha=0.7)
        ax9.set_yticks(y_pos); ax9.set_yticklabels(val_labels)
        ax9.set(xlabel='Pass/Fail', title=f'System Validation (Score: {self.results["validation"]["validation_score"]:.2%})', xlim=[0, 1.2]); ax9.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle('SCPN Layer 4: Tissue Synchronization Simulation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig
    
    def save_results(self, filename: Optional[str] = None):
        """
        Saves the simulation results to a JSON file.

        Args:
            filename (Optional[str]): The name of the file to save. If None, a
                timestamped filename is generated.
        """
        if filename is None: filename = f"scpn_layer4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self._make_serializable(self.results), f, indent=2)
        print(f"Results saved to {filename}")
    
    def _make_serializable(self, obj):
        """Recursively converts numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._make_serializable(i) for i in obj]
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        return obj
    
    def generate_report(self) -> str:
        """
        Generates a comprehensive text report of the simulation results.

        Returns:
            str: A formatted string containing the full report.
        """
        report = ["="*80, "SCPN LAYER 4 - TISSUE SYNCHRONIZATION SIMULATION REPORT", "="*80, ""]
        report.append("SIMULATION CONFIGURATION:\n" + "-"*40)
        report.extend([f"  {k}: {v}" for k, v in self.config.items()])
        
        metrics = self.results['metrics']
        report.extend(["", "KEY METRICS:\n" + "-"*40, f"  Mean Synchronization: {metrics['mean_synchronization']:.3f}", f"  Metastability Index: {metrics['metastability_index']:.3f}", f"  Number of Avalanches: {metrics['num_avalanches']}"])
        if 'avalanche_exponent' in metrics: report.append(f"  Avalanche Exponent (τ): {metrics['avalanche_exponent']:.3f}")
        report.append(f"  Griffiths Phase: {metrics.get('is_griffiths', False)}")
        
        clinical = self.results['clinical']
        report.extend(["", "CLINICAL BIOMARKERS:\n" + "-"*40, f"  Phase Locking Value: {clinical['PLV']:.3f}", f"  Weighted Phase Lag Index: {clinical['wPLI']:.3f}", f"  Criticality State: {clinical['criticality']['state']}", f"  Branching Parameter: {clinical['criticality']['branching_parameter']:.3f}", f"  Consciousness Level: {clinical['pathological_markers']['consciousness_level']}"])
        
        info = self.results['information']
        report.extend(["", "INFORMATION-THEORETIC MEASURES:\n" + "-"*40, f"  Mean Integrated Information (Φ): {info['mean_phi']:.3f} bits", f"  Transfer Entropy: {info['transfer_entropy']:.3f} bits"])
        report.extend([f"  {m.replace('_', ' ').title()}: {v:.3f}" for m, v in info['complexity_measures'].items()])
        
        validation = self.results['validation']
        report.extend(["", "SYSTEM VALIDATION:\n" + "-"*40, f"  Overall Validation Score: {validation['validation_score']:.2%}", "  Property Checks:"])
        report.extend([f"    {'✓' if p else '✗'} {prop.replace('_', ' ').title()}" for prop, p in validation['property_checks'].items()])
        
        therapy = clinical['therapeutic_recommendations']
        stim = therapy['stimulation']
        report.extend(["", "THERAPEUTIC RECOMMENDATIONS:\n" + "-"*40, "  Stimulation Protocol:", f"    Type: {stim['type']}", f"    Frequency: {stim.get('frequency', 'N/A')} Hz", f"    Intensity: {stim['intensity']} mA", f"    Duration: {stim['duration']} minutes", "  Pharmacological Suggestions:"])
        report.extend([f"    {i}. {s}" for i, s in enumerate(therapy['pharmacological'][:3], 1)])
        
        report.extend(["", "SUMMARY:\n" + "-"*40])
        if validation['validation_score'] > 0.8: report.extend(["✓ Simulation successfully demonstrates tissue synchronization dynamics", "✓ System exhibits critical brain dynamics", "✓ Multi-scale coupling mechanisms validated"])
        elif validation['validation_score'] > 0.5: report.extend(["◐ Partial validation of tissue synchronization theory", "◐ Some expected properties observed", "◐ Further parameter tuning may improve results"])
        else: report.extend(["✗ Limited validation of expected dynamics", "✗ Consider adjusting model parameters", "✗ Check component coupling strengths"])
        
        report.extend(["", "="*80, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*80])
        return "\n".join(report)

if __name__ == "__main__":
    print("="*80, "\nSCPN LAYER 4 - COMPREHENSIVE TISSUE SYNCHRONIZATION SIMULATION\n" + "="*80, "\n")
    config = {
        'n_neurons': 100, 'n_glia': 40, 'n_vascular': 20,
        'topology': 'small_world', 'simulation_duration': 50.0, 'dt': 0.01,
        'schumann_coupling': True, 'lunar_coupling': True, 'save_results': True
    }
    sim = ComprehensiveTissueSimulation(config)
    results = sim.run_simulation()
    print("\n" + sim.generate_report())
    print("\nGenerating visualizations...")
    sim.visualize_results(save_path='scpn_layer4_results.png')
    if config['save_results']: sim.save_results()
    print("\n" + "="*80, "\nSIMULATION COMPLETE\n" + "="*80)
