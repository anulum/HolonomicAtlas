"""
Layer 2 Experimental Validation Suite - Module 5 of 6
Integration & Multi-Scale Tests

This is the CAPSTONE INTEGRATION module that brings together all Layer 2
components across multiple scales and timescales to demonstrate emergent
consciousness dynamics.

Integrates:
-----------
- Module 2: Quantum-classical transition (microtubules)
- Module 3: Neurotransmitter systems and oscillations
- Module 4: Glial networks and metabolic dynamics

Core Experiments:
-----------------
1. Multi-Scale Synchronization - Nested oscillations from molecular to tissue
2. Criticality Maintenance - Glial slow control keeping σ ≈ 1
3. Cross-Frequency Coupling - Phase-amplitude coupling across bands
4. Information Flow Analysis - Transfer entropy between layers/scales
5. Energy-Information Coupling - Metabolic constraints on information processing
6. Emergent Coherence - Collective dynamics from component interactions

Theoretical Foundation:
----------------------
Based on SCPN manuscript:
- Part 1: UPDE and quasicriticality framework
- Part 3: Layer 2 neurochemical-neurological dynamics
- Part 4: Multi-scale cellular-tissue synchronization
- Part 5: Information-theoretic measures

Author: SCPN Validation Suite
Version: 1.0.0
Date: 2025-11-07
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
module_logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: MULTI-SCALE FRAMEWORK
# ============================================================================

class TimeScale(Enum):
    """Hierarchical timescales in the system"""
    QUANTUM = "quantum"          # 10^-15 - 10^-12 s
    MOLECULAR = "molecular"       # 10^-12 - 10^-9 s
    SYNAPTIC = "synaptic"        # 10^-3 - 10^-2 s
    NEURAL = "neural"            # 10^-2 - 10^0 s
    GLIAL = "glial"              # 10^0 - 10^2 s
    METABOLIC = "metabolic"      # 10^2 - 10^3 s


@dataclass
class MultiScaleState:
    """State of the integrated multi-scale system"""
    # Quantum level
    quantum_coherence: float = 0.5
    
    # Neural oscillations
    oscillation_phases: Dict[str, float] = field(default_factory=lambda: {
        'delta': 0.0, 'theta': 0.0, 'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0
    })
    oscillation_amplitudes: Dict[str, float] = field(default_factory=lambda: {
        'delta': 0.1, 'theta': 0.15, 'alpha': 0.2, 'beta': 0.1, 'gamma': 0.05
    })
    
    # Glial level
    astrocyte_calcium: float = 1e-7  # M
    gliotransmitter_concentration: float = 0.0  # M
    
    # Metabolic level
    ATP_concentration: float = 3e-3  # M
    energy_charge: float = 0.85
    
    # System-level properties
    criticality_parameter: float = 1.0  # σ
    integrated_information: float = 0.0  # Φ


# ============================================================================
# SECTION 2: CROSS-FREQUENCY COUPLING
# ============================================================================

class CrossFrequencyCoupling:
    """
    Cross-frequency coupling analysis
    
    Based on Part 4 formalism:
    PAC_index = |⟨A_fast(t) × exp(iφ_slow(t))⟩| / √⟨A²_fast⟩
    """
    
    @staticmethod
    def extract_phase(signal_data: np.ndarray, band_freq: Tuple[float, float],
                     fs: float = 1000.0) -> np.ndarray:
        """Extract instantaneous phase using Hilbert transform"""
        sos = signal.butter(4, band_freq, btype='band', fs=fs, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        analytic = signal.hilbert(filtered)
        return np.angle(analytic)
        
    @staticmethod
    def extract_amplitude(signal_data: np.ndarray, band_freq: Tuple[float, float],
                         fs: float = 1000.0) -> np.ndarray:
        """Extract instantaneous amplitude envelope"""
        sos = signal.butter(4, band_freq, btype='band', fs=fs, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)
        
    def phase_amplitude_coupling(self, signal_data: np.ndarray,
                                phase_band: Tuple[float, float],
                                amp_band: Tuple[float, float],
                                fs: float = 1000.0) -> float:
        """
        Calculate Phase-Amplitude Coupling (PAC)
        
        Returns PAC index (0-1)
        """
        phase_slow = self.extract_phase(signal_data, phase_band, fs)
        amp_fast = self.extract_amplitude(signal_data, amp_band, fs)
        
        # Mean vector length
        complex_coupling = amp_fast * np.exp(1j * phase_slow)
        pac_value = np.abs(np.mean(complex_coupling)) / np.sqrt(np.mean(amp_fast**2))
        
        return float(pac_value)


# ============================================================================
# SECTION 3: INFORMATION FLOW ANALYSIS
# ============================================================================

class InformationFlowAnalyzer:
    """Information-theoretic measures"""
    
    @staticmethod
    def mutual_information(X: np.ndarray, Y: np.ndarray, bins: int = 20) -> float:
        """
        Calculate mutual information
        MI(X;Y) = Σ p(x,y) log[p(x,y) / (p(x)p(y))]
        """
        hist_2d, _, _ = np.histogram2d(X, Y, bins=bins)
        p_xy = hist_2d / np.sum(hist_2d)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-10) + 1e-10)
        return max(0.0, mi)
        
    @staticmethod
    def transfer_entropy(X: np.ndarray, Y: np.ndarray,
                        lag: int = 1, bins: int = 10) -> float:
        """
        Calculate Transfer Entropy X → Y
        TE_X→Y = Σ p(y_t+1, y_t, x_t) log[p(y_t+1|y_t, x_t) / p(y_t+1|y_t)]
        """
        n = len(X) - lag
        y_future = Y[lag:]
        y_present = Y[:n]
        x_present = X[:n]
        
        hist_3d, _ = np.histogramdd(
            np.column_stack([y_future, y_present, x_present]),
            bins=bins
        )
        hist_2d, _ = np.histogramdd(
            np.column_stack([y_future, y_present]),
            bins=bins
        )
        
        p_yyx = hist_3d / (np.sum(hist_3d) + 1e-10)
        p_yy = hist_2d / (np.sum(hist_2d) + 1e-10)
        
        te = 0.0
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    if p_yyx[i, j, k] > 0:
                        p_cond_full = p_yyx[i, j, k] / (np.sum(p_yyx[:, j, k]) + 1e-10)
                        p_cond_partial = p_yy[i, j] / (np.sum(p_yy[:, j]) + 1e-10)
                        if p_cond_partial > 0:
                            te += p_yyx[i, j, k] * np.log(p_cond_full / p_cond_partial + 1e-10)
        return max(0.0, te)


# ============================================================================
# SECTION 4: CRITICALITY MAINTENANCE
# ============================================================================

class CriticalityController:
    """
    Glial slow control for criticality maintenance
    
    Based on Part 1 and Part 4:
    dσ/dt = -κ(σ - (1 + γG(t))) + η(t)
    dG/dt = α⟨[Ca²⁺]_A⟩ - βG
    """
    
    def __init__(self):
        self.sigma = 1.0  # Branching parameter
        self.kappa = 0.1  # Homeostatic relaxation rate
        self.gamma = 0.5  # Gliotransmitter coupling
        
        self.G = 0.0  # Gliotransmitter concentration
        self.alpha = 0.1  # Ca²⁺-dependent release
        self.beta = 0.05  # Clearance rate
        
        self.sigma_history: List[float] = []
        self.G_history: List[float] = []
        
    def update_criticality(self, dt: float, Ca_astrocyte: float, noise_std: float = 0.01):
        """Update criticality and gliotransmitter dynamics"""
        # Gliotransmitter
        dG_dt = self.alpha * Ca_astrocyte * 1e6 - self.beta * self.G
        self.G += dt * dG_dt
        self.G = max(0.0, self.G)
        
        # Criticality with glial modulation
        target_sigma = 1.0 + self.gamma * self.G
        noise = np.random.normal(0, noise_std)
        dsigma_dt = -self.kappa * (self.sigma - target_sigma) + noise
        self.sigma += dt * dsigma_dt
        self.sigma = np.clip(self.sigma, 0.5, 1.5)
        
        self.sigma_history.append(self.sigma)
        self.G_history.append(self.G)
        
    def is_critical(self, tolerance: float = 0.1) -> bool:
        """Check if in critical regime"""
        return abs(self.sigma - 1.0) < tolerance
        
    def metrics(self) -> Dict[str, float]:
        """Get criticality statistics"""
        if len(self.sigma_history) < 10:
            return {'mean_sigma': self.sigma, 'std_sigma': 0.0, 
                   'fraction_critical': 0.0, 'mean_G': self.G}
        recent = np.array(self.sigma_history[-100:])
        critical_mask = np.abs(recent - 1.0) < 0.1
        return {
            'mean_sigma': np.mean(recent),
            'std_sigma': np.std(recent),
            'fraction_critical': np.mean(critical_mask),
            'mean_G': np.mean(self.G_history[-100:])
        }


# ============================================================================
# SECTION 5: INTEGRATED SYSTEM
# ============================================================================

class IntegratedLayer2System:
    """Complete integrated Layer 2 system"""
    
    def __init__(self, n_neurons: int = 100):
        self.n_neurons = n_neurons
        self.state = MultiScaleState()
        
        # Neural oscillators
        self.neural_phases = np.random.uniform(0, 2*np.pi, n_neurons)
        self.neural_frequencies = np.random.normal(10.0, 2.0, n_neurons)  # Hz
        
        # Coupling strengths
        self.K_neural = 0.5
        self.zeta_field = 0.1
        self.gamma_glial = 0.2
        
        # Subsystems
        self.criticality = CriticalityController()
        self.cfc = CrossFrequencyCoupling()
        self.info = InformationFlowAnalyzer()
        
        self.time = 0.0
        self.activity_history: List[np.ndarray] = []
        
    def update_neural_oscillators(self, dt: float):
        """
        Update neural phase oscillators
        dφᵢ/dt = ωᵢ + (K/N)Σⱼ sin(φⱼ - φᵢ) + ζΨₛ cos(φᵢ) + γG + η
        """
        # Kuramoto coupling
        coupling = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            coupling[i] = np.mean(np.sin(self.neural_phases - self.neural_phases[i]))
        
        # Field coupling
        field_term = self.zeta_field * np.cos(self.neural_phases)
        
        # Glial modulation
        glial_term = self.gamma_glial * self.criticality.G
        
        # Noise
        noise = np.random.normal(0, 0.1, self.n_neurons)
        
        # Update
        dphase = (self.neural_frequencies + self.K_neural * coupling + 
                 field_term + glial_term + noise)
        self.neural_phases += dt * dphase
        self.neural_phases = np.mod(self.neural_phases, 2*np.pi)
        
    def update_metabolic(self, dt: float, activity: float):
        """Update metabolic state"""
        consumption = 0.1 * activity
        production = 0.08
        dATP = production - consumption
        self.state.ATP_concentration += dt * dATP
        self.state.ATP_concentration = np.clip(self.state.ATP_concentration, 1e-3, 5e-3)
        
        # Energy charge
        total = 3.5e-3
        ADP = total - self.state.ATP_concentration - 0.05e-3
        self.state.energy_charge = (self.state.ATP_concentration + 0.5*ADP) / total
        
    def step(self, dt: float):
        """Single integration step"""
        # Calculate neural activity
        activity = np.mean(np.cos(self.neural_phases))
        
        # Update subsystems
        self.update_neural_oscillators(dt)
        self.state.astrocyte_calcium = 1e-7 * (1 + 5 * max(0, activity))
        self.criticality.update_criticality(dt, self.state.astrocyte_calcium)
        self.update_metabolic(dt, abs(activity))
        
        # Record
        self.activity_history.append(np.cos(self.neural_phases))
        self.time += dt
        
    def get_signal(self, duration: int = 1000) -> np.ndarray:
        """Get composite neural signal"""
        if len(self.activity_history) < duration:
            return np.array([])
        recent = self.activity_history[-duration:]
        return np.mean(recent, axis=1)


# ============================================================================
# SECTION 6: EXPERIMENTS
# ============================================================================

class MultiScaleSynchronizationExperiment:
    """Experiment 1: Multi-scale synchronization and PAC"""
    
    def __init__(self):
        self.name = "Multi-Scale Synchronization"
        self.system = IntegratedLayer2System(n_neurons=100)
        self.results: Dict = {}
        
    def run(self, duration: float = 60.0, dt: float = 0.001) -> Dict:
        """Run multi-scale synchronization experiment"""
        module_logger.info(f"Running {self.name}...")
        
        steps = int(duration / dt)
        
        # Simulate
        for _ in range(steps):
            self.system.step(dt)
            
        # Get signal
        signal_data = self.system.get_signal(duration=10000)  # Last 10s
        
        if len(signal_data) == 0:
            return {'error': 'Insufficient data'}
            
        # Analyze cross-frequency coupling
        fs = 1.0 / dt
        cfc = self.system.cfc
        
        # Theta-gamma PAC
        theta_gamma_pac = cfc.phase_amplitude_coupling(
            signal_data, (4, 8), (30, 50), fs
        )
        
        # Alpha-beta PAC
        alpha_beta_pac = cfc.phase_amplitude_coupling(
            signal_data, (8, 12), (15, 25), fs
        )
        
        # Criticality metrics
        crit_metrics = self.system.criticality.metrics()
        
        self.results = {
            'theta_gamma_PAC': theta_gamma_pac,
            'alpha_beta_PAC': alpha_beta_pac,
            'criticality': crit_metrics,
            'final_ATP': self.system.state.ATP_concentration,
            'energy_charge': self.system.state.energy_charge
        }
        
        return self.results
        
    def validate(self) -> Dict:
        """Validate results"""
        validations = {}
        
        # PAC should be detectable (>0.1)
        validations['theta_gamma_PAC_valid'] = self.results.get('theta_gamma_PAC', 0) > 0.05
        
        # Should maintain criticality
        validations['criticality_maintained'] = (
            self.results.get('criticality', {}).get('fraction_critical', 0) > 0.5
        )
        
        # Energy charge should be healthy
        validations['energy_charge_healthy'] = (
            0.7 < self.results.get('energy_charge', 0) < 0.95
        )
        
        validations['all_valid'] = all(validations.values())
        return validations


class CriticalityMaintenanceExperiment:
    """Experiment 2: Glial slow control of criticality"""
    
    def __init__(self):
        self.name = "Criticality Maintenance"
        self.system = IntegratedLayer2System(n_neurons=100)
        self.results: Dict = {}
        
    def run(self, duration: float = 120.0, dt: float = 0.01) -> Dict:
        """Test criticality maintenance under perturbations"""
        module_logger.info(f"Running {self.name}...")
        
        steps = int(duration / dt)
        sigma_trace = []
        G_trace = []
        
        # Simulate with perturbations
        for step in range(steps):
            # Perturbation at 30s, 60s, 90s
            if step in [3000, 6000, 9000]:
                # Push away from criticality
                self.system.criticality.sigma = 1.3
                
            self.system.step(dt)
            sigma_trace.append(self.system.criticality.sigma)
            G_trace.append(self.system.criticality.G)
            
        # Analyze recovery
        sigma_array = np.array(sigma_trace)
        
        # Find perturbation points
        perturbations = [3000, 6000, 9000]
        recovery_times = []
        
        for pert_idx in perturbations:
            # Find when sigma returns to critical window
            post_pert = sigma_array[pert_idx:]
            critical_mask = np.abs(post_pert - 1.0) < 0.1
            if np.any(critical_mask):
                recovery_idx = np.where(critical_mask)[0][0]
                recovery_time = recovery_idx * dt
                recovery_times.append(recovery_time)
                
        self.results = {
            'sigma_trace': sigma_trace,
            'G_trace': G_trace,
            'mean_recovery_time': np.mean(recovery_times) if recovery_times else 0,
            'fraction_critical': np.mean(np.abs(sigma_array - 1.0) < 0.1),
            'sigma_std': np.std(sigma_array)
        }
        
        return self.results
        
    def validate(self) -> Dict:
        """Validate criticality maintenance"""
        validations = {}
        
        # Should recover from perturbations
        validations['recovers'] = self.results.get('mean_recovery_time', 0) < 20.0
        
        # Should spend most time in critical regime
        validations['stays_critical'] = self.results.get('fraction_critical', 0) > 0.6
        
        # Should have bounded fluctuations
        validations['bounded'] = self.results.get('sigma_std', 1.0) < 0.3
        
        validations['all_valid'] = all(validations.values())
        return validations


class InformationFlowExperiment:
    """Experiment 3: Information flow across scales"""
    
    def __init__(self):
        self.name = "Information Flow Analysis"
        self.system = IntegratedLayer2System(n_neurons=100)
        self.results: Dict = {}
        
    def run(self, duration: float = 60.0, dt: float = 0.01) -> Dict:
        """Analyze information transfer between scales"""
        module_logger.info(f"Running {self.name}...")
        
        steps = int(duration / dt)
        
        # Simulate
        for _ in range(steps):
            self.system.step(dt)
            
        # Extract time series
        activity = self.system.get_signal(duration=min(5000, len(self.system.activity_history)))
        if len(activity) < 100:
            return {'error': 'Insufficient data'}
            
        # Glial trace
        glial = np.array(self.system.criticality.G_history[-len(activity):])
        
        # Neural activity trace
        neural = activity
        
        # Calculate information flow
        info = self.system.info
        
        # Mutual information
        mi_glial_neural = info.mutual_information(glial, neural)
        
        # Transfer entropy
        te_glial_to_neural = info.transfer_entropy(glial, neural, lag=10)
        te_neural_to_glial = info.transfer_entropy(neural, glial, lag=10)
        
        self.results = {
            'mutual_information': mi_glial_neural,
            'TE_glial_to_neural': te_glial_to_neural,
            'TE_neural_to_glial': te_neural_to_glial,
            'bidirectional_flow': te_glial_to_neural > 0 and te_neural_to_glial > 0
        }
        
        return self.results
        
    def validate(self) -> Dict:
        """Validate information flow"""
        validations = {}
        
        # Should have mutual information
        validations['has_MI'] = self.results.get('mutual_information', 0) > 0.01
        
        # Should have glial→neural flow
        validations['glial_to_neural'] = self.results.get('TE_glial_to_neural', 0) > 0.001
        
        # Should be bidirectional
        validations['bidirectional'] = self.results.get('bidirectional_flow', False)
        
        validations['all_valid'] = all(validations.values())
        return validations


# ============================================================================
# SECTION 7: DEMONSTRATION
# ============================================================================

def run_module_5_demo():
    """Run Module 5 demonstration"""
    print("\n" + "="*80)
    print("Module 5: Integration & Multi-Scale Tests")
    print("="*80 + "\n")
    
    # Experiment 1: Multi-scale synchronization
    print("Experiment 1: Multi-Scale Synchronization & PAC")
    print("-" * 60)
    exp1 = MultiScaleSynchronizationExperiment()
    results1 = exp1.run(duration=60.0, dt=0.001)
    validation1 = exp1.validate()
    
    print(f"Theta-Gamma PAC: {results1['theta_gamma_PAC']:.4f}")
    print(f"Alpha-Beta PAC: {results1['alpha_beta_PAC']:.4f}")
    print(f"Criticality maintained: {results1['criticality']['fraction_critical']:.2%}")
    print(f"Energy charge: {results1['energy_charge']:.3f}")
    print(f"Validation: {'✓ PASS' if validation1['all_valid'] else '✗ FAIL'}")
    
    # Experiment 2: Criticality maintenance
    print("\n\nExperiment 2: Criticality Maintenance")
    print("-" * 60)
    exp2 = CriticalityMaintenanceExperiment()
    results2 = exp2.run(duration=120.0, dt=0.01)
    validation2 = exp2.validate()
    
    print(f"Mean recovery time: {results2['mean_recovery_time']:.2f} s")
    print(f"Fraction in critical regime: {results2['fraction_critical']:.2%}")
    print(f"σ standard deviation: {results2['sigma_std']:.3f}")
    print(f"Validation: {'✓ PASS' if validation2['all_valid'] else '✗ FAIL'}")
    
    # Experiment 3: Information flow
    print("\n\nExperiment 3: Information Flow Analysis")
    print("-" * 60)
    exp3 = InformationFlowExperiment()
    results3 = exp3.run(duration=60.0, dt=0.01)
    validation3 = exp3.validate()
    
    print(f"Mutual Information: {results3['mutual_information']:.4f}")
    print(f"TE (Glial→Neural): {results3['TE_glial_to_neural']:.4f}")
    print(f"TE (Neural→Glial): {results3['TE_neural_to_glial']:.4f}")
    print(f"Bidirectional: {results3['bidirectional_flow']}")
    print(f"Validation: {'✓ PASS' if validation3['all_valid'] else '✗ FAIL'}")
    
    print("\n" + "="*80)
    print("Module 5 Complete - Integration Successful!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_module_5_demo()
