"""
Multi-Path Coupling Mechanisms and Clinical Biomarkers
Extension of SCPN Layer 4 Architecture
"""

import numpy as np
import scipy as sp
from scipy import signal, stats, linalg
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import warnings

class MultiPathCouplingHamiltonian:
    """
    Implements total coupling Hamiltonian:
    H_total = H_synaptic + H_gap + H_ephaptic + H_glial + H_vascular
    """
    
    def __init__(self, N_neurons: int, N_glia: int, N_vascular: int):
        self.N_neurons = N_neurons
        self.N_glia = N_glia
        self.N_vascular = N_vascular
        
        # Initialize coupling matrices
        self.J_synaptic = self._initialize_synaptic_matrix()
        self.g_gap = self._initialize_gap_junctions()
        
        # Field parameters
        self.epsilon_0 = 8.854e-12  # Permittivity of free space
        self.tissue_permittivity = 80  # Relative permittivity of brain tissue
        
        # State variables
        self.neuron_states = np.random.randn(N_neurons)
        self.glia_calcium = np.random.uniform(50, 200, N_glia)  # nM
        self.blood_flow = np.random.uniform(40, 60, N_vascular)  # ml/100g/min
        
    def _initialize_synaptic_matrix(self) -> np.ndarray:
        """Initialize synaptic coupling matrix with Dale's law"""
        J = np.random.randn(self.N_neurons, self.N_neurons) * 0.1
        
        # Enforce Dale's law: 80% excitatory, 20% inhibitory
        n_excitatory = int(0.8 * self.N_neurons)
        J[:n_excitatory, :] = np.abs(J[:n_excitatory, :])
        J[n_excitatory:, :] = -np.abs(J[n_excitatory:, :])
        
        # Remove self-connections
        np.fill_diagonal(J, 0)
        
        return J
    
    def _initialize_gap_junctions(self) -> np.ndarray:
        """Initialize gap junction conductances"""
        g = np.zeros((self.N_neurons, self.N_neurons))
        
        # Gap junctions primarily between inhibitory neurons
        n_excitatory = int(0.8 * self.N_neurons)
        for i in range(n_excitatory, self.N_neurons):
            for j in range(i + 1, self.N_neurons):
                if np.random.random() < 0.3:  # 30% connection probability
                    conductance = np.random.uniform(0.1, 1.0)  # nS
                    g[i, j] = g[j, i] = conductance
        
        return g
    
    def H_synaptic(self, sigma: np.ndarray) -> float:
        """
        Chemical synaptic Hamiltonian: H_synaptic = -∑ᵢⱼ Jᵢⱼ σᵢσⱼ
        """
        return -0.5 * np.sum(sigma @ self.J_synaptic @ sigma)
    
    def H_gap(self, V: np.ndarray) -> float:
        """
        Gap junction Hamiltonian: H_gap = -∑⟨ij⟩ gᵢⱼ(Vᵢ - Vⱼ)²
        """
        energy = 0
        for i in range(self.N_neurons):
            for j in range(i + 1, self.N_neurons):
                if self.g_gap[i, j] > 0:
                    energy -= self.g_gap[i, j] * (V[i] - V[j]) ** 2
        return energy
    
    def H_ephaptic(self, E_field: np.ndarray) -> float:
        """
        Ephaptic field Hamiltonian: H_ephaptic = -∫ ε₀|E|²/2 dV
        """
        # Simplified: discrete sum over field points
        epsilon = self.epsilon_0 * self.tissue_permittivity
        return -0.5 * epsilon * np.sum(np.abs(E_field) ** 2)
    
    def H_glial(self, Ca_concentration: np.ndarray) -> float:
        """
        Glial calcium wave Hamiltonian: H_glial = -∑ₖ αₖ[Ca²⁺]ₖ
        """
        # Alpha represents calcium-dependent energy
        alpha = 0.01  # Coupling constant
        return -alpha * np.sum(Ca_concentration)
    
    def H_vascular(self, flow: np.ndarray) -> float:
        """
        Hemodynamic coupling Hamiltonian: H_vascular = -∑ᵥ βᵥ(flow)ᵥ
        """
        beta = 0.1  # Neurovascular coupling constant
        return -beta * np.sum(flow)
    
    def total_hamiltonian(self) -> float:
        """Calculate total system Hamiltonian"""
        # Generate field from current neural states
        E_field = self._calculate_electric_field(self.neuron_states)
        
        H_total = (
            self.H_synaptic(self.neuron_states) +
            self.H_gap(self.neuron_states * 70)  # Convert to mV
            + self.H_ephaptic(E_field)
            + self.H_glial(self.glia_calcium)
            + self.H_vascular(self.blood_flow)
        )
        
        return H_total
    
    def _calculate_electric_field(self, neural_states: np.ndarray) -> np.ndarray:
        """
        Calculate electric field from neural activity
        Simplified: field proportional to local neural activity gradient
        """
        # Create spatial arrangement
        grid_size = int(np.sqrt(self.N_neurons))
        if grid_size ** 2 < self.N_neurons:
            grid_size += 1
        
        # Pad states to fit grid
        padded_states = np.zeros(grid_size ** 2)
        padded_states[:self.N_neurons] = neural_states
        
        # Reshape to 2D grid
        grid = padded_states.reshape(grid_size, grid_size)
        
        # Calculate gradient (electric field)
        E_y, E_x = np.gradient(grid)
        E_field = np.sqrt(E_x ** 2 + E_y ** 2).flatten()[:self.N_neurons]
        
        return E_field
    
    def evolve_system(self, dt: float = 0.001) -> None:
        """Evolve the coupled system forward in time"""
        # Neural dynamics
        dS_dt = -np.gradient(self.total_hamiltonian())
        self.neuron_states += dS_dt * dt
        
        # Glial calcium dynamics (simplified)
        # Calcium waves propagate through gap junctions
        Ca_diffusion = 0.1  # Diffusion constant
        for i in range(self.N_glia):
            # Simple diffusion to neighbors
            if i > 0:
                self.glia_calcium[i] += Ca_diffusion * (self.glia_calcium[i-1] - self.glia_calcium[i]) * dt
            if i < self.N_glia - 1:
                self.glia_calcium[i] += Ca_diffusion * (self.glia_calcium[i+1] - self.glia_calcium[i]) * dt
        
        # Neurovascular coupling
        # Blood flow follows neural activity with delay
        tau_neurovasc = 1.0  # seconds
        target_flow = 40 + 20 * np.mean(np.abs(self.neuron_states))
        self.blood_flow += (target_flow - self.blood_flow) * dt / tau_neurovasc


class ClinicalBiomarkers:
    """
    Clinical biomarkers for consciousness states
    """
    
    def __init__(self):
        self.biomarkers = {}
        
    def phase_locking_value(self, signals: np.ndarray, freq_band: Tuple[float, float] = (8, 12)) -> float:
        """
        Calculate Phase Locking Value (PLV) for given frequency band
        High PLV indicates strong phase synchronization
        """
        n_signals, n_samples = signals.shape
        
        # Bandpass filter
        fs = 1000  # Sampling frequency (Hz)
        nyquist = fs / 2
        low = freq_band[0] / nyquist
        high = freq_band[1] / nyquist
        
        # Avoid filter issues
        if low <= 0 or high >= 1:
            return 0.0
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Filter signals and extract phase
        phases = np.zeros_like(signals)
        for i in range(n_signals):
            filtered = signal.filtfilt(b, a, signals[i])
            analytic = signal.hilbert(filtered)
            phases[i] = np.angle(analytic)
        
        # Calculate PLV between all pairs
        plv_values = []
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_values.append(plv)
        
        return np.mean(plv_values) if plv_values else 0.0
    
    def weighted_phase_lag_index(self, signals: np.ndarray) -> float:
        """
        Calculate weighted Phase Lag Index (wPLI)
        Robust to volume conduction effects
        """
        n_signals, n_samples = signals.shape
        
        # Calculate cross-spectral density
        nperseg = min(256, n_samples // 4)
        freqs, csd = signal.csd(signals[0], signals[0], fs=1000, nperseg=nperseg)
        
        wpli_matrix = np.zeros((n_signals, n_signals))
        
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                _, Cxy = signal.csd(signals[i], signals[j], fs=1000, nperseg=nperseg)
                
                # Imaginary part of cross-spectrum
                imag_csd = np.imag(Cxy)
                
                # wPLI calculation
                numerator = np.abs(np.mean(imag_csd))
                denominator = np.mean(np.abs(imag_csd))
                
                if denominator > 0:
                    wpli_matrix[i, j] = numerator / denominator
        
        return np.mean(wpli_matrix[wpli_matrix > 0])
    
    def criticality_markers(self, activity: np.ndarray) -> Dict:
        """
        Calculate markers of criticality
        σ branching parameter: =1 critical, >1 supercritical, <1 subcritical
        """
        markers = {}
        
        # Branching parameter
        branching_ratios = []
        for t in range(len(activity) - 1):
            active_now = np.sum(activity[t] > 0)
            active_next = np.sum(activity[t + 1] > 0)
            if active_now > 0:
                branching_ratios.append(active_next / active_now)
        
        markers['branching_parameter'] = np.mean(branching_ratios) if branching_ratios else 0.0
        
        # Determine state
        sigma = markers['branching_parameter']
        if sigma > 1.1:
            markers['state'] = 'supercritical'
            markers['clinical_correlate'] = 'epilepsy_risk'
        elif sigma < 0.9:
            markers['state'] = 'subcritical'
            markers['clinical_correlate'] = 'depression_risk'
        else:
            markers['state'] = 'critical'
            markers['clinical_correlate'] = 'healthy'
        
        # Long-range temporal correlations (DFA)
        markers['dfa_exponent'] = self._detrended_fluctuation_analysis(activity.flatten())
        
        return markers
    
    def _detrended_fluctuation_analysis(self, time_series: np.ndarray, scales: Optional[np.ndarray] = None) -> float:
        """
        Detrended Fluctuation Analysis (DFA) for long-range correlations
        Returns scaling exponent (0.5 = white noise, 1.0 = 1/f noise)
        """
        if scales is None:
            scales = np.logspace(1, 3, 20, dtype=int)
        
        # Integrate time series
        y = np.cumsum(time_series - np.mean(time_series))
        
        fluctuations = []
        for scale in scales:
            if scale >= len(y):
                continue
                
            # Divide into segments
            n_segments = len(y) // scale
            
            if n_segments == 0:
                continue
            
            segment_vars = []
            for i in range(n_segments):
                segment = y[i * scale:(i + 1) * scale]
                
                # Detrend with linear fit
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.poly1d(coeffs)(x)
                
                # Calculate fluctuation
                segment_var = np.sqrt(np.mean((segment - trend) ** 2))
                segment_vars.append(segment_var)
            
            fluctuations.append(np.mean(segment_vars))
        
        if len(fluctuations) > 2:
            # Fit power law
            valid_scales = scales[:len(fluctuations)]
            log_scales = np.log(valid_scales)
            log_fluct = np.log(fluctuations)
            
            # Linear regression in log-log space
            alpha, _ = np.polyfit(log_scales, log_fluct, 1)
            return alpha
        
        return 0.5  # Default to white noise
    
    def pathological_detection(self, neural_data: np.ndarray) -> Dict:
        """
        Detect pathological states from neural dynamics
        """
        detections = {}
        
        # Calculate biomarkers
        plv = self.phase_locking_value(neural_data)
        wpli = self.weighted_phase_lag_index(neural_data)
        criticality = self.criticality_markers(neural_data)
        
        # Epilepsy detection
        if criticality['branching_parameter'] > 1.2:
            detections['epilepsy_risk'] = 'high'
            detections['recommendation'] = 'Consider anti-epileptic intervention'
        
        # Depression detection
        if criticality['branching_parameter'] < 0.8 and plv < 0.3:
            detections['depression_markers'] = 'present'
            detections['recommendation'] = 'Evaluate for mood disorders'
        
        # Schizophrenia markers
        if wpli < 0.2 and criticality['dfa_exponent'] < 0.6:
            detections['schizophrenia_markers'] = 'present'
            detections['recommendation'] = 'Assess connectivity disruption'
        
        # Consciousness level
        consciousness_score = (plv + wpli + (1 - abs(1 - criticality['branching_parameter']))) / 3
        
        if consciousness_score > 0.7:
            detections['consciousness_level'] = 'fully_conscious'
        elif consciousness_score > 0.4:
            detections['consciousness_level'] = 'altered_consciousness'
        else:
            detections['consciousness_level'] = 'minimal_consciousness'
        
        detections['biomarker_values'] = {
            'PLV': plv,
            'wPLI': wpli,
            'branching_parameter': criticality['branching_parameter'],
            'DFA_exponent': criticality['dfa_exponent']
        }
        
        return detections


class TherapeuticInterventions:
    """
    Therapeutic intervention protocols based on tissue synchronization
    """
    
    def __init__(self):
        self.protocols = {}
        
    def transcranial_stimulation_protocol(self, 
                                         target_frequency: float,
                                         current_state: str) -> Dict:
        """
        Design transcranial stimulation protocol
        """
        protocol = {
            'type': 'tACS',  # Transcranial Alternating Current Stimulation
            'frequency': target_frequency,
            'intensity': 2.0,  # mA
            'duration': 20,  # minutes
            'electrode_montage': 'F3-F4',  # Frontal sites
        }
        
        # Adjust based on pathological state
        if current_state == 'epilepsy_risk':
            # Reduce excitability
            protocol['type'] = 'tDCS'  # Direct current
            protocol['polarity'] = 'cathodal'
            protocol['intensity'] = 1.0
            
        elif current_state == 'depression_risk':
            # Increase excitability
            protocol['frequency'] = 10.0  # Alpha frequency
            protocol['intensity'] = 2.5
            protocol['electrode_montage'] = 'F3-Fp2'  # Left DLPFC
            
        elif current_state == 'schizophrenia_markers':
            # Enhance gamma synchronization
            protocol['frequency'] = 40.0  # Gamma
            protocol['intensity'] = 1.5
            
        return protocol
    
    def pharmacological_modulation(self, biomarkers: Dict) -> List[str]:
        """
        Suggest pharmacological interventions based on criticality
        """
        suggestions = []
        
        branching = biomarkers.get('branching_parameter', 1.0)
        
        if branching > 1.1:
            # Supercritical - reduce excitation
            suggestions.append("GABAergic enhancement (e.g., benzodiazepines)")
            suggestions.append("Sodium channel blockers (e.g., carbamazepine)")
            
        elif branching < 0.9:
            # Subcritical - increase excitation
            suggestions.append("NMDA receptor modulation (e.g., ketamine)")
            suggestions.append("Monoamine enhancement (e.g., SSRIs)")
            
        # Based on synchronization
        if biomarkers.get('PLV', 0) < 0.3:
            suggestions.append("Cholinergic enhancement for attention")
            
        if biomarkers.get('wPLI', 0) < 0.2:
            suggestions.append("Dopaminergic modulation for connectivity")
            
        return suggestions
    
    def circadian_rhythm_intervention(self, current_phase: float) -> Dict:
        """
        Design circadian rhythm intervention
        """
        optimal_phase = np.pi  # Noon
        phase_shift = optimal_phase - current_phase
        
        intervention = {
            'light_therapy': {
                'timing': 'morning' if phase_shift > 0 else 'evening',
                'intensity': 10000,  # lux
                'duration': 30,  # minutes
                'wavelength': 480  # nm (blue light)
            },
            'melatonin': {
                'dose': 3,  # mg
                'timing': 'evening' if phase_shift > 0 else 'avoid'
            },
            'activity_schedule': {
                'exercise': 'morning' if phase_shift > 0 else 'afternoon',
                'meals': 'regular times, avoid late eating'
            }
        }
        
        return intervention


# Example usage and testing
if __name__ == "__main__":
    print("Testing Multi-Path Coupling Hamiltonian System...")
    
    # Initialize system
    hamiltonian = MultiPathCouplingHamiltonian(
        N_neurons=50,
        N_glia=20,
        N_vascular=10
    )
    
    # Calculate total energy
    H_total = hamiltonian.total_hamiltonian()
    print(f"Initial Hamiltonian energy: {H_total:.3f}")
    
    # Evolve system
    for _ in range(100):
        hamiltonian.evolve_system(dt=0.01)
    
    H_final = hamiltonian.total_hamiltonian()
    print(f"Final Hamiltonian energy: {H_final:.3f}")
    
    print("\nTesting Clinical Biomarkers...")
    
    # Generate test data
    n_channels = 8
    n_samples = 1000
    test_signals = np.random.randn(n_channels, n_samples)
    
    # Add some synchronization
    common_signal = np.sin(2 * np.pi * 10 * np.arange(n_samples) / 1000)
    for i in range(n_channels):
        test_signals[i] += 0.5 * common_signal
    
    # Calculate biomarkers
    biomarkers = ClinicalBiomarkers()
    plv = biomarkers.phase_locking_value(test_signals)
    wpli = biomarkers.weighted_phase_lag_index(test_signals)
    criticality = biomarkers.criticality_markers(test_signals)
    
    print(f"PLV (alpha band): {plv:.3f}")
    print(f"wPLI: {wpli:.3f}")
    print(f"Criticality state: {criticality['state']}")
    print(f"Branching parameter: {criticality['branching_parameter']:.3f}")
    
    # Pathological detection
    detections = biomarkers.pathological_detection(test_signals)
    print(f"\nConsciousness level: {detections['consciousness_level']}")
    print(f"Clinical recommendations: {detections.get('recommendation', 'None')}")
    
    # Therapeutic interventions
    therapy = TherapeuticInterventions()
    
    # Get stimulation protocol
    stim_protocol = therapy.transcranial_stimulation_protocol(
        target_frequency=10.0,
        current_state=criticality['clinical_correlate']
    )
    print(f"\nStimulation protocol: {stim_protocol}")
    
    # Get pharmacological suggestions
    pharma = therapy.pharmacological_modulation(detections['biomarker_values'])
    if pharma:
        print(f"Pharmacological suggestions: {pharma[0]}")
