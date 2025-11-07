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
    Implements the total coupling Hamiltonian for a multi-path neural system.

    This class models the energy landscape of a neural system by integrating five
    distinct coupling mechanisms: chemical synaptic, gap junction, ephaptic field,
    glial calcium wave, and vascular (hemodynamic) coupling. The total energy,
    or Hamiltonian, of the system is the sum of these components.

    Attributes:
        N_neurons (int): The number of neurons in the simulation.
        N_glia (int): The number of glial cells in the simulation.
        N_vascular (int): The number of vascular compartments in the simulation.
        J_synaptic (np.ndarray): The synaptic coupling matrix [N_neurons x N_neurons].
        g_gap (np.ndarray): The gap junction conductance matrix [N_neurons x N_neurons].
        neuron_states (np.ndarray): The current activity state of each neuron.
        glia_calcium (np.ndarray): The calcium concentration in each glial cell.
        blood_flow (np.ndarray): The blood flow in each vascular compartment.
    """
    
    def __init__(self, N_neurons: int, N_glia: int, N_vascular: int):
        """
        Initializes the MultiPathCouplingHamiltonian.

        Args:
            N_neurons (int): The number of neurons in the system.
            N_glia (int): The number of glial cells.
            N_vascular (int): The number of vascular compartments.
        """
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
        """
        Initializes the synaptic coupling matrix J, enforcing Dale's law.

        Dale's law states that a neuron releases the same neurotransmitter(s) at
        all of its synapses. Here, we model this by setting 80% of neurons to be
        excitatory (positive coupling) and 20% to be inhibitory (negative coupling).

        Returns:
            np.ndarray: The [N_neurons x N_neurons] synaptic coupling matrix.
        """
        J = np.random.randn(self.N_neurons, self.N_neurons) * 0.1
        
        # Enforce Dale's law: 80% excitatory, 20% inhibitory
        n_excitatory = int(0.8 * self.N_neurons)
        J[:n_excitatory, :] = np.abs(J[:n_excitatory, :])
        J[n_excitatory:, :] = -np.abs(J[n_excitatory:, :])
        
        # Remove self-connections
        np.fill_diagonal(J, 0)
        
        return J
    
    def _initialize_gap_junctions(self) -> np.ndarray:
        """
        Initializes the gap junction conductance matrix g.

        Gap junctions are primarily established between inhibitory interneurons. This
        method creates a sparse matrix representing the electrical connections.

        Returns:
            np.ndarray: The [N_neurons x N_neurons] gap junction conductance matrix.
        """
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
        Calculates the chemical synaptic Hamiltonian.

        Args:
            sigma (np.ndarray): The current states of the neurons.

        Returns:
            float: The energy contribution from chemical synapses.
        """
        return -0.5 * np.sum(sigma @ self.J_synaptic @ sigma)
    
    def H_gap(self, V: np.ndarray) -> float:
        """
        Calculates the gap junction Hamiltonian.

        Args:
            V (np.ndarray): The membrane potentials of the neurons.

        Returns:
            float: The energy contribution from gap junctions.
        """
        energy = 0
        for i in range(self.N_neurons):
            for j in range(i + 1, self.N_neurons):
                if self.g_gap[i, j] > 0:
                    energy -= self.g_gap[i, j] * (V[i] - V[j]) ** 2
        return energy
    
    def H_ephaptic(self, E_field: np.ndarray) -> float:
        """
        Calculates the ephaptic field Hamiltonian.

        Args:
            E_field (np.ndarray): The local electric field.

        Returns:
            float: The energy contribution from the electric field.
        """
        epsilon = self.epsilon_0 * self.tissue_permittivity
        return -0.5 * epsilon * np.sum(np.abs(E_field) ** 2)
    
    def H_glial(self, Ca_concentration: np.ndarray) -> float:
        """
        Calculates the glial calcium wave Hamiltonian.

        Args:
            Ca_concentration (np.ndarray): The calcium concentrations in glial cells.

        Returns:
            float: The energy contribution from glial cell activity.
        """
        alpha = 0.01  # Coupling constant
        return -alpha * np.sum(Ca_concentration)
    
    def H_vascular(self, flow: np.ndarray) -> float:
        """
        Calculates the hemodynamic (vascular) coupling Hamiltonian.

        Args:
            flow (np.ndarray): The blood flow in vascular compartments.

        Returns:
            float: The energy contribution from neurovascular coupling.
        """
        beta = 0.1  # Neurovascular coupling constant
        return -beta * np.sum(flow)
    
    def total_hamiltonian(self) -> float:
        """
        Calculates the total system Hamiltonian.

        Returns:
            float: The total energy of the coupled system.
        """
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
        Calculates the local electric field from neural activity.

        This is a simplified model where the field is proportional to the gradient
        of neural activity across a 2D grid.

        Args:
            neural_states (np.ndarray): The current activity of neurons.

        Returns:
            np.ndarray: The magnitude of the electric field at each neuron's location.
        """
        grid_size = int(np.sqrt(self.N_neurons))
        if grid_size ** 2 < self.N_neurons:
            grid_size += 1
        
        padded_states = np.zeros(grid_size ** 2)
        padded_states[:self.N_neurons] = neural_states
        
        grid = padded_states.reshape(grid_size, grid_size)
        
        E_y, E_x = np.gradient(grid)
        E_field = np.sqrt(E_x ** 2 + E_y ** 2).flatten()[:self.N_neurons]
        
        return E_field
    
    def evolve_system(self, dt: float = 0.001) -> None:
        """
        Evolves the coupled system forward in time by one step.

        Args:
            dt (float): The time step for the simulation.
        """
        dS_dt = -np.gradient(self.total_hamiltonian())
        self.neuron_states += dS_dt * dt
        
        Ca_diffusion = 0.1
        for i in range(self.N_glia):
            if i > 0:
                self.glia_calcium[i] += Ca_diffusion * (self.glia_calcium[i-1] - self.glia_calcium[i]) * dt
            if i < self.N_glia - 1:
                self.glia_calcium[i] += Ca_diffusion * (self.glia_calcium[i+1] - self.glia_calcium[i]) * dt
        
        tau_neurovasc = 1.0
        target_flow = 40 + 20 * np.mean(np.abs(self.neuron_states))
        self.blood_flow += (target_flow - self.blood_flow) * dt / tau_neurovasc

class ClinicalBiomarkers:
    """
    Calculates clinical biomarkers from neural signals to assess consciousness states.
    """
    
    def __init__(self):
        """Initializes the ClinicalBiomarkers calculator."""
        self.biomarkers = {}
        
    def phase_locking_value(self, signals: np.ndarray, freq_band: Tuple[float, float] = (8, 12)) -> float:
        """
        Calculates the Phase Locking Value (PLV) for a given frequency band.

        Args:
            signals (np.ndarray): A [channels x samples] array of neural signals.
            freq_band (Tuple[float, float]): The frequency band of interest (e.g., alpha band).

        Returns:
            float: The average PLV across all pairs of signals.
        """
        n_signals, n_samples = signals.shape
        fs = 1000
        nyquist = fs / 2
        low = freq_band[0] / nyquist
        high = freq_band[1] / nyquist
        
        if low <= 0 or high >= 1:
            return 0.0
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        phases = np.zeros_like(signals)
        for i in range(n_signals):
            filtered = signal.filtfilt(b, a, signals[i])
            analytic = signal.hilbert(filtered)
            phases[i] = np.angle(analytic)
        
        plv_values = []
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_values.append(plv)
        
        return np.mean(plv_values) if plv_values else 0.0
    
    def weighted_phase_lag_index(self, signals: np.ndarray) -> float:
        """
        Calculates the weighted Phase Lag Index (wPLI).

        Args:
            signals (np.ndarray): A [channels x samples] array of neural signals.

        Returns:
            float: The average wPLI.
        """
        n_signals, n_samples = signals.shape
        nperseg = min(256, n_samples // 4)
        
        wpli_matrix = np.zeros((n_signals, n_signals))
        
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                _, Cxy = signal.csd(signals[i], signals[j], fs=1000, nperseg=nperseg)
                imag_csd = np.imag(Cxy)
                numerator = np.abs(np.mean(imag_csd))
                denominator = np.mean(np.abs(imag_csd))
                
                if denominator > 0:
                    wpli_matrix[i, j] = numerator / denominator
        
        return np.mean(wpli_matrix[wpli_matrix > 0])
    
    def criticality_markers(self, activity: np.ndarray) -> Dict:
        """
        Calculates markers of criticality, including the branching parameter.

        Args:
            activity (np.ndarray): A time-series of neural activity.

        Returns:
            Dict: A dictionary containing the branching parameter and clinical state.
        """
        markers = {}
        branching_ratios = []
        for t in range(len(activity) - 1):
            active_now = np.sum(activity[t] > 0)
            active_next = np.sum(activity[t + 1] > 0)
            if active_now > 0:
                branching_ratios.append(active_next / active_now)
        
        markers['branching_parameter'] = np.mean(branching_ratios) if branching_ratios else 0.0
        
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
        
        markers['dfa_exponent'] = self._detrended_fluctuation_analysis(activity.flatten())
        return markers
    
    def _detrended_fluctuation_analysis(self, time_series: np.ndarray, scales: Optional[np.ndarray] = None) -> float:
        """
        Performs Detrended Fluctuation Analysis (DFA).

        Args:
            time_series (np.ndarray): The time series to analyze.
            scales (Optional[np.ndarray]): The scales to evaluate.

        Returns:
            float: The DFA scaling exponent.
        """
        if scales is None:
            scales = np.logspace(1, 3, 20, dtype=int)
        
        y = np.cumsum(time_series - np.mean(time_series))
        
        fluctuations = []
        for scale in scales:
            if scale >= len(y): continue
            n_segments = len(y) // scale
            if n_segments == 0: continue
            
            segment_vars = []
            for i in range(n_segments):
                segment = y[i * scale:(i + 1) * scale]
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.poly1d(coeffs)(x)
                segment_var = np.sqrt(np.mean((segment - trend) ** 2))
                segment_vars.append(segment_var)
            
            fluctuations.append(np.mean(segment_vars))
        
        if len(fluctuations) > 2:
            valid_scales = scales[:len(fluctuations)]
            alpha, _ = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
            return alpha
        
        return 0.5
    
    def pathological_detection(self, neural_data: np.ndarray) -> Dict:
        """
        Detects pathological states from neural dynamics using calculated biomarkers.

        Args:
            neural_data (np.ndarray): A [channels x samples] array of neural signals.

        Returns:
            Dict: A dictionary of detected pathologies and recommendations.
        """
        detections = {}
        plv = self.phase_locking_value(neural_data)
        wpli = self.weighted_phase_lag_index(neural_data)
        criticality = self.criticality_markers(neural_data)
        
        if criticality['branching_parameter'] > 1.2:
            detections['epilepsy_risk'] = 'high'
            detections['recommendation'] = 'Consider anti-epileptic intervention'
        
        if criticality['branching_parameter'] < 0.8 and plv < 0.3:
            detections['depression_markers'] = 'present'
            detections['recommendation'] = 'Evaluate for mood disorders'
        
        if wpli < 0.2 and criticality['dfa_exponent'] < 0.6:
            detections['schizophrenia_markers'] = 'present'
            detections['recommendation'] = 'Assess connectivity disruption'
        
        consciousness_score = (plv + wpli + (1 - abs(1 - criticality['branching_parameter']))) / 3
        
        if consciousness_score > 0.7:
            detections['consciousness_level'] = 'fully_conscious'
        elif consciousness_score > 0.4:
            detections['consciousness_level'] = 'altered_consciousness'
        else:
            detections['consciousness_level'] = 'minimal_consciousness'
        
        detections['biomarker_values'] = {
            'PLV': plv, 'wPLI': wpli,
            'branching_parameter': criticality['branching_parameter'],
            'DFA_exponent': criticality['dfa_exponent']
        }
        return detections

class TherapeuticInterventions:
    """
    Designs therapeutic interventions based on tissue synchronization biomarkers.
    """
    
    def __init__(self):
        """Initializes the TherapeuticInterventions class."""
        self.protocols = {}
        
    def transcranial_stimulation_protocol(self, target_frequency: float, current_state: str) -> Dict:
        """
        Designs a transcranial stimulation protocol based on the current brain state.

        Args:
            target_frequency (float): The target frequency for stimulation.
            current_state (str): The clinical state (e.g., 'epilepsy_risk').

        Returns:
            Dict: A dictionary describing the stimulation protocol.
        """
        protocol = {
            'type': 'tACS', 'frequency': target_frequency,
            'intensity': 2.0, 'duration': 20, 'electrode_montage': 'F3-F4',
        }
        
        if current_state == 'epilepsy_risk':
            protocol.update({'type': 'tDCS', 'polarity': 'cathodal', 'intensity': 1.0})
        elif current_state == 'depression_risk':
            protocol.update({'frequency': 10.0, 'intensity': 2.5, 'electrode_montage': 'F3-Fp2'})
        elif current_state == 'schizophrenia_markers':
            protocol.update({'frequency': 40.0, 'intensity': 1.5})
            
        return protocol
    
    def pharmacological_modulation(self, biomarkers: Dict) -> List[str]:
        """
        Suggests pharmacological interventions based on criticality and synchronization.

        Args:
            biomarkers (Dict): A dictionary of biomarker values.

        Returns:
            List[str]: A list of suggested pharmacological agents.
        """
        suggestions = []
        branching = biomarkers.get('branching_parameter', 1.0)
        
        if branching > 1.1:
            suggestions.extend(["GABAergic enhancement (e.g., benzodiazepines)", "Sodium channel blockers (e.g., carbamazepine)"])
        elif branching < 0.9:
            suggestions.extend(["NMDA receptor modulation (e.g., ketamine)", "Monoamine enhancement (e.g., SSRIs)"])

        if biomarkers.get('PLV', 0) < 0.3:
            suggestions.append("Cholinergic enhancement for attention")
        if biomarkers.get('wPLI', 0) < 0.2:
            suggestions.append("Dopaminergic modulation for connectivity")
            
        return suggestions
    
    def circadian_rhythm_intervention(self, current_phase: float) -> Dict:
        """
        Designs a circadian rhythm intervention protocol.

        Args:
            current_phase (float): The current phase of the circadian rhythm.

        Returns:
            Dict: A dictionary describing the light, melatonin, and activity schedule.
        """
        optimal_phase = np.pi
        phase_shift = optimal_phase - current_phase
        
        return {
            'light_therapy': {
                'timing': 'morning' if phase_shift > 0 else 'evening', 'intensity': 10000,
                'duration': 30, 'wavelength': 480
            },
            'melatonin': {'dose': 3, 'timing': 'evening' if phase_shift > 0 else 'avoid'},
            'activity_schedule': {'exercise': 'morning' if phase_shift > 0 else 'afternoon', 'meals': 'regular times, avoid late eating'}
        }

if __name__ == "__main__":
    print("Testing Multi-Path Coupling Hamiltonian System...")
    hamiltonian = MultiPathCouplingHamiltonian(N_neurons=50, N_glia=20, N_vascular=10)
    H_total = hamiltonian.total_hamiltonian()
    print(f"Initial Hamiltonian energy: {H_total:.3f}")
    
    for _ in range(100):
        hamiltonian.evolve_system(dt=0.01)
    
    H_final = hamiltonian.total_hamiltonian()
    print(f"Final Hamiltonian energy: {H_final:.3f}")
    
    print("\nTesting Clinical Biomarkers...")
    n_channels, n_samples = 8, 1000
    test_signals = np.random.randn(n_channels, n_samples)
    common_signal = np.sin(2 * np.pi * 10 * np.arange(n_samples) / 1000)
    for i in range(n_channels):
        test_signals[i] += 0.5 * common_signal
    
    biomarkers = ClinicalBiomarkers()
    plv = biomarkers.phase_locking_value(test_signals)
    wpli = biomarkers.weighted_phase_lag_index(test_signals)
    criticality = biomarkers.criticality_markers(test_signals)
    
    print(f"PLV (alpha band): {plv:.3f}")
    print(f"wPLI: {wpli:.3f}")
    print(f"Criticality state: {criticality['state']}")
    
    detections = biomarkers.pathological_detection(test_signals)
    print(f"\nConsciousness level: {detections['consciousness_level']}")
    
    therapy = TherapeuticInterventions()
    stim_protocol = therapy.transcranial_stimulation_protocol(target_frequency=10.0, current_state=criticality['clinical_correlate'])
    print(f"\nStimulation protocol: {stim_protocol}")
    
    pharma = therapy.pharmacological_modulation(detections['biomarker_values'])
    if pharma:
        print(f"Pharmacological suggestions: {pharma[0]}")
