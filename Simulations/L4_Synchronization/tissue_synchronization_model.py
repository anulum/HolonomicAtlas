"""
Tissue Synchronization Model for SCPN Layer 4 Consciousness Studies.

This module implements the core components of the tissue synchronization model,
including the hierarchical oscillator network, quasicritical dynamics, and
cosmological entrainment, based on the SCPN Layer 4 architecture extensions.
"""

import numpy as np
import scipy as sp
from scipy import signal, stats
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import networkx as nx

@dataclass
class OscillatorParameters:
    """
    Data class for parameters of individual cellular oscillators.

    Attributes:
        intrinsic_freq (float): The natural frequency of the oscillator (ωᵢ).
        synaptic_coupling (float): The strength of synaptic coupling (K).
        field_coupling (float): The coupling strength to the consciousness field (ζ).
        ephaptic_strength (float): The strength of the ephaptic field coupling (λ).
        noise_amplitude (float): The amplitude of the stochastic noise term.
    """
    intrinsic_freq: float
    synaptic_coupling: float
    field_coupling: float
    ephaptic_strength: float
    noise_amplitude: float
    
class HierarchicalOscillatorNetwork:
    """
    A multilayer Kuramoto model with field coupling for tissue synchronization.

    This class implements a hierarchical network of oscillators with multiple coupling
    mechanisms, including synaptic, field, and ephaptic coupling, to model the
    dynamics of tissue synchronization in the brain.

    Attributes:
        N (int): The number of oscillators in the network.
        topology (str): The network topology (e.g., 'small_world', 'rich_club').
        phases (np.ndarray): The current phases of the oscillators.
        frequencies (np.ndarray): The intrinsic frequencies of the oscillators.
        network (nx.Graph): The networkX graph representing the network topology.
        params (OscillatorParameters): The parameters for the oscillator dynamics.
    """
    
    def __init__(self, N: int, topology: str = 'small_world'):
        """
        Initializes the HierarchicalOscillatorNetwork.

        Args:
            N (int): The number of oscillators in the network.
            topology (str): The network topology to use.
        """
        self.N, self.topology = N, topology
        self.phases = np.random.uniform(0, 2*np.pi, N)
        self.frequencies = self._initialize_frequencies()
        self.network = self._build_topology()
        self.gap_junction_matrix = self._create_gap_junctions()
        self.params = OscillatorParameters(intrinsic_freq=7.83, synaptic_coupling=1.0, field_coupling=0.5, ephaptic_strength=0.3, noise_amplitude=0.1)
        self.synchronization_history, self.avalanche_sizes, self.metastability_index = [], [], []
        
    def _initialize_frequencies(self) -> np.ndarray:
        """Initializes oscillator frequencies with a log-normal distribution."""
        return 10 ** np.random.normal(1, 2, self.N)
    
    def _build_topology(self) -> nx.Graph:
        """Builds the network topology."""
        if self.topology == 'small_world': return nx.watts_strogatz_graph(self.N, k=6, p=0.3)
        if self.topology == 'rich_club': return nx.barabasi_albert_graph(self.N, m=3)
        if self.topology == 'modular': return self._create_modular_network()
        return nx.erdos_renyi_graph(self.N, p=0.1)
    
    def _create_modular_network(self) -> nx.Graph:
        """Creates a modular network with dense intra-module and sparse inter-module connections."""
        modules, module_size = 5, self.N // 5
        G = nx.Graph()
        for m in range(modules):
            start, end = m * module_size, (m + 1) * module_size
            for i in range(start, end):
                for j in range(i + 1, end):
                    if np.random.random() < 0.6: G.add_edge(i, j)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if i // module_size != j // module_size and np.random.random() < 0.05:
                    G.add_edge(i, j)
        return G
    
    def _create_gap_junctions(self) -> np.ndarray:
        """Creates the gap junction conductance matrix with spatial decay."""
        G = np.zeros((self.N, self.N))
        for i, j in self.network.edges():
            distance = np.abs(i - j) / self.N
            G[i, j] = G[j, i] = np.exp(-distance / 0.1)
        return G
    
    def kuramoto_dynamics(self, t: float, phases: np.ndarray) -> np.ndarray:
        """
        Calculates the time derivative of the oscillator phases.

        Args:
            t (float): The current time.
            phases (np.ndarray): The current phases of the oscillators.

        Returns:
            np.ndarray: The time derivatives of the phases (dφ/dt).
        """
        dphase_dt = np.zeros(self.N)
        adj_matrix = nx.adjacency_matrix(self.network).toarray()
        for i in range(self.N):
            dphase_dt[i] = self.frequencies[i]
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                dphase_dt[i] += self.params.synaptic_coupling * np.sum(np.sin(phases[neighbors] - phases[i])) / len(neighbors)
            dphase_dt[i] += self.params.field_coupling * np.abs(np.mean(np.exp(1j * phases))) * np.cos(phases[i])
            dphase_dt[i] += self.params.ephaptic_strength * np.sum(self.gap_junction_matrix[i] * (phases - phases[i]))
            dphase_dt[i] += self.params.noise_amplitude * self._generate_colored_noise()
        return dphase_dt
    
    def _generate_colored_noise(self) -> float:
        """Generates 1/f (pink) noise."""
        white = np.random.randn()
        pink = 0.9 * getattr(self, '_prev_noise', 0) + 0.1 * white
        self._prev_noise = pink
        return pink
    
    def calculate_synchronization(self) -> float:
        """Calculates the global synchronization level (Kuramoto order parameter)."""
        return np.abs(np.mean(np.exp(1j * self.phases)))
    
    def detect_avalanches(self, threshold: float = 0.1) -> List[int]:
        """Detects neuronal avalanches in the network."""
        velocities = np.gradient(self.phases)
        active = np.abs(velocities) > threshold
        active_graph = self.network.subgraph(np.where(active)[0])
        return [len(comp) for comp in nx.connected_components(active_graph)]
    
    def calculate_metastability_index(self, window: int = 100) -> float:
        """Calculates the metastability index of the network dynamics."""
        if len(self.synchronization_history) < window: return 0.0
        recent_sync = np.array(self.synchronization_history[-window:])
        fluctuations = recent_sync - np.mean(recent_sync)
        numerator = np.mean(fluctuations ** 2)
        denominator = np.mean(np.abs(fluctuations)) ** 2
        return numerator / denominator if denominator > 0 else 0.0
    
    def simulate(self, duration: float, dt: float = 0.001) -> Dict:
        """
        Runs the simulation and collects metrics.

        Args:
            duration (float): The total duration of the simulation.
            dt (float): The time step for the simulation.

        Returns:
            Dict: A dictionary of simulation results.
        """
        solution = solve_ivp(self.kuramoto_dynamics, [0, duration], self.phases, t_eval=np.arange(0, duration, dt), method='RK45')
        results = {'time': solution.t, 'phases': solution.y, 'synchronization': [], 'avalanches': [], 'metastability': []}
        for i, t in enumerate(solution.t):
            self.phases = solution.y[:, i]
            sync = self.calculate_synchronization()
            results['synchronization'].append(sync)
            self.synchronization_history.append(sync)
            if i % 100 == 0:
                avalanches = self.detect_avalanches()
                self.avalanche_sizes.extend(avalanches)
                results['avalanches'].append(avalanches)
            if i % 10 == 0:
                mi = self.calculate_metastability_index()
                results['metastability'].append(mi)
                self.metastability_index.append(mi)
        return results

class QuasicriticalDynamics:
    """
    Implements Griffiths phase dynamics and criticality measures.

    Attributes:
        N (int): The size of the network.
        branching_parameter (float): The branching parameter (σ).
        tau_exponent (float): The power-law exponent for avalanche sizes (τ).
    """
    
    def __init__(self, network_size: int):
        """Initializes the QuasicriticalDynamics module."""
        self.N, self.branching_parameter, self.tau_exponent = network_size, 1.0, 1.5
        
    def avalanche_distribution(self, sizes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the avalanche size distribution."""
        if not sizes: return np.array([]), np.array([])
        hist, bins = np.histogram(sizes, bins=50)
        s_cutoff = self.N ** (1 / 1.5)
        s_values = (bins[:-1] + bins[1:]) / 2
        theoretical = s_values ** (-self.tau_exponent) * np.exp(-s_values / s_cutoff)
        return hist, theoretical
    
    def calculate_branching_ratio(self, activity: np.ndarray) -> float:
        """Calculates the branching parameter (σ)."""
        descendants = [np.sum(activity[t + 1] > 0) / np.sum(activity[t] > 0) for t in range(len(activity) - 1) if np.sum(activity[t] > 0) > 0]
        return np.mean(descendants) if descendants else 0.0
    
    def griffiths_phase_detection(self, time_series: np.ndarray) -> Dict:
        """Detects characteristics of the Griffiths phase."""
        autocorr = np.correlate(time_series, time_series, mode='full')[len(time_series)//2:]
        log_x, log_y = np.log(np.arange(1, len(autocorr) + 1)[autocorr > 0]), np.log(autocorr[autocorr > 0])
        decay_exponent = -np.polyfit(log_x[:100], log_y[:100], 1)[0] if len(log_x) > 10 else 0.0
        window = 50
        local_variance = [np.var(time_series[i:i+window]) for i in range(0, len(time_series) - window, window)]
        return {'decay_exponent': decay_exponent, 'is_griffiths': 0.1 < decay_exponent < 1.0, 'heterogeneity': np.std(local_variance) / np.mean(local_variance) if local_variance else 0}

class CosmologicalEntrainment:
    """
    Models the entrainment of biological rhythms with cosmological cycles.

    This class provides methods to generate signals for Schumann resonances,
    lunar phases, and circadian rhythms.
    """
    
    def __init__(self):
        """Initializes the CosmologicalEntrainment module."""
        self.schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]
        self.lunar_period, self.circadian_period = 29.53, 24.0
        
    def schumann_resonance_field(self, t: float, location: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """Generates the Schumann resonance field at a given time."""
        field = np.zeros(len(self.schumann_freqs))
        for i, freq in enumerate(self.schumann_freqs):
            geomag_modulation = 1.0 + 0.1 * np.sin(2 * np.pi * t / (24 * 3600))
            field[i] = (geomag_modulation / (i + 1)) * np.sin(2 * np.pi * freq * t)
        return field
    
    def lunar_influence(self, t_days: float) -> Dict:
        """Calculates the lunar phase influence on biological rhythms."""
        phase = (t_days % self.lunar_period) / self.lunar_period * 2 * np.pi
        return {'phase': phase, 'gravitational': 1.0 + 0.1 * np.cos(phase), 'electromagnetic': 0.01 * np.sin(phase), 'illumination': (1 + np.cos(phase)) / 2}
    
    def circadian_modulation(self, t_hours: float) -> float:
        """Calculates the circadian rhythm modulation."""
        phase = (t_hours % self.circadian_period) / self.circadian_period * 2 * np.pi
        return 37.0 + 0.5 * np.sin(phase - np.pi/2)

if __name__ == "__main__":
    network = HierarchicalOscillatorNetwork(N=100, topology='small_world')
    print("Starting tissue synchronization simulation...")
    results = network.simulate(duration=10.0, dt=0.01)
    
    qc = QuasicriticalDynamics(network_size=100)
    if network.avalanche_sizes:
        hist, theory = qc.avalanche_distribution(network.avalanche_sizes)
        print(f"Detected {len(network.avalanche_sizes)} avalanches")
    
    print(f"Final synchronization: {network.calculate_synchronization():.3f}")
    print(f"Metastability index: {network.calculate_metastability_index():.3f}")
    
    cosmo = CosmologicalEntrainment()
    print(f"Schumann field amplitudes: {cosmo.schumann_resonance_field(0)}")
    print(f"Lunar phase influence: {cosmo.lunar_influence(15)}")
