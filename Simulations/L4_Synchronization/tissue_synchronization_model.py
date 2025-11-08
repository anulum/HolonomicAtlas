"""
Tissue Synchronization Model for Consciousness Studies
Implementation of Layer 4 SCPN Architecture Extensions
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
    """Parameters for individual cellular oscillators"""
    intrinsic_freq: float  # ωᵢ: 10⁻⁵ to 10¹⁴ Hz range
    synaptic_coupling: float  # K: synaptic coupling strength
    field_coupling: float  # ζ: consciousness field coupling
    ephaptic_strength: float  # λ: ephaptic field strength
    noise_amplitude: float  # noise term amplitude
    
class HierarchicalOscillatorNetwork:
    """
    Multilayer Kuramoto Model with Field Coupling
    Implements: dφᵢ/dt = ωᵢ + (K/N)∑ⱼ sin(φⱼ - φᵢ) + ζΨₛcos(φᵢ) + λ∑ₖ Gᵢₖ(φₖ - φᵢ) + ηᵢ(t)
    """
    
    def __init__(self, N: int, topology: str = 'small_world'):
        self.N = N
        self.topology = topology
        self.phases = np.random.uniform(0, 2*np.pi, N)
        self.frequencies = self._initialize_frequencies()
        self.network = self._build_topology()
        self.gap_junction_matrix = self._create_gap_junctions()
        
        # Coupling parameters
        self.params = OscillatorParameters(
            intrinsic_freq=7.83,  # Schumann resonance base
            synaptic_coupling=1.0,
            field_coupling=0.5,
            ephaptic_strength=0.3,
            noise_amplitude=0.1
        )
        
        # Track dynamics
        self.synchronization_history = []
        self.avalanche_sizes = []
        self.metastability_index = []
        
    def _initialize_frequencies(self) -> np.ndarray:
        """Initialize intrinsic frequencies across biological scales"""
        # Multi-scale frequency distribution (log-normal)
        # Ranges from metabolic (10⁻⁵ Hz) to molecular vibrations (10¹⁴ Hz)
        log_freqs = np.random.normal(1, 2, self.N)  # Centered around 10 Hz
        return 10 ** log_freqs
    
    def _build_topology(self) -> nx.Graph:
        """Create network topology (small-world, rich-club, etc.)"""
        if self.topology == 'small_world':
            # Watts-Strogatz small-world network
            G = nx.watts_strogatz_graph(self.N, k=6, p=0.3)
        elif self.topology == 'rich_club':
            # Rich-club topology (hub-based)
            G = nx.barabasi_albert_graph(self.N, m=3)
        elif self.topology == 'modular':
            # Modular network with communities
            G = self._create_modular_network()
        else:
            # Default: random network
            G = nx.erdos_renyi_graph(self.N, p=0.1)
        
        return G
    
    def _create_modular_network(self) -> nx.Graph:
        """Create modular network structure mimicking brain organization"""
        modules = 5  # Number of modules
        module_size = self.N // modules
        G = nx.Graph()
        
        # Create dense modules
        for m in range(modules):
            start = m * module_size
            end = start + module_size
            for i in range(start, end):
                for j in range(i + 1, end):
                    if np.random.random() < 0.6:  # High intra-module connectivity
                        G.add_edge(i, j)
        
        # Add sparse inter-module connections
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if i // module_size != j // module_size:
                    if np.random.random() < 0.05:  # Low inter-module connectivity
                        G.add_edge(i, j)
        
        return G
    
    def _create_gap_junctions(self) -> np.ndarray:
        """Create gap junction conductance matrix"""
        G = np.zeros((self.N, self.N))
        
        # Gap junctions follow network topology with spatial decay
        for edge in self.network.edges():
            i, j = edge
            # Conductance decreases with distance
            distance = np.abs(i - j) / self.N
            G[i, j] = G[j, i] = np.exp(-distance / 0.1)
        
        return G
    
    def kuramoto_dynamics(self, t: float, phases: np.ndarray) -> np.ndarray:
        """
        Core Kuramoto dynamics with multiple coupling mechanisms
        """
        dphase_dt = np.zeros(self.N)
        
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.network).toarray()
        
        for i in range(self.N):
            # Intrinsic frequency
            dphase_dt[i] = self.frequencies[i]
            
            # Synaptic coupling (Kuramoto term)
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                coupling_sum = np.sum(np.sin(phases[neighbors] - phases[i]))
                dphase_dt[i] += self.params.synaptic_coupling * coupling_sum / len(neighbors)
            
            # Consciousness field coupling
            mean_field = np.mean(np.exp(1j * phases))
            dphase_dt[i] += self.params.field_coupling * np.abs(mean_field) * np.cos(phases[i])
            
            # Gap junction coupling
            gap_coupling = np.sum(self.gap_junction_matrix[i] * (phases - phases[i]))
            dphase_dt[i] += self.params.ephaptic_strength * gap_coupling
            
            # Colored noise
            noise = self.params.noise_amplitude * self._generate_colored_noise()
            dphase_dt[i] += noise
        
        return dphase_dt
    
    def _generate_colored_noise(self) -> float:
        """Generate 1/f colored noise for biological realism"""
        # Simple approximation of 1/f noise
        white = np.random.randn()
        pink = 0.9 * getattr(self, '_prev_noise', 0) + 0.1 * white
        self._prev_noise = pink
        return pink
    
    def calculate_order_parameter(self) -> complex:
        """Calculate Kuramoto order parameter"""
        return np.mean(np.exp(1j * self.phases))
    
    def calculate_synchronization(self) -> float:
        """Calculate global synchronization level"""
        return np.abs(self.calculate_order_parameter())
    
    def detect_avalanches(self, threshold: float = 0.1) -> List[int]:
        """
        Detect neuronal avalanches in the network
        Returns list of avalanche sizes
        """
        # Calculate phase velocities
        velocities = np.gradient(self.phases)
        
        # Identify active sites
        active = np.abs(velocities) > threshold
        
        # Find connected components of active sites
        active_graph = self.network.subgraph(np.where(active)[0])
        components = list(nx.connected_components(active_graph))
        
        avalanche_sizes = [len(comp) for comp in components]
        return avalanche_sizes
    
    def calculate_metastability_index(self, window: int = 100) -> float:
        """
        Calculate metastability index: MI = ⟨(δφ)²⟩/⟨|δφ|⟩²
        Quantifies switching between integrated/segregated states
        """
        if len(self.synchronization_history) < window:
            return 0.0
        
        recent_sync = np.array(self.synchronization_history[-window:])
        fluctuations = recent_sync - np.mean(recent_sync)
        
        numerator = np.mean(fluctuations ** 2)
        denominator = np.mean(np.abs(fluctuations)) ** 2
        
        if denominator > 0:
            return numerator / denominator
        return 0.0
    
    def simulate(self, duration: float, dt: float = 0.001) -> Dict:
        """
        Run simulation and collect metrics
        """
        time_points = np.arange(0, duration, dt)
        
        # Solve ODE
        solution = solve_ivp(
            self.kuramoto_dynamics,
            [0, duration],
            self.phases,
            t_eval=time_points,
            method='RK45'
        )
        
        # Store results
        results = {
            'time': solution.t,
            'phases': solution.y,
            'synchronization': [],
            'avalanches': [],
            'metastability': []
        }
        
        # Calculate metrics at each time point
        for i, t in enumerate(solution.t):
            self.phases = solution.y[:, i]
            
            # Track synchronization
            sync = self.calculate_synchronization()
            results['synchronization'].append(sync)
            self.synchronization_history.append(sync)
            
            # Detect avalanches periodically
            if i % 100 == 0:
                avalanches = self.detect_avalanches()
                self.avalanche_sizes.extend(avalanches)
                results['avalanches'].append(avalanches)
            
            # Calculate metastability
            if i % 10 == 0:
                mi = self.calculate_metastability_index()
                results['metastability'].append(mi)
                self.metastability_index.append(mi)
        
        return results

class QuasicriticalDynamics:
    """
    Implements Griffiths phase dynamics and criticality measures
    """
    
    def __init__(self, network_size: int):
        self.N = network_size
        self.branching_parameter = 1.0  # σ = 1 at criticality
        self.tau_exponent = 1.5  # Power law exponent
        
    def avalanche_distribution(self, sizes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate avalanche size distribution: P(s) ~ s⁻τ exp(-s/s₀)
        """
        if not sizes:
            return np.array([]), np.array([])
        
        # Create histogram
        hist, bins = np.histogram(sizes, bins=50)
        
        # Cutoff scale
        s_cutoff = self.N ** (1 / 1.5)
        
        # Theoretical distribution
        s_values = (bins[:-1] + bins[1:]) / 2
        theoretical = s_values ** (-self.tau_exponent) * np.exp(-s_values / s_cutoff)
        
        return hist, theoretical
    
    def calculate_branching_ratio(self, activity: np.ndarray) -> float:
        """
        Calculate branching parameter σ
        σ > 1: supercritical (e.g., epilepsy)
        σ = 1: critical
        σ < 1: subcritical (e.g., depression)
        """
        descendants = []
        
        for t in range(len(activity) - 1):
            active_now = np.sum(activity[t] > 0)
            active_next = np.sum(activity[t + 1] > 0)
            
            if active_now > 0:
                descendants.append(active_next / active_now)
        
        if descendants:
            return np.mean(descendants)
        return 0.0
    
    def griffiths_phase_detection(self, time_series: np.ndarray) -> Dict:
        """
        Detect Griffiths phase characteristics
        """
        # Calculate autocorrelation
        autocorr = np.correlate(time_series, time_series, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Fit power law decay
        x = np.arange(1, len(autocorr) + 1)
        log_x = np.log(x[autocorr > 0])
        log_y = np.log(autocorr[autocorr > 0])
        
        if len(log_x) > 10:
            slope, intercept = np.polyfit(log_x[:100], log_y[:100], 1)
            decay_exponent = -slope
        else:
            decay_exponent = 0.0
        
        # Check for rare regions (local deviations from criticality)
        local_variance = []
        window = 50
        for i in range(0, len(time_series) - window, window):
            local_var = np.var(time_series[i:i+window])
            local_variance.append(local_var)
        
        return {
            'decay_exponent': decay_exponent,
            'is_griffiths': 0.1 < decay_exponent < 1.0,  # Slow decay characteristic
            'heterogeneity': np.std(local_variance) / np.mean(local_variance) if local_variance else 0
        }

class CosmologicalEntrainment:
    """
    Models entrainment with cosmological rhythms
    """
    
    def __init__(self):
        # Schumann resonance frequencies (Hz)
        self.schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]
        
        # Lunar cycle parameters
        self.lunar_period = 29.53  # days
        self.lunar_phase = 0.0
        
        # Circadian parameters
        self.circadian_period = 24.0  # hours
        
    def schumann_resonance_field(self, t: float, location: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Generate Schumann resonance field at time t
        """
        field = np.zeros(len(self.schumann_freqs))
        
        for i, freq in enumerate(self.schumann_freqs):
            # Add geomagnetic modulation
            geomag_modulation = 1.0 + 0.1 * np.sin(2 * np.pi * t / (24 * 3600))
            
            # Amplitude decreases with harmonic number
            amplitude = geomag_modulation / (i + 1)
            
            field[i] = amplitude * np.sin(2 * np.pi * freq * t)
        
        return field
    
    def lunar_influence(self, t_days: float) -> Dict:
        """
        Calculate lunar phase influence on biological rhythms
        """
        # Current lunar phase
        phase = (t_days % self.lunar_period) / self.lunar_period * 2 * np.pi
        
        # Gravitational influence (simplified)
        gravitational = 1.0 + 0.1 * np.cos(phase)
        
        # Electromagnetic influence (subtle)
        electromagnetic = 0.01 * np.sin(phase)
        
        return {
            'phase': phase,
            'gravitational': gravitational,
            'electromagnetic': electromagnetic,
            'illumination': (1 + np.cos(phase)) / 2  # 0 = new moon, 1 = full moon
        }
    
    def circadian_modulation(self, t_hours: float) -> float:
        """
        Circadian rhythm modulation
        """
        phase = (t_hours % self.circadian_period) / self.circadian_period * 2 * np.pi
        
        # Core body temperature rhythm as proxy
        # Minimum at ~4-6 AM, maximum at ~7-9 PM
        return 37.0 + 0.5 * np.sin(phase - np.pi/2)

# Test the implementation
if __name__ == "__main__":
    # Create a small test network
    network = HierarchicalOscillatorNetwork(N=100, topology='small_world')
    
    # Run simulation
    print("Starting tissue synchronization simulation...")
    results = network.simulate(duration=10.0, dt=0.01)
    
    # Analyze criticality
    qc = QuasicriticalDynamics(network_size=100)
    
    # Check avalanche distribution
    if network.avalanche_sizes:
        hist, theory = qc.avalanche_distribution(network.avalanche_sizes)
        print(f"Detected {len(network.avalanche_sizes)} avalanches")
    
    # Calculate final metrics
    final_sync = network.calculate_synchronization()
    final_meta = network.calculate_metastability_index()
    
    print(f"Final synchronization: {final_sync:.3f}")
    print(f"Metastability index: {final_meta:.3f}")
    
    # Test cosmological entrainment
    cosmo = CosmologicalEntrainment()
    schumann = cosmo.schumann_resonance_field(0)
    lunar = cosmo.lunar_influence(15)  # Day 15 of lunar cycle
    
    print(f"Schumann field amplitudes: {schumann}")
    print(f"Lunar phase: {lunar['phase']:.2f} rad, Illumination: {lunar['illumination']:.2%}")
