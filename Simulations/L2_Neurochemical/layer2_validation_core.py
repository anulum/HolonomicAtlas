"""
Layer 2 (Neurochemical-Neurological) Experimental Validation Suite
Core Architecture & Framework - Module 1 of 6

This module provides the foundational classes and structures for validating
the SCPN Layer 2 theoretical framework through computational experiments.

Author: Generated for SCPN Manuscript Validation
Date: 2025-11-07
Version: 1.0.0

Architecture Overview:
---------------------
The validation suite is organized into 6 modular components:
1. Core Framework (this module) - Base classes and infrastructure
2. Quantum-Classical Validators - L1-L2 transition tests
3. Neurotransmitter Tests - NT dynamics and oscillations
4. Glial Network Validators - Astrocyte, oligodendrocyte dynamics
5. Integration Tests - Multi-scale and cross-layer validation
6. Analysis & Visualization - Statistical analysis and reporting

Key Design Principles:
---------------------
- Modularity: Each component is independently functional
- Extensibility: Easy to add new experiments
- Reproducibility: All parameters documented and versioned
- Performance: Optimized for large-scale simulations
- Validation: Built-in consistency checks at each level
"""

import numpy as np
import scipy as sp
from scipy import integrate, signal, stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import json
import logging
from pathlib import Path
import h5py
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Layer2Validation')


# ============================================================================
# SECTION 1: FUNDAMENTAL CONSTANTS AND PARAMETERS
# ============================================================================

class PhysicalConstants:
    """Physical constants used throughout Layer 2 validation"""
    
    # Fundamental constants
    h = 6.62607015e-34  # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    e = 1.602176634e-19  # Elementary charge (C)
    
    # Biological temperature range
    T_min = 290  # K (17°C - hypothermia threshold)
    T_body = 310  # K (37°C - normal body temp)
    T_max = 320  # K (47°C - hyperthermia threshold)
    
    # Energy scales
    eV_to_J = 1.602176634e-19
    meV_to_J = 1.602176634e-22
    
    # Time scales (seconds)
    QUANTUM_TIMESCALE = 1e-6  # Quantum coherence (~1 μs)
    MOLECULAR_TIMESCALE = 1e-3  # Molecular dynamics (~1 ms)
    SYNAPTIC_TIMESCALE = 1e-2  # Synaptic transmission (~10 ms)
    OSCILLATION_TIMESCALE = 1.0  # Neural oscillations (~1 s)
    CIRCADIAN_TIMESCALE = 86400  # Circadian rhythm (~24 h)
    
    # Spatial scales (meters)
    QUANTUM_SCALE = 1e-9  # Nanometer (quantum/molecular)
    SYNAPTIC_SCALE = 1e-6  # Micrometer (synapse)
    CELLULAR_SCALE = 1e-5  # 10 micrometers (cell body)
    NETWORK_SCALE = 1e-3  # Millimeter (local network)
    
    # Coupling strengths (eV)
    COUPLING_MIN = 0.001
    COUPLING_MAX = 1.0
    
    # Coherence times (seconds)
    COHERENCE_MIN = 0.0001  # 0.1 ms
    COHERENCE_MAX = 0.1     # 100 ms


class NeurotransmitterParams:
    """Standard parameters for major neurotransmitter systems"""
    
    # Concentration ranges (Molar)
    CONCENTRATIONS = {
        'glutamate': {'baseline': 1.0e-3, 'min': 1e-9, 'max': 1e-2, 'synaptic_peak': 1e-3},
        'GABA': {'baseline': 0.3e-3, 'min': 1e-9, 'max': 5e-3, 'synaptic_peak': 3e-3},
        'dopamine': {'baseline': 1e-6, 'min': 1e-9, 'max': 1e-5, 'synaptic_peak': 1e-6},
        'serotonin': {'baseline': 5e-7, 'min': 1e-9, 'max': 5e-6, 'synaptic_peak': 5e-7},
        'acetylcholine': {'baseline': 1e-6, 'min': 1e-9, 'max': 1e-5, 'synaptic_peak': 1e-5},
        'norepinephrine': {'baseline': 5e-7, 'min': 1e-9, 'max': 5e-6, 'synaptic_peak': 5e-7},
    }
    
    # Kinetic parameters
    KINETICS = {
        'glutamate': {'release_rate': 0.3, 'uptake_rate': 0.1, 'diffusion': 3.3e-10},
        'GABA': {'release_rate': 0.2, 'uptake_rate': 0.05, 'diffusion': 2.5e-10},
        'dopamine': {'release_rate': 0.01, 'uptake_rate': 0.02, 'diffusion': 3.0e-10},
        'serotonin': {'release_rate': 0.01, 'uptake_rate': 0.015, 'diffusion': 2.8e-10},
        'acetylcholine': {'release_rate': 0.05, 'uptake_rate': 0.03, 'diffusion': 3.5e-10},
        'norepinephrine': {'release_rate': 0.01, 'uptake_rate': 0.015, 'diffusion': 2.9e-10},
    }
    
    # Oscillation frequency bands (Hz)
    OSCILLATION_BANDS = {
        'slow': (0.01, 0.1),      # Slow waves (serotonin modulation)
        'delta': (0.5, 4),        # Delta (sleep, plasticity)
        'theta': (4, 8),          # Theta (memory, navigation)
        'alpha': (8, 13),         # Alpha (attention, acetylcholine)
        'beta': (13, 30),         # Beta (motor, dopamine)
        'gamma': (30, 80),        # Gamma (binding, glutamate)
        'high_gamma': (80, 200),  # High gamma (local processing)
    }


class OscillationParams:
    """Parameters for neural oscillation hierarchies"""
    
    # Phase-amplitude coupling (PAC) parameters
    PAC_STRENGTH_RANGE = (0.0, 1.0)  # Normalized coupling strength
    PAC_OPTIMAL = 0.5  # Optimal coupling for information processing
    
    # Frequency nesting relationships
    THETA_GAMMA_RATIO = (30 / 6)  # ~5:1 ratio for optimal nesting
    
    # Ψ_s field modulation parameters
    PSI_S_BASELINE = 1.0
    PSI_S_RANGE = (0.5, 2.0)  # Modulation range
    PSI_S_COHERENCE_THRESHOLD = 0.7  # Threshold for consciousness coupling


# ============================================================================
# SECTION 2: BASE EXPERIMENT CLASS
# ============================================================================

class ExperimentStatus(Enum):
    """Status tracking for experiments"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


@dataclass
class ExperimentConfig:
    """Configuration for a validation experiment"""
    name: str
    description: str
    layer_components: List[str]
    duration: float  # seconds
    dt: float  # time step
    temperature: float = PhysicalConstants.T_body
    psi_s_field: float = OscillationParams.PSI_S_BASELINE
    random_seed: Optional[int] = None
    save_trajectory: bool = True
    validation_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.duration > 0, "Duration must be positive"
        assert self.dt > 0, "Time step must be positive"
        assert self.dt < self.duration, "Time step must be smaller than duration"
        assert PhysicalConstants.T_min <= self.temperature <= PhysicalConstants.T_max, \
            f"Temperature {self.temperature}K outside biological range"
        
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'layer_components': self.layer_components,
            'duration': self.duration,
            'dt': self.dt,
            'temperature': self.temperature,
            'psi_s_field': self.psi_s_field,
            'random_seed': self.random_seed,
            'save_trajectory': self.save_trajectory,
            'validation_checks': self.validation_checks,
            'metadata': self.metadata
        }
    
    def save(self, filepath: Path):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ExperimentResults:
    """Container for experiment results"""
    config: ExperimentConfig
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Time series data
    times: Optional[np.ndarray] = None
    states: Optional[np.ndarray] = None
    
    # Validation metrics
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    computation_time: Optional[float] = None
    memory_usage: Optional[float] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_validation(self, name: str, passed: bool, value: Any, 
                      threshold: Any = None, description: str = ""):
        """Add a validation result"""
        self.validation_results[name] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'description': description
        }
    
    def save(self, filepath: Path):
        """Save results to HDF5 file"""
        with h5py.File(filepath, 'w') as f:
            # Save configuration
            config_group = f.create_group('config')
            for key, value in self.config.to_dict().items():
                if isinstance(value, (list, dict)):
                    config_group.attrs[key] = json.dumps(value)
                else:
                    config_group.attrs[key] = value
            
            # Save status and timing
            f.attrs['status'] = self.status.value
            f.attrs['start_time'] = self.start_time.isoformat()
            if self.end_time:
                f.attrs['end_time'] = self.end_time.isoformat()
            
            # Save time series data
            if self.times is not None:
                f.create_dataset('times', data=self.times, compression='gzip')
            if self.states is not None:
                f.create_dataset('states', data=self.states, compression='gzip')
            
            # Save validation results
            validation_group = f.create_group('validation')
            for key, value in self.validation_results.items():
                val_subgroup = validation_group.create_group(key)
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, list)):
                        val_subgroup.create_dataset(k, data=v)
                    else:
                        val_subgroup.attrs[k] = v
            
            # Save performance metrics
            if self.computation_time:
                f.attrs['computation_time'] = self.computation_time
            if self.memory_usage:
                f.attrs['memory_usage'] = self.memory_usage
            
            # Save errors and warnings
            if self.errors:
                f.attrs['errors'] = json.dumps(self.errors)
            if self.warnings:
                f.attrs['warnings'] = json.dumps(self.warnings)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load results from HDF5 file"""
        with h5py.File(filepath, 'r') as f:
            # Load configuration
            config_dict = {}
            for key, value in f['config'].attrs.items():
                try:
                    config_dict[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    config_dict[key] = value
            config = ExperimentConfig(**config_dict)
            
            # Load status and timing
            status = ExperimentStatus(f.attrs['status'])
            start_time = datetime.fromisoformat(f.attrs['start_time'])
            end_time = datetime.fromisoformat(f.attrs['end_time']) if 'end_time' in f.attrs else None
            
            # Create results object
            results = cls(config=config, status=status, start_time=start_time, end_time=end_time)
            
            # Load time series
            if 'times' in f:
                results.times = f['times'][:]
            if 'states' in f:
                results.states = f['states'][:]
            
            # Load validation results
            if 'validation' in f:
                for key in f['validation'].keys():
                    val_dict = {}
                    for k in f['validation'][key].attrs.keys():
                        val_dict[k] = f['validation'][key].attrs[k]
                    for k in f['validation'][key].keys():
                        val_dict[k] = f['validation'][key][k][:]
                    results.validation_results[key] = val_dict
            
            # Load performance metrics
            if 'computation_time' in f.attrs:
                results.computation_time = f.attrs['computation_time']
            if 'memory_usage' in f.attrs:
                results.memory_usage = f.attrs['memory_usage']
            
            # Load errors and warnings
            if 'errors' in f.attrs:
                results.errors = json.loads(f.attrs['errors'])
            if 'warnings' in f.attrs:
                results.warnings = json.loads(f.attrs['warnings'])
            
            return results


class BaseExperiment:
    """
    Base class for all Layer 2 validation experiments
    
    This class provides the fundamental infrastructure for running
    computational experiments that validate Layer 2 theoretical predictions.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(f'Layer2.{config.name}')
        self.results = None
        
        # Initialize time array
        self.n_steps = int(config.duration / config.dt)
        self.times = np.linspace(0, config.duration, self.n_steps)
        
        self.logger.info(f"Initialized experiment: {config.name}")
        self.logger.info(f"Duration: {config.duration}s, dt: {config.dt}s, steps: {self.n_steps}")
    
    def setup(self):
        """Setup experiment (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement setup()")
    
    def run_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """Execute one time step (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement run_step()")
    
    def validate(self) -> Dict[str, bool]:
        """Validate results against theoretical predictions (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement validate()")
    
    def run(self) -> ExperimentResults:
        """
        Execute the complete experiment
        
        Returns:
            ExperimentResults object containing all data and validation results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting experiment: {self.config.name}")
        
        # Initialize results container
        self.results = ExperimentResults(
            config=self.config,
            status=ExperimentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Setup experiment
            self.logger.info("Setting up experiment...")
            initial_state = self.setup()
            
            # Prepare trajectory storage if requested
            if self.config.save_trajectory:
                trajectory = np.zeros((self.n_steps, len(initial_state)))
                trajectory[0] = initial_state
            
            # Main simulation loop
            self.logger.info("Running simulation...")
            state = initial_state.copy()
            
            for i, t in enumerate(self.times[1:], 1):
                # Execute one time step
                state = self.run_step(t, state)
                
                # Store trajectory
                if self.config.save_trajectory:
                    trajectory[i] = state
                
                # Progress logging
                if i % (self.n_steps // 10) == 0:
                    progress = 100 * i / self.n_steps
                    self.logger.info(f"Progress: {progress:.1f}%")
            
            # Store results
            self.results.times = self.times
            if self.config.save_trajectory:
                self.results.states = trajectory
            
            # Validate results
            self.logger.info("Validating results...")
            validation_results = self.validate()
            
            for check_name, passed in validation_results.items():
                self.results.add_validation(
                    name=check_name,
                    passed=passed,
                    value=None,  # Subclasses should add actual values
                    description=check_name
                )
            
            # Mark as completed
            self.results.status = ExperimentStatus.COMPLETED
            self.results.end_time = datetime.now()
            self.results.computation_time = (self.results.end_time - start_time).total_seconds()
            
            self.logger.info(f"Experiment completed successfully in {self.results.computation_time:.2f}s")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            self.results.status = ExperimentStatus.FAILED
            self.results.errors.append(str(e))
            self.results.end_time = datetime.now()
            return self.results
    
    def save_results(self, output_dir: Path):
        """Save experiment results to file"""
        if self.results is None:
            raise ValueError("No results to save. Run experiment first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.results.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.h5"
        filepath = output_dir / filename
        
        self.results.save(filepath)
        self.logger.info(f"Results saved to: {filepath}")
        
        return filepath


# ============================================================================
# SECTION 3: STATE REPRESENTATION
# ============================================================================

@dataclass
class NeuralState:
    """
    Complete state representation for Layer 2 simulations
    
    This class encapsulates all state variables needed to represent
    the neurochemical-neurological system at any given time point.
    """
    # Electrical state
    V_membrane: float  # Membrane potential (mV)
    
    # Neurotransmitter concentrations (M)
    NT_concentrations: Dict[str, float]
    
    # Calcium dynamics (M)
    Ca_internal: float
    Ca_external: float
    
    # Quantum state representation
    quantum_coherence: float  # 0-1
    quantum_phase: float  # radians
    
    # Ψ_s field coupling
    psi_s_local: float
    
    # Oscillation phases (radians)
    oscillation_phases: Dict[str, float]
    
    # Glial state
    astrocyte_Ca: float
    
    # Time
    t: float
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for integration"""
        arr = [
            self.V_membrane,
            self.Ca_internal,
            self.Ca_external,
            self.quantum_coherence,
            self.quantum_phase,
            self.psi_s_local,
            self.astrocyte_Ca,
        ]
        # Add NT concentrations
        for nt in sorted(self.NT_concentrations.keys()):
            arr.append(self.NT_concentrations[nt])
        # Add oscillation phases
        for band in sorted(self.oscillation_phases.keys()):
            arr.append(self.oscillation_phases[band])
        
        return np.array(arr)
    
    @classmethod
    def from_array(cls, arr: np.ndarray, t: float, nt_keys: List[str], 
                   osc_keys: List[str]) -> 'NeuralState':
        """Reconstruct state from numpy array"""
        idx = 0
        V_membrane = arr[idx]; idx += 1
        Ca_internal = arr[idx]; idx += 1
        Ca_external = arr[idx]; idx += 1
        quantum_coherence = arr[idx]; idx += 1
        quantum_phase = arr[idx]; idx += 1
        psi_s_local = arr[idx]; idx += 1
        astrocyte_Ca = arr[idx]; idx += 1
        
        NT_concentrations = {}
        for nt in nt_keys:
            NT_concentrations[nt] = arr[idx]
            idx += 1
        
        oscillation_phases = {}
        for band in osc_keys:
            oscillation_phases[band] = arr[idx]
            idx += 1
        
        return cls(
            V_membrane=V_membrane,
            NT_concentrations=NT_concentrations,
            Ca_internal=Ca_internal,
            Ca_external=Ca_external,
            quantum_coherence=quantum_coherence,
            quantum_phase=quantum_phase,
            psi_s_local=psi_s_local,
            oscillation_phases=oscillation_phases,
            astrocyte_Ca=astrocyte_Ca,
            t=t
        )
    
    @classmethod
    def initialize_default(cls, temperature: float = PhysicalConstants.T_body) -> 'NeuralState':
        """Create default initial state"""
        NT_concentrations = {
            nt: params['baseline'] 
            for nt, params in NeurotransmitterParams.CONCENTRATIONS.items()
        }
        
        oscillation_phases = {
            band: np.random.uniform(0, 2*np.pi)
            for band in NeurotransmitterParams.OSCILLATION_BANDS.keys()
        }
        
        return cls(
            V_membrane=-70.0,  # Resting potential
            NT_concentrations=NT_concentrations,
            Ca_internal=1e-7,  # 100 nM baseline
            Ca_external=2e-3,  # 2 mM extracellular
            quantum_coherence=0.5,  # Moderate coherence
            quantum_phase=0.0,
            psi_s_local=1.0,  # Baseline field strength
            oscillation_phases=oscillation_phases,
            astrocyte_Ca=1e-7,  # 100 nM baseline
            t=0.0
        )


# ============================================================================
# SECTION 4: VALIDATION UTILITIES
# ============================================================================

class ValidationMetrics:
    """
    Collection of validation metrics for Layer 2 experiments
    
    These methods implement quantitative tests of theoretical predictions
    from the SCPN manuscript.
    """
    
    @staticmethod
    def check_energy_conservation(states: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Verify energy conservation throughout simulation
        
        Theory: Total system energy should be conserved within numerical precision
        Manuscript ref: Layer 2, Conservation Laws
        """
        # Calculate total energy at each time point
        # This is a simplified version - full implementation would include
        # kinetic, potential, and field energy terms
        energies = np.sum(states**2, axis=1)  # Simplified energy metric
        
        energy_variation = np.std(energies) / np.mean(energies)
        return energy_variation < tolerance
    
    @staticmethod
    def check_probability_conservation(quantum_states: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Verify quantum probability conservation
        
        Theory: Tr(ρ) = 1 for all time
        Manuscript ref: Layer 2, Quantum-Classical Bridge
        """
        probabilities = np.sum(np.abs(quantum_states)**2, axis=1)
        return np.all(np.abs(probabilities - 1.0) < tolerance)
    
    @staticmethod
    def check_causality(times: np.ndarray, states: np.ndarray) -> bool:
        """
        Verify causal ordering of events
        
        Theory: No retrocausal influences in Layer 2
        Manuscript ref: Layer 2, Temporal Dynamics
        """
        # Check that correlations decay with time lag
        # (simplified version - full implementation would use transfer entropy)
        for i in range(states.shape[1]):
            signal = states[:, i]
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Autocorrelation should generally decay
            if not np.all(np.diff(autocorr[:len(autocorr)//2]) <= 0):
                # Allow some fluctuations but overall trend should be decreasing
                linear_fit = np.polyfit(range(len(autocorr)//2), autocorr[:len(autocorr)//2], 1)
                if linear_fit[0] > 0:  # Positive slope = violation
                    return False
        
        return True
    
    @staticmethod
    def measure_oscillation_pac(phase_signal: np.ndarray, amplitude_signal: np.ndarray) -> float:
        """
        Measure Phase-Amplitude Coupling (PAC) strength
        
        Theory: PAC(θ,γ) quantifies theta-gamma nesting
        Manuscript ref: Layer 2, Oscillatory Hierarchies
        
        Returns:
            PAC strength (0-1, higher = stronger coupling)
        """
        # Extract phase from low-frequency signal
        analytic_signal = signal.hilbert(phase_signal)
        phase = np.angle(analytic_signal)
        
        # Extract amplitude from high-frequency signal
        analytic_amplitude = signal.hilbert(amplitude_signal)
        amplitude = np.abs(analytic_amplitude)
        
        # Calculate mean amplitude in phase bins
        n_bins = 18  # 20-degree bins
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amplitudes = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
            if np.any(mask):
                mean_amplitudes[i] = np.mean(amplitude[mask])
        
        # Calculate modulation index (Kullback-Leibler divergence from uniform)
        mean_amplitudes = mean_amplitudes / np.sum(mean_amplitudes)  # Normalize
        uniform = np.ones(n_bins) / n_bins
        
        # KL divergence
        pac = np.sum(mean_amplitudes * np.log(mean_amplitudes / uniform + 1e-10))
        pac = pac / np.log(n_bins)  # Normalize to 0-1
        
        return pac
    
    @staticmethod
    def calculate_coherence_time(correlation_function: np.ndarray, dt: float) -> float:
        """
        Calculate quantum coherence time from correlation function
        
        Theory: τ_coherence = integral of |<Ψ(t)|Ψ(0)>|^2 dt
        Manuscript ref: Layer 2, Quantum Coherence Parameters
        """
        # Find where correlation drops to 1/e
        threshold = 1.0 / np.e
        try:
            idx = np.where(correlation_function < threshold)[0][0]
            coherence_time = idx * dt
        except IndexError:
            # Correlation never drops below threshold
            coherence_time = len(correlation_function) * dt
        
        return coherence_time
    
    @staticmethod
    def verify_cooperativity(Ca_conc: np.ndarray, response: np.ndarray, 
                            expected_hill: float = 4.0, tolerance: float = 0.5) -> bool:
        """
        Verify Ca^4 cooperativity in vesicle release
        
        Theory: P_release ∝ [Ca^2+]^4 (Hill coefficient ~4)
        Manuscript ref: Layer 2, Vesicle Release Mechanisms
        """
        # Fit Hill equation: y = x^n / (K^n + x^n)
        def hill_func(x, K, n):
            return x**n / (K**n + x**n)
        
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(hill_func, Ca_conc, response, p0=[1e-6, 4.0])
            fitted_hill = popt[1]
            
            return np.abs(fitted_hill - expected_hill) < tolerance
        except:
            return False


# ============================================================================
# SECTION 5: EXPERIMENT REGISTRY
# ============================================================================

class ExperimentRegistry:
    """
    Central registry for all validation experiments
    
    This class maintains a catalog of all available experiments and
    provides utilities for running and managing them.
    """
    
    def __init__(self):
        self.experiments: Dict[str, type] = {}
        self.logger = logging.getLogger('Layer2.Registry')
    
    def register(self, name: str, experiment_class: type):
        """Register a new experiment class"""
        if not issubclass(experiment_class, BaseExperiment):
            raise ValueError(f"{experiment_class} must inherit from BaseExperiment")
        
        self.experiments[name] = experiment_class
        self.logger.info(f"Registered experiment: {name}")
    
    def create_experiment(self, name: str, config: ExperimentConfig) -> BaseExperiment:
        """Create an instance of a registered experiment"""
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment: {name}")
        
        return self.experiments[name](config)
    
    def list_experiments(self) -> List[str]:
        """List all registered experiments"""
        return list(self.experiments.keys())
    
    def run_experiment(self, name: str, config: ExperimentConfig, 
                      output_dir: Optional[Path] = None) -> ExperimentResults:
        """Create and run an experiment"""
        experiment = self.create_experiment(name, config)
        results = experiment.run()
        
        if output_dir is not None:
            experiment.save_results(output_dir)
        
        return results
    
    def run_batch(self, configs: List[ExperimentConfig], 
                  output_dir: Optional[Path] = None,
                  parallel: bool = False) -> List[ExperimentResults]:
        """Run multiple experiments"""
        results = []
        
        for config in configs:
            self.logger.info(f"Running batch experiment: {config.name}")
            result = self.run_experiment(config.name, config, output_dir)
            results.append(result)
        
        return results


# Initialize global registry
EXPERIMENT_REGISTRY = ExperimentRegistry()


# ============================================================================
# SECTION 6: MODULE INITIALIZATION
# ============================================================================

def initialize_layer2_validation():
    """Initialize the Layer 2 validation framework"""
    logger.info("="*80)
    logger.info("SCPN Layer 2 (Neurochemical-Neurological) Validation Suite")
    logger.info("Core Framework Module Initialized")
    logger.info("="*80)
    logger.info("")
    logger.info("Available components:")
    logger.info("  - Physical constants and parameters")
    logger.info("  - Base experiment infrastructure")
    logger.info("  - State representation classes")
    logger.info("  - Validation metrics")
    logger.info("  - Experiment registry")
    logger.info("")
    logger.info("To use this framework:")
    logger.info("  1. Define experiment configs with ExperimentConfig")
    logger.info("  2. Create experiment classes inheriting from BaseExperiment")
    logger.info("  3. Register experiments with EXPERIMENT_REGISTRY")
    logger.info("  4. Run experiments and analyze results")
    logger.info("")
    logger.info("Next modules to load:")
    logger.info("  - Module 2: Quantum-Classical Transition Validators")
    logger.info("  - Module 3: Neurotransmitter & Oscillation Tests")
    logger.info("  - Module 4: Glial Network & Metabolic Validators")
    logger.info("  - Module 5: Integration & Multi-Scale Tests")
    logger.info("  - Module 6: Analysis Tools & Visualization Suite")
    logger.info("="*80)


# Run initialization when module is imported
initialize_layer2_validation()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Layer 2 Validation Framework - Core Module")
    print("="*80 + "\n")
    
    # Example 1: Create a basic experiment configuration
    print("Example 1: Creating experiment configuration")
    print("-" * 40)
    
    config = ExperimentConfig(
        name="example_basic_dynamics",
        description="Basic neurochemical dynamics test",
        layer_components=["neurotransmitter_dynamics", "oscillations"],
        duration=10.0,  # 10 seconds
        dt=0.001,  # 1 ms time step
        temperature=310.0,  # Body temperature
        random_seed=42
    )
    
    print(f"Config created: {config.name}")
    print(f"  Duration: {config.duration}s")
    print(f"  Time steps: {int(config.duration/config.dt)}")
    print(f"  Components: {', '.join(config.layer_components)}")
    
    # Example 2: Initialize a neural state
    print("\nExample 2: Initializing neural state")
    print("-" * 40)
    
    state = NeuralState.initialize_default()
    print(f"Initial membrane potential: {state.V_membrane} mV")
    print(f"Neurotransmitter concentrations:")
    for nt, conc in state.NT_concentrations.items():
        print(f"  {nt}: {conc:.2e} M")
    
    # Example 3: Convert state to array and back
    print("\nExample 3: State array conversion")
    print("-" * 40)
    
    nt_keys = sorted(state.NT_concentrations.keys())
    osc_keys = sorted(state.oscillation_phases.keys())
    
    state_array = state.to_array()
    print(f"State vector length: {len(state_array)}")
    
    reconstructed = NeuralState.from_array(state_array, 0.0, nt_keys, osc_keys)
    print(f"Reconstruction successful: {np.allclose(state_array, reconstructed.to_array())}")
    
    # Example 4: Validation metrics
    print("\nExample 4: Validation metrics")
    print("-" * 40)
    
    # Generate synthetic data for testing
    n_points = 1000
    test_states = np.random.randn(n_points, 10) * 0.1 + 1.0
    
    energy_conserved = ValidationMetrics.check_energy_conservation(test_states)
    print(f"Energy conservation check: {'PASS' if energy_conserved else 'FAIL'}")
    
    # Test PAC measurement
    t = np.linspace(0, 10, 1000)
    theta_signal = np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
    gamma_signal = np.sin(2 * np.pi * 40 * t) * (1 + 0.5 * np.cos(2 * np.pi * 6 * t))  # Modulated 40 Hz
    
    pac_strength = ValidationMetrics.measure_oscillation_pac(theta_signal, gamma_signal)
    print(f"Theta-Gamma PAC strength: {pac_strength:.3f}")
    
    print("\n" + "="*80)
    print("Core module demonstration complete!")
    print("="*80 + "\n")
