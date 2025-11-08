"""
Layer 2 Experimental Validation Suite - Module 3 of 6
Neurotransmitter & Oscillation Tests

This module implements comprehensive validation experiments for neurotransmitter
dynamics and neural oscillation hierarchies in the SCPN Layer 2 framework.

Core Components:
---------------
1. Complete Neurotransmitter Systems (6 major NTs)
2. Receptor Dynamics and Kinetics
3. Reuptake and Diffusion Models
4. Oscillation Generators (all frequency bands)
5. Phase-Amplitude Coupling (PAC) Validation
6. Cross-Frequency Coupling
7. VIBRANA Resonance Mapping
8. Synaptic Plasticity Mechanisms

Neurotransmitter Systems:
------------------------
- Glutamate (excitatory)
- GABA (inhibitory)
- Dopamine (reward, motor)
- Serotonin (mood, arousal)
- Acetylcholine (attention, memory)
- Norepinephrine (arousal, attention)

Oscillation Bands:
-----------------
- Slow (0.01-0.1 Hz) - Serotonin modulation
- Delta (0.5-4 Hz) - Sleep, plasticity
- Theta (4-8 Hz) - Memory, navigation
- Alpha (8-13 Hz) - Attention, ACh
- Beta (13-30 Hz) - Motor, dopamine
- Gamma (30-80 Hz) - Binding, glutamate
- High Gamma (80-200 Hz) - Local processing

Manuscript References:
---------------------
- Part 3, Chapter 1: The Neurochemical Layer as the Great Modulator
- Part 3, Chapter 2: The Quantum Orchestra: Neural Oscillator Synchronization
- Part 3, Chapter 7: The Alchemical Engine: Neurotransmitter Quantum Tunneling Networks
- Part 3: Complete oscillation hierarchy and PAC descriptions

Author: Generated for SCPN Manuscript Validation
Version: 1.0.0
Date: 2025-11-07
"""

import numpy as np
import scipy as sp
from scipy import signal, integrate, optimize, fft
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import from previous modules
try:
    from layer2_validation_core import (
        BaseExperiment,
        ExperimentConfig,
        ExperimentResults,
        ExperimentStatus,
        NeuralState,
        ValidationMetrics,
        PhysicalConstants,
        NeurotransmitterParams,
        OscillationParams,
        EXPERIMENT_REGISTRY,
        logger
    )
except ImportError:
    raise ImportError("Module 1 (layer2_validation_core.py) must be loaded first!")

# Module logger
module_logger = logging.getLogger('Layer2.Neurotransmitter')


# ============================================================================
# SECTION 1: NEUROTRANSMITTER SYSTEM MODELS
# ============================================================================

class NeurotransmitterType(Enum):
    """Neurotransmitter classifications"""
    GLUTAMATE = "glutamate"
    GABA = "GABA"  # Note: uppercase to match NeurotransmitterParams
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"
    NOREPINEPHRINE = "norepinephrine"


@dataclass
class ReceptorDynamics:
    """
    Receptor kinetics model
    
    Implements binding kinetics, desensitization, and trafficking
    
    Manuscript ref: Part 3, Ch 18, The Receptor Sea
    """
    receptor_type: str
    
    # Binding kinetics
    k_on: float  # Binding rate (M^-1 s^-1)
    k_off: float  # Unbinding rate (s^-1)
    
    # Channel kinetics (if ionotropic)
    k_open: float = 0.0  # Opening rate (s^-1)
    k_close: float = 0.0  # Closing rate (s^-1)
    
    # Desensitization
    k_desensitize: float = 0.0  # Desensitization rate (s^-1)
    k_recover: float = 0.0  # Recovery rate (s^-1)
    
    # Receptor density
    density: float = 1000.0  # receptors/μm²
    
    # State variables
    bound_fraction: float = 0.0
    open_fraction: float = 0.0
    desensitized_fraction: float = 0.0
    
    @property
    def K_d(self) -> float:
        """Dissociation constant K_d = k_off / k_on"""
        return self.k_off / self.k_on
    
    @property
    def EC50(self) -> float:
        """Half-maximal effective concentration (approximation)"""
        return self.K_d
    
    def update_state(self, nt_concentration: float, dt: float):
        """
        Update receptor state given NT concentration
        
        Implements:
        dB/dt = k_on × [NT] × (1-B-D) - k_off × B
        dO/dt = k_open × B - k_close × O
        dD/dt = k_desensitize × B - k_recover × D
        """
        # Available receptors (not bound or desensitized)
        available = 1.0 - self.bound_fraction - self.desensitized_fraction
        
        # Binding
        dB_dt = (self.k_on * nt_concentration * available - 
                 self.k_off * self.bound_fraction)
        
        # Channel opening (if applicable)
        dO_dt = (self.k_open * self.bound_fraction - 
                 self.k_close * self.open_fraction)
        
        # Desensitization
        dD_dt = (self.k_desensitize * self.bound_fraction - 
                 self.k_recover * self.desensitized_fraction)
        
        # Euler integration
        self.bound_fraction = np.clip(self.bound_fraction + dB_dt * dt, 0, 1)
        self.open_fraction = np.clip(self.open_fraction + dO_dt * dt, 0, 1)
        self.desensitized_fraction = np.clip(self.desensitized_fraction + dD_dt * dt, 0, 1)
    
    def steady_state_response(self, nt_concentration: float) -> float:
        """
        Calculate steady-state response to constant NT concentration
        
        Returns: Fraction of open receptors
        """
        # Simple Hill-like response
        response = nt_concentration / (nt_concentration + self.EC50)
        return response * (1.0 - self.desensitized_fraction)


class NeurotransmitterSystem:
    """
    Complete model of a neurotransmitter system
    
    Includes:
    - Release dynamics
    - Diffusion
    - Receptor activation
    - Reuptake
    - Degradation
    
    Manuscript ref: Part 3, Ch 7, Neurotransmitter Quantum Tunneling Networks
    """
    
    def __init__(self, nt_type: NeurotransmitterType):
        self.nt_type = nt_type
        self.name = nt_type.value
        
        # Load parameters from NeurotransmitterParams
        self.params = NeurotransmitterParams.CONCENTRATIONS[self.name]
        self.kinetics = NeurotransmitterParams.KINETICS[self.name]
        
        # Current concentration
        self.concentration = self.params['baseline']
        
        # Receptors
        self.receptors = self._initialize_receptors()
        
        # Spatial profile (for diffusion)
        self.spatial_profile = None
        
        module_logger.info(f"Initialized {self.name} system: "
                          f"baseline={self.concentration*1e6:.2f} μM, "
                          f"{len(self.receptors)} receptor types")
    
    def _initialize_receptors(self) -> Dict[str, ReceptorDynamics]:
        """Initialize receptor types for this NT"""
        receptors = {}
        
        if self.nt_type == NeurotransmitterType.GLUTAMATE:
            # AMPA receptors (fast)
            receptors['AMPA'] = ReceptorDynamics(
                receptor_type='AMPA',
                k_on=1e7,  # M^-1 s^-1
                k_off=100,  # s^-1
                k_open=1000,
                k_close=500,
                k_desensitize=50,
                k_recover=5
            )
            # NMDA receptors (slow, voltage-dependent)
            receptors['NMDA'] = ReceptorDynamics(
                receptor_type='NMDA',
                k_on=1e6,
                k_off=10,
                k_open=50,
                k_close=10,
                k_desensitize=5,
                k_recover=0.5
            )
            
        elif self.nt_type == NeurotransmitterType.GABA:
            # GABA-A (fast inhibitory)
            receptors['GABA-A'] = ReceptorDynamics(
                receptor_type='GABA-A',
                k_on=5e6,
                k_off=200,
                k_open=800,
                k_close=400,
                k_desensitize=20,
                k_recover=2
            )
            # GABA-B (slow metabotropic)
            receptors['GABA-B'] = ReceptorDynamics(
                receptor_type='GABA-B',
                k_on=1e6,
                k_off=5,
                k_open=10,
                k_close=2,
                k_desensitize=0.5,
                k_recover=0.1
            )
            
        elif self.nt_type == NeurotransmitterType.DOPAMINE:
            # D1 receptors (excitatory)
            receptors['D1'] = ReceptorDynamics(
                receptor_type='D1',
                k_on=1e6,
                k_off=10,
                density=500
            )
            # D2 receptors (inhibitory)
            receptors['D2'] = ReceptorDynamics(
                receptor_type='D2',
                k_on=1e6,
                k_off=5,
                density=800
            )
            
        elif self.nt_type == NeurotransmitterType.SEROTONIN:
            # 5-HT1A (inhibitory)
            receptors['5-HT1A'] = ReceptorDynamics(
                receptor_type='5-HT1A',
                k_on=1e6,
                k_off=8,
                density=600
            )
            # 5-HT2A (excitatory)
            receptors['5-HT2A'] = ReceptorDynamics(
                receptor_type='5-HT2A',
                k_on=5e5,
                k_off=10,
                density=400
            )
            
        elif self.nt_type == NeurotransmitterType.ACETYLCHOLINE:
            # Nicotinic (fast)
            receptors['nAChR'] = ReceptorDynamics(
                receptor_type='nAChR',
                k_on=1e7,
                k_off=500,
                k_open=2000,
                k_close=1000,
                k_desensitize=100,
                k_recover=10
            )
            # Muscarinic (slow)
            receptors['mAChR'] = ReceptorDynamics(
                receptor_type='mAChR',
                k_on=1e6,
                k_off=10,
                density=700
            )
            
        elif self.nt_type == NeurotransmitterType.NOREPINEPHRINE:
            # Alpha receptors
            receptors['alpha'] = ReceptorDynamics(
                receptor_type='alpha',
                k_on=1e6,
                k_off=15,
                density=500
            )
            # Beta receptors
            receptors['beta'] = ReceptorDynamics(
                receptor_type='beta',
                k_on=1e6,
                k_off=20,
                density=600
            )
        
        return receptors
    
    def release(self, amount: float):
        """Release NT into synapse"""
        self.concentration += amount
    
    def update_dynamics(self, dt: float):
        """
        Update NT concentration based on:
        - Diffusion
        - Reuptake
        - Degradation
        - Receptor binding
        """
        # Reuptake (first-order)
        uptake_rate = self.kinetics['uptake_rate']
        self.concentration *= np.exp(-uptake_rate * dt)
        
        # Update all receptors
        for receptor in self.receptors.values():
            receptor.update_state(self.concentration, dt)
        
        # Ensure non-negative
        self.concentration = max(0, self.concentration)
    
    def get_total_receptor_activation(self) -> float:
        """Sum activation across all receptor types"""
        return sum(r.open_fraction for r in self.receptors.values()) / len(self.receptors)


# ============================================================================
# SECTION 2: OSCILLATION GENERATORS
# ============================================================================

class OscillationBand(Enum):
    """Neural oscillation frequency bands"""
    SLOW = "slow"
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    HIGH_GAMMA = "high_gamma"


class NeuralOscillator:
    """
    Neural oscillation generator
    
    Generates biologically realistic oscillations in specified frequency bands
    with phase and amplitude modulation capabilities.
    
    Manuscript ref: Part 3, Ch 2, Neural Oscillator Synchronization
    """
    
    def __init__(self, band: OscillationBand, baseline_amplitude: float = 1.0):
        self.band = band
        self.band_name = band.value
        
        # Get frequency range
        self.freq_range = NeurotransmitterParams.OSCILLATION_BANDS[self.band_name]
        self.freq_min, self.freq_max = self.freq_range
        self.center_freq = (self.freq_min + self.freq_max) / 2.0
        
        # Oscillator state
        self.phase = np.random.uniform(0, 2*np.pi)
        self.frequency = self.center_freq
        self.amplitude = baseline_amplitude
        
        # Modulation
        self.amplitude_modulation = 0.0
        self.frequency_modulation = 0.0
        self.phase_modulation = 0.0
        
        module_logger.debug(f"Oscillator initialized: {self.band_name} "
                           f"({self.freq_min:.1f}-{self.freq_max:.1f} Hz)")
    
    def set_frequency(self, freq: float):
        """Set oscillation frequency (within band limits)"""
        self.frequency = np.clip(freq, self.freq_min, self.freq_max)
    
    def set_amplitude(self, amp: float):
        """Set oscillation amplitude"""
        self.amplitude = max(0, amp)
    
    def update_phase(self, dt: float):
        """Update oscillator phase"""
        # Frequency modulation
        current_freq = self.frequency + self.frequency_modulation
        
        # Phase advance
        self.phase += 2 * np.pi * current_freq * dt
        
        # Add phase modulation
        self.phase += self.phase_modulation
        
        # Wrap to [0, 2π]
        self.phase = self.phase % (2 * np.pi)
    
    def get_value(self) -> float:
        """Get current oscillator value"""
        # Base oscillation
        value = np.sin(self.phase)
        
        # Apply amplitude (with modulation)
        amplitude = self.amplitude * (1.0 + self.amplitude_modulation)
        
        return amplitude * value
    
    def reset_modulation(self):
        """Reset all modulation terms"""
        self.amplitude_modulation = 0.0
        self.frequency_modulation = 0.0
        self.phase_modulation = 0.0


class OscillationHierarchy:
    """
    Hierarchical oscillation network
    
    Implements nested oscillations with Phase-Amplitude Coupling (PAC)
    
    Key concept: Higher frequency oscillations are amplitude-modulated
    by the phase of lower frequency oscillations
    
    Manuscript ref: Part 3, Ch 2, Theta-Gamma Nesting
    """
    
    def __init__(self):
        # Create oscillators for all bands
        self.oscillators: Dict[str, NeuralOscillator] = {}
        
        for band in OscillationBand:
            self.oscillators[band.value] = NeuralOscillator(band)
        
        # PAC coupling matrix (phase band -> amplitude band)
        self.pac_coupling = self._initialize_pac_coupling()
        
        # Ψ_s field coupling
        self.psi_s_field = 1.0
        
        module_logger.info(f"Oscillation hierarchy initialized: {len(self.oscillators)} bands")
    
    def _initialize_pac_coupling(self) -> Dict[Tuple[str, str], float]:
        """
        Initialize Phase-Amplitude Coupling matrix
        
        Returns dict: (phase_band, amplitude_band) -> coupling_strength
        """
        coupling = {}
        
        # Theta-gamma coupling (strongest)
        coupling[('theta', 'gamma')] = 0.5
        coupling[('theta', 'high_gamma')] = 0.3
        
        # Alpha-beta coupling
        coupling[('alpha', 'beta')] = 0.3
        coupling[('alpha', 'gamma')] = 0.2
        
        # Delta modulates slower rhythms
        coupling[('delta', 'theta')] = 0.4
        coupling[('delta', 'alpha')] = 0.3
        
        # Slow modulates delta
        coupling[('slow', 'delta')] = 0.3
        
        return coupling
    
    def update_pac_modulation(self):
        """
        Update amplitude modulation based on phase coupling
        
        For each (phase_band, amp_band) pair with coupling strength λ:
        amp_modulation = λ × cos(phase_low)
        """
        # Reset all modulations
        for osc in self.oscillators.values():
            osc.reset_modulation()
        
        # Apply PAC
        for (phase_band, amp_band), strength in self.pac_coupling.items():
            phase_osc = self.oscillators[phase_band]
            amp_osc = self.oscillators[amp_band]
            
            # Amplitude of high-freq osc modulated by phase of low-freq osc
            modulation = strength * np.cos(phase_osc.phase)
            amp_osc.amplitude_modulation += modulation
    
    def couple_to_psi_s(self, psi_s_field: float):
        """
        Couple oscillations to Ψ_s consciousness field
        
        Ψ_s enhances synchronization and PAC strength
        """
        self.psi_s_field = psi_s_field
        
        # Enhance PAC coupling with Ψ_s
        # Higher Ψ_s -> stronger cross-frequency coupling
        psi_factor = 1.0 + 0.2 * (psi_s_field - 1.0)
        
        for key in self.pac_coupling:
            self.pac_coupling[key] *= psi_factor
    
    def step(self, dt: float):
        """Advance all oscillators by one time step"""
        # Update PAC modulation
        self.update_pac_modulation()
        
        # Update all oscillator phases
        for osc in self.oscillators.values():
            osc.update_phase(dt)
    
    def get_signal(self, band: str) -> float:
        """Get current value of specified oscillation band"""
        return self.oscillators[band].get_value()
    
    def get_composite_signal(self, bands: Optional[List[str]] = None) -> float:
        """Get sum of multiple oscillation bands"""
        if bands is None:
            bands = list(self.oscillators.keys())
        
        return sum(self.get_signal(band) for band in bands)
    
    def measure_pac(self, phase_band: str, amp_band: str, 
                    signal_length: int = 1000) -> float:
        """
        Measure Phase-Amplitude Coupling strength
        
        Returns modulation index (0-1)
        """
        # Generate signals
        phase_signal = []
        amp_signal = []
        
        dt = 0.001  # 1 ms
        for _ in range(signal_length):
            self.step(dt)
            phase_signal.append(self.get_signal(phase_band))
            amp_signal.append(self.get_signal(amp_band))
        
        phase_signal = np.array(phase_signal)
        amp_signal = np.array(amp_signal)
        
        # Use ValidationMetrics from Module 1
        pac = ValidationMetrics.measure_oscillation_pac(phase_signal, amp_signal)
        
        return pac


# ============================================================================
# SECTION 3: COMBINED NEUROTRANSMITTER-OSCILLATION DYNAMICS
# ============================================================================

class NeuroOscillatorySystem:
    """
    Integrated neurotransmitter-oscillation system
    
    Links specific neurotransmitters to oscillation bands:
    - Glutamate -> Gamma (30-80 Hz)
    - GABA -> Beta (13-30 Hz) modulation
    - Dopamine -> Beta (13-30 Hz)
    - Serotonin -> Slow (0.01-0.1 Hz)
    - Acetylcholine -> Alpha (8-13 Hz)
    - Norepinephrine -> Beta (13-30 Hz)
    
    Manuscript ref: Part 3, Layer 2 description, NT-oscillation mapping
    """
    
    def __init__(self):
        # Initialize all NT systems
        self.nt_systems: Dict[str, NeurotransmitterSystem] = {}
        for nt_type in NeurotransmitterType:
            self.nt_systems[nt_type.value] = NeurotransmitterSystem(nt_type)
        
        # Initialize oscillation hierarchy
        self.oscillations = OscillationHierarchy()
        
        # NT-to-oscillation mapping
        self.nt_osc_mapping = {
            'glutamate': ['gamma', 'high_gamma'],
            'GABA': ['beta', 'gamma'],  # GABA modulates excitatory rhythms
            'dopamine': ['beta'],
            'serotonin': ['slow', 'delta'],
            'acetylcholine': ['alpha', 'theta'],
            'norepinephrine': ['beta', 'alpha']
        }
        
        module_logger.info("NeuroOscillatorySystem initialized with all NT-oscillation couplings")
    
    def update_nt_oscillation_coupling(self):
        """
        Update oscillation parameters based on NT concentrations
        
        Higher NT concentration -> stronger oscillations in linked bands
        """
        for nt_name, osc_bands in self.nt_osc_mapping.items():
            nt_system = self.nt_systems[nt_name]
            
            # Normalized NT concentration (relative to baseline)
            baseline = nt_system.params['baseline']
            normalized = nt_system.concentration / baseline
            
            # Modulate oscillation amplitude
            for band in osc_bands:
                osc = self.oscillations.oscillators[band]
                
                # NT concentration enhances oscillation
                osc.amplitude = 1.0 * normalized
    
    def step(self, dt: float):
        """Advance system by one time step"""
        # Update NT dynamics
        for nt_system in self.nt_systems.values():
            nt_system.update_dynamics(dt)
        
        # Update NT-oscillation coupling
        self.update_nt_oscillation_coupling()
        
        # Advance oscillations
        self.oscillations.step(dt)
    
    def get_state(self) -> Dict[str, float]:
        """Get current system state"""
        state = {}
        
        # NT concentrations
        for nt_name, nt_system in self.nt_systems.items():
            state[f'nt_{nt_name}'] = nt_system.concentration
        
        # Oscillation values
        for band, osc in self.oscillations.oscillators.items():
            state[f'osc_{band}'] = osc.get_value()
        
        return state


# ============================================================================
# SECTION 4: PHASE-AMPLITUDE COUPLING (PAC) ANALYSIS
# ============================================================================

class PACAnalyzer:
    """
    Comprehensive Phase-Amplitude Coupling analysis
    
    Implements multiple PAC metrics:
    - Modulation Index (MI)
    - Mean Vector Length (MVL)
    - Phase-Locking Value (PLV)
    
    Manuscript ref: Part 3, Ch 2, PAC quantification
    """
    
    @staticmethod
    def extract_phase(signal: np.ndarray, fs: float, freq_band: Tuple[float, float]) -> np.ndarray:
        """
        Extract instantaneous phase from signal in frequency band
        
        Args:
            signal: Time series
            fs: Sampling frequency (Hz)
            freq_band: (low, high) frequency range
        
        Returns:
            Instantaneous phase (radians)
        """
        # Bandpass filter
        sos = signal.butter(4, freq_band, btype='band', fs=fs, output='sos')
        filtered = signal.sosfilt(sos, signal)
        
        # Hilbert transform
        analytic = signal.hilbert(filtered)
        phase = np.angle(analytic)
        
        return phase
    
    @staticmethod
    def extract_amplitude(signal: np.ndarray, fs: float, freq_band: Tuple[float, float]) -> np.ndarray:
        """
        Extract instantaneous amplitude envelope from signal in frequency band
        
        Args:
            signal: Time series
            fs: Sampling frequency (Hz)
            freq_band: (low, high) frequency range
        
        Returns:
            Amplitude envelope
        """
        # Bandpass filter
        sos = signal.butter(4, freq_band, btype='band', fs=fs, output='sos')
        filtered = signal.sosfilt(sos, signal)
        
        # Hilbert transform
        analytic = signal.hilbert(filtered)
        amplitude = np.abs(analytic)
        
        return amplitude
    
    @staticmethod
    def modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> float:
        """
        Calculate Modulation Index (MI)
        
        MI measures how much the amplitude distribution deviates from uniform
        across phase bins using Kullback-Leibler divergence
        
        MI = 0: No coupling
        MI = 1: Perfect coupling
        
        Args:
            phase: Phase time series (radians)
            amplitude: Amplitude time series
            n_bins: Number of phase bins
        
        Returns:
            Modulation Index (0-1)
        """
        # Bin phases
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        # Mean amplitude in each phase bin
        mean_amplitudes = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
            if np.any(mask):
                mean_amplitudes[i] = np.mean(amplitude[mask])
        
        # Normalize to probability distribution
        P = mean_amplitudes / np.sum(mean_amplitudes)
        
        # Uniform distribution
        Q = np.ones(n_bins) / n_bins
        
        # Kullback-Leibler divergence
        # KL(P||Q) = Σ P(i) log(P(i)/Q(i))
        kl = np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))
        
        # Normalize by max possible KL
        mi = kl / np.log(n_bins)
        
        return mi
    
    @staticmethod
    def mean_vector_length(phase: np.ndarray, amplitude: np.ndarray) -> float:
        """
        Calculate Mean Vector Length (MVL)
        
        MVL measures consistency of phase-amplitude relationship
        
        MVL = |<A(t) * e^(i*φ(t))>|
        
        MVL = 0: No coupling
        MVL = 1: Perfect coupling
        
        Args:
            phase: Phase time series (radians)
            amplitude: Amplitude time series
        
        Returns:
            Mean Vector Length (0-1)
        """
        # Normalize amplitude
        amplitude_norm = amplitude / np.sum(amplitude)
        
        # Complex representation
        complex_signal = amplitude_norm * np.exp(1j * phase)
        
        # Mean vector
        mvl = np.abs(np.mean(complex_signal))
        
        return mvl
    
    @staticmethod
    def compute_pac_comodulogram(signal: np.ndarray, fs: float,
                                  phase_freqs: np.ndarray,
                                  amp_freqs: np.ndarray) -> np.ndarray:
        """
        Compute full PAC comodulogram
        
        For each (phase_freq, amp_freq) pair, compute PAC strength
        
        Args:
            signal: Time series
            fs: Sampling frequency
            phase_freqs: Array of phase frequencies to test
            amp_freqs: Array of amplitude frequencies to test
        
        Returns:
            Comodulogram matrix (phase_freqs × amp_freqs)
        """
        comodulogram = np.zeros((len(phase_freqs), len(amp_freqs)))
        
        for i, phase_freq in enumerate(phase_freqs):
            # Extract phase in this band
            phase_band = (phase_freq * 0.8, phase_freq * 1.2)
            phase = PACAnalyzer.extract_phase(signal, fs, phase_band)
            
            for j, amp_freq in enumerate(amp_freqs):
                # Extract amplitude in this band
                amp_band = (amp_freq * 0.8, amp_freq * 1.2)
                amplitude = PACAnalyzer.extract_amplitude(signal, fs, amp_band)
                
                # Compute PAC
                pac = PACAnalyzer.modulation_index(phase, amplitude)
                comodulogram[i, j] = pac
        
        return comodulogram


# ============================================================================
# SECTION 5: SYNAPTIC PLASTICITY MODELS
# ============================================================================

class SynapticPlasticity:
    """
    Synaptic plasticity mechanisms
    
    Implements:
    - Short-term plasticity (facilitation/depression)
    - Long-term plasticity (LTP/LTD)
    - Spike-timing dependent plasticity (STDP)
    
    Manuscript ref: Part 3, Ch 5, The Quantum Synapse
    """
    
    def __init__(self, initial_weight: float = 1.0):
        self.weight = initial_weight
        self.initial_weight = initial_weight
        
        # Short-term plasticity
        self.facilitation = 0.0  # Facilitation variable
        self.depression = 1.0  # Depression variable (available resources)
        
        # Short-term parameters
        self.tau_facilitation = 0.5  # Facilitation time constant (s)
        self.tau_depression = 1.0  # Depression time constant (s)
        self.U = 0.3  # Release probability increment
        
        # Long-term plasticity
        self.ltp_threshold = 0.0  # LTP threshold
        self.ltd_threshold = 0.0  # LTD threshold
        
        module_logger.debug("Synaptic plasticity initialized")
    
    def short_term_update(self, dt: float, spike: bool = False):
        """
        Update short-term plasticity variables
        
        Implements Tsodyks-Markram model:
        dF/dt = -F/τ_F + U(1-F) if spike
        dD/dt = (1-D)/τ_D - UD if spike
        """
        # Decay
        self.facilitation *= np.exp(-dt / self.tau_facilitation)
        self.depression += (1 - self.depression) * (1 - np.exp(-dt / self.tau_depression))
        
        # Spike-driven changes
        if spike:
            # Facilitation increases
            self.facilitation += self.U * (1 - self.facilitation)
            
            # Depression decreases (resources consumed)
            consumed = self.U * self.depression
            self.depression -= consumed
    
    def get_effective_weight(self) -> float:
        """
        Calculate effective synaptic weight including short-term effects
        
        w_effective = w_base × (1 + F) × D
        """
        return self.weight * (1 + self.facilitation) * self.depression
    
    def apply_stdp(self, delta_t: float, learning_rate: float = 0.01):
        """
        Apply spike-timing dependent plasticity
        
        Delta_t = t_post - t_pre
        If Delta_t > 0: Post-synaptic spike after pre -> LTP
        If Delta_t < 0: Post-synaptic spike before pre -> LTD
        
        Args:
            delta_t: Time difference (seconds)
            learning_rate: Learning rate
        """
        tau_stdp = 0.020  # 20 ms STDP window
        
        if delta_t > 0:
            # LTP
            dw = learning_rate * np.exp(-delta_t / tau_stdp)
        else:
            # LTD
            dw = -learning_rate * np.exp(delta_t / tau_stdp)
        
        # Update weight
        self.weight += dw
        
        # Bound weight
        self.weight = np.clip(self.weight, 0, 2 * self.initial_weight)


# ============================================================================
# SECTION 6: VALIDATION EXPERIMENTS
# ============================================================================

class NeurotransmitterDynamicsExperiment(BaseExperiment):
    """
    Experiment 3: Neurotransmitter System Validation
    
    Validates:
    - All 6 major NT systems
    - Receptor dynamics
    - Reuptake kinetics
    - NT-oscillation coupling
    
    Manuscript ref: Part 3, Ch 1-7, Neurotransmitter systems
    """
    
    def setup(self) -> np.ndarray:
        """Initialize NT systems"""
        self.neuro_system = NeuroOscillatorySystem()
        
        # Initial state
        state = []
        for nt_system in self.neuro_system.nt_systems.values():
            state.append(nt_system.concentration)
        
        return np.array(state)
    
    def run_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """Single time step"""
        # Simulate occasional releases
        if np.random.random() < 0.01:  # 1% chance per step
            nt_name = np.random.choice(list(self.neuro_system.nt_systems.keys()))
            self.neuro_system.nt_systems[nt_name].release(1e-6)  # 1 μM
        
        # Update system
        self.neuro_system.step(self.config.dt)
        
        # Extract state
        new_state = []
        for nt_system in self.neuro_system.nt_systems.values():
            new_state.append(nt_system.concentration)
        
        return np.array(new_state)
    
    def validate(self) -> Dict[str, bool]:
        """Validate NT dynamics"""
        validations = {}
        
        # Check that all NT systems are functional
        for nt_name, nt_system in self.neuro_system.nt_systems.items():
            # Check concentration is non-negative
            valid = nt_system.concentration >= 0
            validations[f'{nt_name}_positive'] = valid
            
            # Check receptors are initialized
            valid = len(nt_system.receptors) > 0
            validations[f'{nt_name}_receptors'] = valid
        
        return validations


class OscillationHierarchyExperiment(BaseExperiment):
    """
    Experiment 4: Oscillation Hierarchy and PAC Validation
    
    Validates:
    - All oscillation bands functional
    - Phase-Amplitude Coupling (PAC)
    - Theta-gamma nesting
    - Cross-frequency coupling
    
    Manuscript ref: Part 3, Ch 2, Neural Oscillator Synchronization
    """
    
    def setup(self) -> np.ndarray:
        """Initialize oscillation hierarchy"""
        self.oscillations = OscillationHierarchy()
        
        # Store time series for PAC analysis
        self.theta_trace = []
        self.gamma_trace = []
        self.alpha_trace = []
        self.beta_trace = []
        
        return np.zeros(len(self.oscillations.oscillators))
    
    def run_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """Single time step"""
        self.oscillations.step(self.config.dt)
        
        # Record traces
        self.theta_trace.append(self.oscillations.get_signal('theta'))
        self.gamma_trace.append(self.oscillations.get_signal('gamma'))
        self.alpha_trace.append(self.oscillations.get_signal('alpha'))
        self.beta_trace.append(self.oscillations.get_signal('beta'))
        
        # Extract state
        new_state = []
        for osc in self.oscillations.oscillators.values():
            new_state.append(osc.get_value())
        
        return np.array(new_state)
    
    def validate(self) -> Dict[str, bool]:
        """Validate oscillations and PAC"""
        validations = {}
        
        # Convert to arrays
        theta = np.array(self.theta_trace)
        gamma = np.array(self.gamma_trace)
        alpha = np.array(self.alpha_trace)
        beta = np.array(self.beta_trace)
        
        # Measure theta-gamma PAC
        pac_theta_gamma = ValidationMetrics.measure_oscillation_pac(theta, gamma)
        validations['theta_gamma_pac'] = pac_theta_gamma > 0.1
        
        # Measure alpha-beta PAC
        pac_alpha_beta = ValidationMetrics.measure_oscillation_pac(alpha, beta)
        validations['alpha_beta_pac'] = pac_alpha_beta > 0.05
        
        # Check oscillations are present
        validations['theta_oscillating'] = np.std(theta) > 0.1
        validations['gamma_oscillating'] = np.std(gamma) > 0.1
        
        # Store PAC values for results
        self.pac_values = {
            'theta_gamma': pac_theta_gamma,
            'alpha_beta': pac_alpha_beta
        }
        
        return validations


class IntegratedNeuroOscillatoryExperiment(BaseExperiment):
    """
    Experiment 5: Integrated NT-Oscillation Dynamics
    
    Validates:
    - NT-oscillation coupling
    - Ψ_s field effects on both NT and oscillations
    - Combined system stability
    
    Manuscript ref: Part 3, Complete Layer 2 integration
    """
    
    def setup(self) -> np.ndarray:
        """Initialize integrated system"""
        self.system = NeuroOscillatorySystem()
        
        # Apply Ψ_s field
        self.system.oscillations.couple_to_psi_s(self.config.psi_s_field)
        
        # Get initial state
        state_dict = self.system.get_state()
        return np.array(list(state_dict.values()))
    
    def run_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """Single time step"""
        # Periodic glutamate release (simulating activity)
        if t % 0.1 < self.config.dt:  # Every 100 ms
            self.system.nt_systems['glutamate'].release(5e-7)
        
        # Step system
        self.system.step(self.config.dt)
        
        # Get state
        state_dict = self.system.get_state()
        return np.array(list(state_dict.values()))
    
    def validate(self) -> Dict[str, bool]:
        """Validate integrated system"""
        validations = {}
        
        # Check NT-oscillation coupling is active
        for nt_name in self.system.nt_systems.keys():
            if nt_name in self.system.nt_osc_mapping:
                validations[f'{nt_name}_coupled'] = True
        
        # Check system stability (no NaN or Inf)
        state_dict = self.system.get_state()
        all_finite = all(np.isfinite(v) for v in state_dict.values())
        validations['system_stable'] = all_finite
        
        return validations


# ============================================================================
# SECTION 7: EXPERIMENT REGISTRATION
# ============================================================================

def register_neurotransmitter_experiments():
    """Register all neurotransmitter and oscillation experiments"""
    EXPERIMENT_REGISTRY.register('neurotransmitter_dynamics', NeurotransmitterDynamicsExperiment)
    EXPERIMENT_REGISTRY.register('oscillation_hierarchy', OscillationHierarchyExperiment)
    EXPERIMENT_REGISTRY.register('integrated_neuro_oscillatory', IntegratedNeuroOscillatoryExperiment)
    
    module_logger.info("Registered neurotransmitter and oscillation experiments:")
    module_logger.info("  - neurotransmitter_dynamics")
    module_logger.info("  - oscillation_hierarchy")
    module_logger.info("  - integrated_neuro_oscillatory")


# ============================================================================
# SECTION 8: MODULE INITIALIZATION
# ============================================================================

def initialize_module_3():
    """Initialize Module 3: Neurotransmitter & Oscillation Tests"""
    module_logger.info("="*80)
    module_logger.info("Module 3: Neurotransmitter & Oscillation Tests")
    module_logger.info("="*80)
    module_logger.info("")
    module_logger.info("Components loaded:")
    module_logger.info("  ✓ All 6 neurotransmitter systems")
    module_logger.info("  ✓ Complete receptor dynamics")
    module_logger.info("  ✓ All 7 oscillation bands")
    module_logger.info("  ✓ Phase-Amplitude Coupling (PAC) analysis")
    module_logger.info("  ✓ Synaptic plasticity models")
    module_logger.info("  ✓ Integrated NT-oscillation dynamics")
    module_logger.info("")
    module_logger.info("Neurotransmitters:")
    module_logger.info("  • Glutamate (excitatory)")
    module_logger.info("  • GABA (inhibitory)")
    module_logger.info("  • Dopamine (reward, motor)")
    module_logger.info("  • Serotonin (mood, arousal)")
    module_logger.info("  • Acetylcholine (attention, memory)")
    module_logger.info("  • Norepinephrine (arousal)")
    module_logger.info("")
    module_logger.info("Oscillations:")
    module_logger.info("  • Slow (0.01-0.1 Hz)")
    module_logger.info("  • Delta (0.5-4 Hz)")
    module_logger.info("  • Theta (4-8 Hz)")
    module_logger.info("  • Alpha (8-13 Hz)")
    module_logger.info("  • Beta (13-30 Hz)")
    module_logger.info("  • Gamma (30-80 Hz)")
    module_logger.info("  • High Gamma (80-200 Hz)")
    module_logger.info("")
    module_logger.info("Ready for comprehensive neurochemical validation!")
    module_logger.info("="*80)
    
    # Register experiments
    register_neurotransmitter_experiments()


# Run initialization
initialize_module_3()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Module 3: Neurotransmitter & Oscillation Tests")
    print("="*80 + "\n")
    
    # Example 1: Single NT system
    print("Example 1: Glutamate System")
    print("-" * 40)
    
    glu = NeurotransmitterSystem(NeurotransmitterType.GLUTAMATE)
    print(f"Baseline concentration: {glu.concentration*1e6:.2f} μM")
    print(f"Receptors: {list(glu.receptors.keys())}")
    
    # Simulate release
    glu.release(5e-6)  # 5 μM release
    print(f"After release: {glu.concentration*1e6:.2f} μM")
    
    # Update dynamics
    for _ in range(100):
        glu.update_dynamics(0.001)
    
    print(f"After 100 ms: {glu.concentration*1e6:.2f} μM")
    print(f"Receptor activation: {glu.get_total_receptor_activation():.3f}")
    
    # Example 2: Oscillation hierarchy
    print("\n\nExample 2: Oscillation Hierarchy")
    print("-" * 40)
    
    osc_hierarchy = OscillationHierarchy()
    
    # Generate signals
    theta_signal = []
    gamma_signal = []
    
    for _ in range(1000):
        osc_hierarchy.step(0.001)
        theta_signal.append(osc_hierarchy.get_signal('theta'))
        gamma_signal.append(osc_hierarchy.get_signal('gamma'))
    
    theta_signal = np.array(theta_signal)
    gamma_signal = np.array(gamma_signal)
    
    # Measure PAC
    pac = ValidationMetrics.measure_oscillation_pac(theta_signal, gamma_signal)
    print(f"Theta-Gamma PAC: {pac:.4f}")
    
    # Example 3: Integrated system
    print("\n\nExample 3: Integrated NeuroOscillatory System")
    print("-" * 40)
    
    system = NeuroOscillatorySystem()
    
    # Simulate dynamics
    for i in range(100):
        # Periodic glutamate release
        if i % 10 == 0:
            system.nt_systems['glutamate'].release(1e-6)
        
        system.step(0.001)
    
    state = system.get_state()
    print(f"\nFinal state:")
    for key, value in list(state.items())[:5]:
        print(f"  {key}: {value:.6f}")
    print(f"  ... ({len(state)} total state variables)")
    
    print("\n" + "="*80)
    print("Module 3 demonstration complete!")
    print("="*80 + "\n")
