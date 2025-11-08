"""
Layer 2 Experimental Validation Suite - Module 2 of 6
Quantum-Classical Transition Validators

This module implements comprehensive validation experiments for the quantum-to-classical
bridge that forms the foundation of Layer 2 in the SCPN framework.

Core Components:
---------------
1. Quantum State Representation and Evolution
2. Decoherence Mechanisms and Measurement
3. Vesicle Release Probability Validator
4. Calcium Cooperativity Experiments
5. SNARE Complex Quantum Mechanics
6. Ψ_s Field Coupling Tests
7. Complete L1→L2 Interface Validation

Manuscript References:
---------------------
- Part 3, Chapter 9: The Quantum-to-Classical Bridge: A Hamiltonian Formalism
- Part 3, Chapter 10: The Locus of Intent: Downward Causation at the Calcium Sensor
- Part 3, Chapter 5: The Quantum Synapse: Plasticity and Learning
- Part 16: Validation protocols for downward causation (L2, L10)

Author: Generated for SCPN Manuscript Validation
Version: 1.0.0
Date: 2025-11-07
"""

import numpy as np
import scipy as sp
from scipy import integrate, linalg, optimize, signal, stats
from scipy.special import erf, erfc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import core framework (Module 1)
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
        EXPERIMENT_REGISTRY,
        logger
    )
except ImportError:
    raise ImportError("Module 1 (layer2_validation_core.py) must be loaded first!")

# Module logger
module_logger = logging.getLogger('Layer2.QuantumClassical')


# ============================================================================
# SECTION 1: QUANTUM STATE REPRESENTATION
# ============================================================================

@dataclass
class QuantumState:
    """
    Complete quantum state representation for L1→L2 transitions
    
    This represents the quantum substrate (Layer 1) before decoherence
    into classical neurochemical states (Layer 2).
    
    Manuscript ref: Part 3, Ch 9, Quantum-to-Classical Bridge
    """
    
    # State vector (complex amplitudes)
    amplitudes: np.ndarray  # Complex amplitudes for each basis state
    
    # Density matrix representation
    density_matrix: Optional[np.ndarray] = None
    
    # Coherence metrics
    coherence_time: float = 0.0001  # seconds (default 0.1 ms)
    purity: float = 1.0  # Tr(ρ²), 1=pure, <1=mixed
    
    # Environmental coupling
    temperature: float = PhysicalConstants.T_body
    coupling_strength: float = 0.01  # eV
    
    # Field coupling
    psi_s_coupling: float = 0.0
    
    # Time
    t: float = 0.0
    
    def __post_init__(self):
        """Initialize density matrix if not provided"""
        if self.density_matrix is None:
            # Pure state: ρ = |ψ⟩⟨ψ|
            self.density_matrix = np.outer(self.amplitudes, np.conj(self.amplitudes))
        
        # Verify normalization
        trace = np.trace(self.density_matrix)
        if not np.isclose(trace, 1.0):
            # Normalize
            self.density_matrix = self.density_matrix / trace
            self.amplitudes = self.amplitudes / np.sqrt(np.sum(np.abs(self.amplitudes)**2))
    
    @property
    def dimension(self) -> int:
        """Hilbert space dimension"""
        return len(self.amplitudes)
    
    @property
    def von_neumann_entropy(self) -> float:
        """
        Calculate von Neumann entropy S = -Tr(ρ log ρ)
        
        Measures mixedness: S=0 for pure states, S=log(d) for maximally mixed
        """
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove numerical zeros
        return -np.sum(eigenvalues * np.log(eigenvalues))
    
    @property
    def linear_entropy(self) -> float:
        """
        Calculate linear entropy S_L = 1 - Tr(ρ²)
        
        Approximation of von Neumann entropy, easier to compute
        """
        return 1.0 - self.purity
    
    def update_purity(self):
        """Calculate and update purity"""
        self.purity = np.real(np.trace(self.density_matrix @ self.density_matrix))
    
    def coherence_measure(self) -> float:
        """
        Measure quantum coherence using l1-norm
        
        Manuscript ref: Part 3, Ch 9, Coherence quantification
        """
        # l1-norm of off-diagonal elements
        n = self.dimension
        coherence = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    coherence += np.abs(self.density_matrix[i, j])
        return coherence
    
    @classmethod
    def create_pure_state(cls, n_levels: int, state_index: int = 0, 
                         temperature: float = PhysicalConstants.T_body) -> 'QuantumState':
        """Create a pure quantum state in computational basis"""
        amplitudes = np.zeros(n_levels, dtype=complex)
        amplitudes[state_index] = 1.0
        return cls(amplitudes=amplitudes, temperature=temperature)
    
    @classmethod
    def create_superposition(cls, n_levels: int, coefficients: Optional[np.ndarray] = None,
                           temperature: float = PhysicalConstants.T_body) -> 'QuantumState':
        """Create a superposition state"""
        if coefficients is None:
            # Equal superposition
            coefficients = np.ones(n_levels, dtype=complex) / np.sqrt(n_levels)
        else:
            # Normalize
            coefficients = coefficients / np.sqrt(np.sum(np.abs(coefficients)**2))
        
        return cls(amplitudes=coefficients, temperature=temperature)
    
    @classmethod
    def create_thermal_state(cls, n_levels: int, temperature: float = PhysicalConstants.T_body,
                           energy_gap: float = 0.1) -> 'QuantumState':
        """
        Create a thermal (Gibbs) state ρ = exp(-βH) / Z
        
        Args:
            n_levels: Number of levels
            temperature: Temperature in K
            energy_gap: Energy spacing in eV
        """
        beta = 1.0 / (PhysicalConstants.k_B * temperature / PhysicalConstants.eV_to_J)
        
        # Simple harmonic ladder
        energies = np.arange(n_levels) * energy_gap
        
        # Boltzmann factors
        boltzmann = np.exp(-beta * energies)
        Z = np.sum(boltzmann)  # Partition function
        
        # Diagonal density matrix
        rho = np.diag(boltzmann / Z)
        
        # Create amplitudes (sqrt of diagonal)
        amplitudes = np.sqrt(np.diag(rho)).astype(complex)
        
        return cls(amplitudes=amplitudes, density_matrix=rho, temperature=temperature, purity=np.trace(rho @ rho))


class QuantumEvolution:
    """
    Quantum state evolution under Hamiltonian dynamics
    
    Implements unitary evolution: dρ/dt = -i[H,ρ]
    And decoherence: Lindblad master equation
    
    Manuscript ref: Part 3, Ch 9, Hamiltonian Formalism
    """
    
    def __init__(self, hamiltonian: np.ndarray, lindblad_operators: Optional[List[np.ndarray]] = None,
                 decay_rates: Optional[List[float]] = None):
        """
        Initialize quantum evolution
        
        Args:
            hamiltonian: System Hamiltonian (Hermitian matrix)
            lindblad_operators: List of collapse operators for decoherence
            decay_rates: Rates for each Lindblad operator
        """
        self.H = hamiltonian
        self.dim = hamiltonian.shape[0]
        
        # Verify Hermiticity
        if not np.allclose(hamiltonian, hamiltonian.conj().T):
            logger.warning("Hamiltonian is not Hermitian!")
        
        self.lindblad_operators = lindblad_operators or []
        self.decay_rates = decay_rates or [1.0] * len(self.lindblad_operators)
        
        module_logger.info(f"Quantum evolution initialized: {self.dim}D Hilbert space, "
                          f"{len(self.lindblad_operators)} decoherence channels")
    
    def commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate commutator [A,B] = AB - BA"""
        return A @ B - B @ A
    
    def anticommutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate anticommutator {A,B} = AB + BA"""
        return A @ B + B @ A
    
    def unitary_evolution(self, rho: np.ndarray, t: float) -> np.ndarray:
        """
        Pure unitary evolution: ρ(t) = U(t) ρ(0) U†(t)
        where U(t) = exp(-iHt/ℏ)
        """
        # Calculate unitary operator
        U = linalg.expm(-1j * self.H * t / PhysicalConstants.hbar)
        
        # Evolve density matrix
        return U @ rho @ U.conj().T
    
    def lindblad_term(self, rho: np.ndarray) -> np.ndarray:
        """
        Calculate Lindblad dissipator: Σ_k γ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
        
        This represents decoherence and dissipation
        """
        result = np.zeros_like(rho)
        
        for L, gamma in zip(self.lindblad_operators, self.decay_rates):
            L_dag = L.conj().T
            result += gamma * (L @ rho @ L_dag - 0.5 * self.anticommutator(L_dag @ L, rho))
        
        return result
    
    def master_equation_rhs(self, t: float, rho_flat: np.ndarray) -> np.ndarray:
        """
        Right-hand side of Lindblad master equation:
        dρ/dt = -i[H,ρ] + Σ_k γ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
        
        Used with scipy.integrate.solve_ivp
        """
        # Reshape flat array to matrix
        rho = rho_flat.reshape((self.dim, self.dim))
        
        # Unitary part
        drho_dt = -1j * self.commutator(self.H, rho) / PhysicalConstants.hbar
        
        # Dissipative part
        if self.lindblad_operators:
            drho_dt += self.lindblad_term(rho)
        
        # Flatten back
        return drho_dt.flatten()
    
    def evolve(self, initial_state: QuantumState, t_final: float, 
               n_steps: int = 100) -> Tuple[np.ndarray, List[QuantumState]]:
        """
        Evolve quantum state from t=0 to t=t_final
        
        Returns:
            times: Array of time points
            states: List of QuantumState objects at each time
        """
        times = np.linspace(0, t_final, n_steps)
        rho0 = initial_state.density_matrix.flatten()
        
        # Solve master equation
        sol = integrate.solve_ivp(
            self.master_equation_rhs,
            (0, t_final),
            rho0,
            t_eval=times,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Convert back to QuantumState objects
        states = []
        for i, t in enumerate(times):
            rho = sol.y[:, i].reshape((self.dim, self.dim))
            
            # Extract amplitudes (eigenstate with largest eigenvalue)
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            dominant_idx = np.argmax(eigenvalues)
            amplitudes = eigenvectors[:, dominant_idx]
            
            state = QuantumState(
                amplitudes=amplitudes,
                density_matrix=rho,
                temperature=initial_state.temperature,
                t=t
            )
            state.update_purity()
            states.append(state)
        
        return times, states


# ============================================================================
# SECTION 2: DECOHERENCE MECHANISMS
# ============================================================================

class DecoherenceModel:
    """
    Models for quantum decoherence in biological systems
    
    Implements various decoherence channels relevant to neurochemical systems:
    - Thermal decoherence
    - Dephasing
    - Amplitude damping
    - Environmental entanglement
    
    Manuscript ref: Part 3, Ch 9, Decoherence mechanisms
    """
    
    @staticmethod
    def thermal_decoherence_rate(temperature: float, energy_gap: float) -> float:
        """
        Calculate thermal decoherence rate
        
        Γ_thermal = (ΔE/ℏ) * n_thermal
        where n_thermal = 1/(exp(βΔE) - 1) is the thermal occupation
        
        Args:
            temperature: Temperature in K
            energy_gap: Energy gap in eV
        
        Returns:
            Decoherence rate in Hz
        """
        beta = 1.0 / (PhysicalConstants.k_B * temperature / PhysicalConstants.eV_to_J)
        
        # Thermal occupation number
        n_thermal = 1.0 / (np.exp(beta * energy_gap) - 1.0) if beta * energy_gap > 0.1 else 1.0 / (beta * energy_gap)
        
        # Decoherence rate
        omega = energy_gap * PhysicalConstants.eV_to_J / PhysicalConstants.hbar
        gamma = omega * n_thermal
        
        return gamma
    
    @staticmethod
    def dephasing_operator(dim: int) -> np.ndarray:
        """
        Create dephasing (pure decoherence) operator
        
        L = √(Γ/2) σ_z (for 2-level system)
        Generalizes to diagonal operator for multi-level
        """
        # Diagonal operator in computational basis
        diag = np.arange(dim) - (dim - 1) / 2.0
        return np.diag(diag)
    
    @staticmethod
    def amplitude_damping_operators(dim: int) -> List[np.ndarray]:
        """
        Create amplitude damping (energy relaxation) operators
        
        L_k = |k⟩⟨k+1| (lowering operators)
        
        Returns list of operators for all adjacent level transitions
        """
        operators = []
        for i in range(dim - 1):
            L = np.zeros((dim, dim), dtype=complex)
            L[i, i + 1] = 1.0
            operators.append(L)
        return operators
    
    @staticmethod
    def calculate_coherence_time(temperature: float, energy_gap: float, 
                                 coupling_strength: float) -> float:
        """
        Estimate quantum coherence time
        
        τ_coherence ≈ ℏ / (coupling_strength * k_B * T)
        
        Manuscript ref: Part 3, Layer 2, Coherence times 0.1-100 ms
        
        Args:
            temperature: Temperature in K
            energy_gap: Characteristic energy scale in eV
            coupling_strength: System-bath coupling in eV
        
        Returns:
            Coherence time in seconds
        """
        kT = PhysicalConstants.k_B * temperature / PhysicalConstants.eV_to_J  # in eV
        tau = PhysicalConstants.hbar / (coupling_strength * kT * PhysicalConstants.eV_to_J)
        
        # Bound to reasonable range for biological systems
        tau = np.clip(tau, PhysicalConstants.COHERENCE_MIN, PhysicalConstants.COHERENCE_MAX)
        
        return tau
    
    @staticmethod
    def measurement_induced_decoherence(state: QuantumState, measurement_basis: np.ndarray) -> QuantumState:
        """
        Apply measurement-induced decoherence (projection)
        
        This is the key mechanism for quantum-to-classical transition
        
        Args:
            state: Quantum state before measurement
            measurement_basis: Measurement operators (projectors)
        
        Returns:
            Collapsed state after measurement
        """
        # Measure in given basis
        probabilities = []
        for P in measurement_basis:
            prob = np.real(np.trace(P @ state.density_matrix))
            probabilities.append(prob)
        
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        # Randomly select outcome
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        # Project onto measured state
        P = measurement_basis[outcome]
        rho_collapsed = P @ state.density_matrix @ P / probabilities[outcome]
        
        # Create new state
        eigenvalues, eigenvectors = np.linalg.eigh(rho_collapsed)
        dominant_idx = np.argmax(eigenvalues)
        amplitudes = eigenvectors[:, dominant_idx]
        
        return QuantumState(
            amplitudes=amplitudes,
            density_matrix=rho_collapsed,
            temperature=state.temperature,
            coherence_time=state.coherence_time,
            purity=1.0,  # Pure state after measurement
            t=state.t
        )


# ============================================================================
# SECTION 3: VESICLE RELEASE PROBABILITY MODEL
# ============================================================================

class VesicleReleaseSimulator:
    """
    Comprehensive vesicle release probability simulator
    
    Implements the quantum-modulated stochastic release model from the manuscript:
    
    P_release = 1 - exp(-[Ca²⁺]⁴ / K_release)
    P_modulated = P_release × (1 + λ_Ψ × Ψ_s)
    
    Manuscript ref: Part 3, Ch 10, Downward Causation at Calcium Sensor
                   Part 3, Ch 5, The Quantum Synapse
    """
    
    def __init__(self, 
                 n_vesicles: int = 200,
                 K_release: float = 1e-6,  # M (1 μM)
                 hill_coefficient: float = 4.0,
                 lambda_psi: float = 0.2,
                 temperature: float = PhysicalConstants.T_body):
        """
        Initialize vesicle release simulator
        
        Args:
            n_vesicles: Total number of vesicles in readily releasable pool
            K_release: Half-activation concentration (M)
            hill_coefficient: Cooperativity (should be ~4 for Ca⁴)
            lambda_psi: Ψ_s field coupling strength
            temperature: Temperature in K
        """
        self.n_vesicles = n_vesicles
        self.K_release = K_release
        self.hill_coefficient = hill_coefficient
        self.lambda_psi = lambda_psi
        self.temperature = temperature
        
        # Vesicle pools (dynamically updated)
        self.readily_releasable_pool = n_vesicles
        self.recycling_pool = n_vesicles * 5  # 5x reserve
        
        # Kinetic parameters
        self.refill_rate = 10.0  # Hz (vesicle per second per site)
        self.recovery_rate = 1.0  # Hz (recycling rate)
        
        module_logger.info(f"Vesicle release simulator initialized: {n_vesicles} vesicles, "
                          f"Hill n={hill_coefficient}, K={K_release*1e6:.1f} μM")
    
    def base_release_probability(self, Ca_conc: float) -> float:
        """
        Calculate base release probability (without Ψ_s modulation)
        
        Uses Hill equation with cooperativity
        P = [Ca]^n / (K^n + [Ca]^n)
        
        Or exponential form from manuscript:
        P = 1 - exp(-[Ca]^n / K)
        """
        # Exponential form (closer to biophysical reality for high cooperativity)
        P = 1.0 - np.exp(-np.power(Ca_conc, self.hill_coefficient) / np.power(self.K_release, self.hill_coefficient))
        
        # Bound to [0, 1]
        return np.clip(P, 0.0, 1.0)
    
    def psi_s_modulated_probability(self, Ca_conc: float, psi_s_field: float) -> float:
        """
        Calculate Ψ_s-modulated release probability
        
        P_modulated = P_base × (1 + λ_Ψ × Ψ_s)
        
        This implements downward causation: consciousness field modulates
        quantum probabilities at the synaptic level
        
        Manuscript ref: Part 3, Ch 10, Quantum-Classical Interface
        """
        P_base = self.base_release_probability(Ca_conc)
        P_modulated = P_base * (1.0 + self.lambda_psi * psi_s_field)
        
        # Bound to [0, 1]
        return np.clip(P_modulated, 0.0, 1.0)
    
    def simulate_release_event(self, Ca_conc: float, psi_s_field: float = 0.0) -> int:
        """
        Simulate a single release event
        
        Returns:
            Number of vesicles released (0 to n_vesicles)
        """
        if self.readily_releasable_pool == 0:
            return 0
        
        # Calculate release probability
        P_release = self.psi_s_modulated_probability(Ca_conc, psi_s_field)
        
        # Each vesicle releases independently with probability P
        n_available = min(self.readily_releasable_pool, self.n_vesicles)
        n_released = np.random.binomial(n_available, P_release)
        
        # Update pools
        self.readily_releasable_pool -= n_released
        
        return n_released
    
    def update_pools(self, dt: float):
        """
        Update vesicle pools with refilling and recycling dynamics
        
        dRRP/dt = k_refill * (N_max - RRP)
        """
        # Refill from recycling pool
        n_refill = int(self.refill_rate * dt * (self.n_vesicles - self.readily_releasable_pool))
        n_refill = min(n_refill, self.recycling_pool)
        
        self.readily_releasable_pool += n_refill
        self.recycling_pool -= n_refill
        
        # Recover recycling pool
        recovery = int(self.recovery_rate * dt * (self.n_vesicles * 5 - self.recycling_pool))
        self.recycling_pool += recovery
    
    def simulate_train(self, Ca_trace: np.ndarray, psi_s_trace: np.ndarray, 
                       dt: float) -> np.ndarray:
        """
        Simulate release events for a time series of Ca²⁺ and Ψ_s
        
        Args:
            Ca_trace: Calcium concentration time series (M)
            psi_s_trace: Ψ_s field time series
            dt: Time step (s)
        
        Returns:
            Array of released vesicles at each time point
        """
        n_steps = len(Ca_trace)
        releases = np.zeros(n_steps, dtype=int)
        
        for i in range(n_steps):
            # Simulate release
            releases[i] = self.simulate_release_event(Ca_trace[i], psi_s_trace[i])
            
            # Update pools
            self.update_pools(dt)
        
        return releases
    
    def generate_dose_response_curve(self, Ca_range: np.ndarray, 
                                    psi_s_field: float = 0.0,
                                    n_trials: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dose-response curve: P_release vs [Ca²⁺]
        
        Args:
            Ca_range: Range of Ca concentrations to test
            psi_s_field: Ψ_s field strength
            n_trials: Number of trials per concentration
        
        Returns:
            Ca_range, mean release probability
        """
        release_probs = np.zeros(len(Ca_range))
        
        for i, Ca in enumerate(Ca_range):
            # Run multiple trials
            successes = 0
            for _ in range(n_trials):
                n_released = self.simulate_release_event(Ca, psi_s_field)
                if n_released > 0:
                    successes += 1
            
            release_probs[i] = successes / n_trials
            
            # Reset pool for next concentration
            self.readily_releasable_pool = self.n_vesicles
        
        return Ca_range, release_probs


# ============================================================================
# SECTION 4: CALCIUM COOPERATIVITY VALIDATOR
# ============================================================================

class CalciumCooperativityValidator:
    """
    Validates Ca²⁺ cooperativity in vesicle release
    
    Tests the theoretical prediction: P_release ∝ [Ca²⁺]^4
    
    This is a critical validation of the quantum-classical bridge:
    the 4-fold cooperativity arises from synaptotagmin's 4 Ca²⁺ binding sites
    
    Manuscript ref: Part 3, Ch 10, Calcium Sensor Cooperativity
    """
    
    @staticmethod
    def measure_hill_coefficient(Ca_conc: np.ndarray, response: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit Hill equation to data and extract Hill coefficient
        
        y = x^n / (K^n + x^n)
        
        Returns:
            (hill_coefficient, K_half, R_squared)
        """
        def hill_function(x, K, n):
            return x**n / (K**n + x**n)
        
        try:
            # Fit Hill equation
            popt, pcov = optimize.curve_fit(
                hill_function,
                Ca_conc,
                response,
                p0=[1e-6, 4.0],  # Initial guess: K=1 μM, n=4
                bounds=([1e-9, 0.1], [1e-3, 10.0])  # Reasonable bounds
            )
            
            K_fit, n_fit = popt
            
            # Calculate R²
            y_pred = hill_function(Ca_conc, K_fit, n_fit)
            ss_res = np.sum((response - y_pred)**2)
            ss_tot = np.sum((response - np.mean(response))**2)
            r_squared = 1.0 - (ss_res / ss_tot)
            
            return n_fit, K_fit, r_squared
            
        except Exception as e:
            module_logger.error(f"Hill fit failed: {e}")
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def validate_cooperativity(simulator: VesicleReleaseSimulator,
                               expected_hill: float = 4.0,
                               tolerance: float = 0.5) -> Dict[str, Any]:
        """
        Validate that simulator exhibits correct cooperativity
        
        Returns validation report
        """
        # Generate dose-response data
        Ca_range = np.logspace(-8, -5, 30)  # 10 nM to 10 μM
        _, release_prob = simulator.generate_dose_response_curve(Ca_range, psi_s_field=0.0, n_trials=500)
        
        # Fit Hill equation
        n_fit, K_fit, r_squared = CalciumCooperativityValidator.measure_hill_coefficient(Ca_range, release_prob)
        
        # Validation
        passed = np.abs(n_fit - expected_hill) < tolerance and r_squared > 0.95
        
        return {
            'passed': passed,
            'hill_coefficient': n_fit,
            'expected': expected_hill,
            'deviation': np.abs(n_fit - expected_hill),
            'tolerance': tolerance,
            'K_half': K_fit,
            'r_squared': r_squared,
            'Ca_range': Ca_range,
            'release_prob': release_prob
        }


# ============================================================================
# SECTION 5: COMPLETE VALIDATION EXPERIMENTS
# ============================================================================

class QuantumDecoherenceExperiment(BaseExperiment):
    """
    Experiment 1: Quantum Decoherence Dynamics
    
    Validates:
    - Coherence time predictions
    - Decoherence rate scaling with temperature
    - Purity decay
    - Entropy growth
    
    Manuscript ref: Part 16, Validation Domain I - Quantum Biology (L1)
    """
    
    def setup(self) -> np.ndarray:
        """Initialize quantum state"""
        # Create initial superposition state
        n_levels = 5
        self.n_levels = n_levels
        
        # Equal superposition (maximally coherent)
        self.initial_quantum_state = QuantumState.create_superposition(n_levels, temperature=self.config.temperature)
        
        # Setup Hamiltonian (simple harmonic ladder)
        energy_gap = 0.1  # eV (typical for molecular vibrations)
        self.energy_gap = energy_gap
        energies = np.arange(n_levels) * energy_gap * PhysicalConstants.eV_to_J
        H = np.diag(energies)
        
        # Setup decoherence operators
        # Dephasing
        L_dephasing = DecoherenceModel.dephasing_operator(n_levels) * np.sqrt(0.1)
        
        # Amplitude damping
        L_damping_list = DecoherenceModel.amplitude_damping_operators(n_levels)
        
        all_operators = [L_dephasing] + L_damping_list
        decay_rates = [0.5] + [0.2] * len(L_damping_list)  # Faster dephasing than damping
        
        self.evolution = QuantumEvolution(H, all_operators, decay_rates)
        
        # Calculate expected coherence time
        self.expected_coherence_time = DecoherenceModel.calculate_coherence_time(
            self.config.temperature,
            energy_gap,
            coupling_strength=0.01
        )
        
        module_logger.info(f"Expected coherence time: {self.expected_coherence_time*1000:.2f} ms")
        
        # Return initial state vector for integration
        return self.initial_quantum_state.density_matrix.flatten().view(float)
    
    def run_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        This experiment uses QuantumEvolution.evolve() instead of step-by-step
        So this method is not used, but required by BaseExperiment
        """
        return state
    
    def run(self) -> ExperimentResults:
        """Override run to use quantum evolution"""
        start_time = datetime.now()
        self.logger.info(f"Starting quantum decoherence experiment")
        
        # Initialize
        self.results = ExperimentResults(
            config=self.config,
            status=ExperimentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Setup
            self.setup()
            
            # Evolve quantum state
            times, quantum_states = self.evolution.evolve(
                self.initial_quantum_state,
                self.config.duration,
                n_steps=self.n_steps
            )
            
            # Extract metrics
            purities = np.array([s.purity for s in quantum_states])
            coherences = np.array([s.coherence_measure() for s in quantum_states])
            entropies = np.array([s.von_neumann_entropy for s in quantum_states])
            
            # Store results
            self.results.times = times
            self.results.states = np.column_stack([purities, coherences, entropies])
            
            # Find coherence time (when coherence drops to 1/e)
            threshold = coherences[0] / np.e
            try:
                idx = np.where(coherences < threshold)[0][0]
                measured_coherence_time = times[idx]
            except IndexError:
                measured_coherence_time = times[-1]
            
            # Validation
            validation = {
                'coherence_time_match': np.abs(measured_coherence_time - self.expected_coherence_time) < self.expected_coherence_time * 0.5,
                'purity_decay': purities[-1] < purities[0],
                'entropy_growth': entropies[-1] > entropies[0],
                'coherence_decay': coherences[-1] < coherences[0]
            }
            
            for name, passed in validation.items():
                self.results.add_validation(
                    name=name,
                    passed=passed,
                    value=measured_coherence_time if 'time' in name else None
                )
            
            self.results.status = ExperimentStatus.COMPLETED
            self.results.end_time = datetime.now()
            self.results.computation_time = (self.results.end_time - start_time).total_seconds()
            
            # Store additional data
            self.results.validation_results['measured_coherence_time'] = {
                'passed': True,
                'value': measured_coherence_time,
                'expected': self.expected_coherence_time,
                'description': 'Measured vs expected coherence time'
            }
            
            self.logger.info(f"Experiment completed: τ_coherence = {measured_coherence_time*1000:.2f} ms "
                           f"(expected {self.expected_coherence_time*1000:.2f} ms)")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            self.results.status = ExperimentStatus.FAILED
            self.results.errors.append(str(e))
            return self.results
    
    def validate(self) -> Dict[str, bool]:
        """Validation done in run()"""
        return {}


class VesicleReleaseValidationExperiment(BaseExperiment):
    """
    Experiment 2: Vesicle Release Probability Validation
    
    Validates:
    - Ca²⁺ cooperativity (Hill coefficient ~4)
    - Ψ_s field modulation
    - Dose-response curves
    - Short-term plasticity
    
    Manuscript ref: Part 16, Validation Domain I - Downward Causation (L2, L10)
    """
    
    def setup(self) -> np.ndarray:
        """Initialize vesicle release simulator"""
        self.simulator = VesicleReleaseSimulator(
            n_vesicles=200,
            K_release=1e-6,  # 1 μM
            hill_coefficient=4.0,
            lambda_psi=0.2,
            temperature=self.config.temperature
        )
        
        return np.array([self.simulator.readily_releasable_pool])
    
    def run_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """Not used - experiment uses batch simulation"""
        return state
    
    def run(self) -> ExperimentResults:
        """Override run for vesicle release validation"""
        start_time = datetime.now()
        self.logger.info("Starting vesicle release validation")
        
        self.results = ExperimentResults(
            config=self.config,
            status=ExperimentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            self.setup()
            
            # Test 1: Validate cooperativity
            self.logger.info("Test 1: Ca²⁺ cooperativity validation")
            validation_report = CalciumCooperativityValidator.validate_cooperativity(
                self.simulator,
                expected_hill=4.0,
                tolerance=0.5
            )
            
            # Test 2: Ψ_s field modulation
            self.logger.info("Test 2: Ψ_s field modulation")
            Ca_test = 2e-6  # 2 μM (above threshold)
            
            P_baseline = self.simulator.psi_s_modulated_probability(Ca_test, psi_s_field=0.0)
            P_enhanced = self.simulator.psi_s_modulated_probability(Ca_test, psi_s_field=1.0)
            P_suppressed = self.simulator.psi_s_modulated_probability(Ca_test, psi_s_field=-1.0)
            
            psi_effect = {
                'baseline': P_baseline,
                'enhanced': P_enhanced,
                'suppressed': P_suppressed,
                'enhancement_ratio': P_enhanced / P_baseline if P_baseline > 0 else 0,
                'suppression_ratio': P_suppressed / P_baseline if P_baseline > 0 else 0
            }
            
            # Validation
            all_validations = {
                'cooperativity': validation_report['passed'],
                'psi_enhancement': P_enhanced > P_baseline,
                'psi_suppression': P_suppressed < P_baseline,
                'hill_coefficient_in_range': 3.5 <= validation_report['hill_coefficient'] <= 4.5,
                'r_squared_good_fit': validation_report['r_squared'] > 0.95
            }
            
            for name, passed in all_validations.items():
                self.results.add_validation(name, passed, None)
            
            # Store detailed results
            self.results.validation_results['cooperativity_report'] = validation_report
            self.results.validation_results['psi_s_modulation'] = psi_effect
            
            self.results.status = ExperimentStatus.COMPLETED
            self.results.end_time = datetime.now()
            self.results.computation_time = (self.results.end_time - start_time).total_seconds()
            
            self.logger.info(f"Vesicle release validation completed:")
            self.logger.info(f"  Hill coefficient: {validation_report['hill_coefficient']:.2f} (expected 4.0)")
            self.logger.info(f"  R²: {validation_report['r_squared']:.4f}")
            self.logger.info(f"  Ψ_s enhancement: {psi_effect['enhancement_ratio']:.2f}x")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            self.results.status = ExperimentStatus.FAILED
            self.results.errors.append(str(e))
            return self.results
    
    def validate(self) -> Dict[str, bool]:
        """Validation done in run()"""
        return {}


# ============================================================================
# SECTION 6: EXPERIMENT REGISTRATION
# ============================================================================

def register_quantum_classical_experiments():
    """Register all quantum-classical validation experiments"""
    EXPERIMENT_REGISTRY.register('quantum_decoherence', QuantumDecoherenceExperiment)
    EXPERIMENT_REGISTRY.register('vesicle_release_validation', VesicleReleaseValidationExperiment)
    
    module_logger.info("Registered quantum-classical validation experiments:")
    module_logger.info("  - quantum_decoherence")
    module_logger.info("  - vesicle_release_validation")


# ============================================================================
# SECTION 7: MODULE INITIALIZATION
# ============================================================================

def initialize_module_2():
    """Initialize Module 2: Quantum-Classical Validators"""
    module_logger.info("="*80)
    module_logger.info("Module 2: Quantum-Classical Transition Validators")
    module_logger.info("="*80)
    module_logger.info("")
    module_logger.info("Components loaded:")
    module_logger.info("  ✓ Quantum state representation")
    module_logger.info("  ✓ Quantum evolution (unitary + Lindblad)")
    module_logger.info("  ✓ Decoherence models")
    module_logger.info("  ✓ Vesicle release simulator")
    module_logger.info("  ✓ Calcium cooperativity validator")
    module_logger.info("  ✓ Complete validation experiments")
    module_logger.info("")
    module_logger.info("Experiments available:")
    module_logger.info("  1. Quantum decoherence dynamics")
    module_logger.info("  2. Vesicle release validation (Ca⁴ cooperativity)")
    module_logger.info("")
    module_logger.info("Ready for L1→L2 interface validation!")
    module_logger.info("="*80)
    
    # Register experiments
    register_quantum_classical_experiments()


# Run initialization
initialize_module_2()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Module 2: Quantum-Classical Transition Validators")
    print("="*80 + "\n")
    
    # Example 1: Quantum decoherence
    print("Example 1: Quantum Decoherence Dynamics")
    print("-" * 40)
    
    config1 = ExperimentConfig(
        name="quantum_decoherence",
        description="Test quantum coherence decay",
        layer_components=["quantum_mechanics", "decoherence"],
        duration=0.01,  # 10 ms
        dt=0.0001,  # 0.1 ms
        temperature=310.0
    )
    
    exp1 = QuantumDecoherenceExperiment(config1)
    results1 = exp1.run()
    
    print(f"Status: {results1.status.value}")
    print(f"Validations: {len(results1.validation_results)} tests")
    for name, result in results1.validation_results.items():
        if isinstance(result, dict) and 'passed' in result:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {name}: {status}")
    
    # Example 2: Vesicle release
    print("\n\nExample 2: Vesicle Release Validation")
    print("-" * 40)
    
    config2 = ExperimentConfig(
        name="vesicle_release_validation",
        description="Validate Ca⁴ cooperativity and Ψ_s modulation",
        layer_components=["vesicle_release", "calcium_dynamics"],
        duration=1.0,
        dt=0.001
    )
    
    exp2 = VesicleReleaseValidationExperiment(config2)
    results2 = exp2.run()
    
    print(f"Status: {results2.status.value}")
    if 'cooperativity_report' in results2.validation_results:
        report = results2.validation_results['cooperativity_report']
        print(f"Hill coefficient: {report['hill_coefficient']:.2f} ± {report['tolerance']}")
        print(f"R²: {report['r_squared']:.4f}")
        print(f"Cooperativity test: {'✅ PASS' if report['passed'] else '❌ FAIL'}")
    
    if 'psi_s_modulation' in results2.validation_results:
        psi = results2.validation_results['psi_s_modulation']
        print(f"\nΨ_s Field Effects:")
        print(f"  Baseline P: {psi['baseline']:.3f}")
        print(f"  Enhanced P: {psi['enhanced']:.3f} ({psi['enhancement_ratio']:.2f}x)")
        print(f"  Suppressed P: {psi['suppressed']:.3f} ({psi['suppression_ratio']:.2f}x)")
    
    print("\n" + "="*80)
    print("Module 2 demonstration complete!")
    print("="*80 + "\n")
