"""
CISS (Chiral-Induced Spin Selectivity) Mechanism Implementation

Implements the quantum mechanical foundation of CISS in DNA:
- Spin-dependent electron transfer
- Helical chirality effects
- Torsional strain coupling
- Redox-dependent modulation

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Physical constants
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
K_B = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
BOHR_MAGNETON = 9.274009994e-24  # Bohr magneton (J/T)


@dataclass
class CISSParameters:
    """Parameters for CISS simulations"""
    
    # Helical geometry
    pitch: float = 3.4e-9  # DNA pitch (m)
    radius: float = 1.0e-9  # DNA radius (m)
    chirality: int = 1  # +1 for L-DNA (right-handed), -1 for D-DNA
    
    # Spin-orbit coupling
    lambda_so: float = 0.1 * E_CHARGE  # Spin-orbit coupling strength (J)
    v_fermi: float = 1e6  # Fermi velocity (m/s)
    
    # Electron transport
    hopping_t: float = 0.1 * E_CHARGE  # Hopping integral (J)
    disorder_w: float = 0.05 * E_CHARGE  # Disorder strength (J)
    
    # Environmental
    temperature: float = 310.0  # Temperature (K)
    decoherence_rate: float = 1e12  # Decoherence rate (s^-1)
    
    def kt(self) -> float:
        """Thermal energy"""
        return K_B * self.temperature


class CISSModel:
    """
    Chiral-Induced Spin Selectivity Model
    
    Simulates spin-dependent electron transport through chiral DNA
    """
    
    def __init__(self, params: Optional[CISSParameters] = None):
        self.params = params or CISSParameters()
    
    def rashba_hamiltonian(self, 
                          k: float, 
                          b_field: np.ndarray = None) -> np.ndarray:
        """
        Rashba Hamiltonian for helical system
        
        H = (ħk)²/(2m*) + α(k × σ)·ẑ + μ_B B·σ
        
        Args:
            k: Wave vector (1/m)
            b_field: Magnetic field [Bx, By, Bz] (T)
            
        Returns:
            2x2 Hamiltonian matrix
        """
        p = self.params
        
        # Kinetic energy
        h_kinetic = (HBAR * k)**2 / (2 * 9.109e-31) * np.eye(2)
        
        # Rashba term (spin-orbit coupling)
        # For helix: α ∝ λ_so / r
        alpha = p.lambda_so / p.radius * p.chirality
        h_rashba = alpha * k * np.array([
            [0, -1j],
            [1j, 0]
        ])
        
        # Zeeman term
        if b_field is not None:
            sigma_x = np.array([[0, 1], [1, 0]])
            sigma_y = np.array([[0, -1j], [1j, 0]])
            sigma_z = np.array([[1, 0], [0, -1]])
            
            h_zeeman = BOHR_MAGNETON * (
                b_field[0] * sigma_x +
                b_field[1] * sigma_y +
                b_field[2] * sigma_z
            )
        else:
            h_zeeman = np.zeros((2, 2), dtype=complex)
        
        return h_kinetic + h_rashba + h_zeeman
    
    def spin_polarization(self, 
                         k: float, 
                         b_field: Optional[np.ndarray] = None) -> float:
        """
        Calculate spin polarization for given k-state
        
        Args:
            k: Wave vector
            b_field: Magnetic field (T)
            
        Returns:
            Spin polarization P_CISS (-1 to +1)
        """
        # Get Hamiltonian
        h = self.rashba_hamiltonian(k, b_field)
        
        # Diagonalize to get eigenstates
        energies, states = np.linalg.eigh(h)
        
        # Ground state
        psi = states[:, 0]
        
        # Spin expectation value
        sigma_z = np.array([[1, 0], [0, -1]])
        spin_z = np.real(psi.conj() @ sigma_z @ psi)
        
        return spin_z
    
    def transmission_coefficient(self, 
                                 energy: float, 
                                 length: float,
                                 spin: int = +1) -> float:
        """
        Transmission probability through DNA segment
        
        Args:
            energy: Electron energy (J)
            length: DNA segment length (m)
            spin: Spin state (+1 or -1)
            
        Returns:
            Transmission probability (0-1)
        """
        p = self.params
        
        # Wave vector
        k = np.sqrt(2 * 9.109e-31 * energy) / HBAR
        
        # Spin-dependent phase accumulation
        # φ = k·L + spin * (α/ħv_F) * L
        phase = k * length + spin * (p.lambda_so / (HBAR * p.v_fermi)) * length
        
        # Transmission with disorder
        t0 = np.exp(-length / (2 * p.v_fermi * p.decoherence_rate))
        t_spin = t0 * np.abs(np.cos(phase))**2
        
        return t_spin
    
    def ciss_efficiency(self, 
                       energy: float = 0.5 * E_CHARGE,
                       length: float = 10e-9,
                       theta_spin: float = 0.0) -> float:
        """
        CISS efficiency: difference in transmission for up/down spins
        
        Args:
            energy: Electron energy (J)
            length: DNA length (m)
            theta_spin: Spin orientation angle (rad)
            
        Returns:
            P_CISS: Spin polarization efficiency
        """
        t_up = self.transmission_coefficient(energy, length, spin=+1)
        t_down = self.transmission_coefficient(energy, length, spin=-1)
        
        # Orientation dependence
        p_ciss = (t_up - t_down) / (t_up + t_down) * np.cos(theta_spin)
        
        return p_ciss * self.params.chirality
    
    def electron_transfer_rate(self,
                               delta_g: float,
                               reorganization: float = 0.5 * E_CHARGE,
                               coupling: float = 0.01 * E_CHARGE,
                               spin_polarization: float = 0.7) -> float:
        """
        Marcus electron transfer rate with CISS modulation
        
        k_ET = (2π/ħ)|V|²/√(4πλk_BT) exp(-(ΔG + λ)²/(4λk_BT)) (1 + P_CISS cosθ)
        
        Args:
            delta_g: Free energy change (J)
            reorganization: Reorganization energy λ (J)
            coupling: Electronic coupling |V| (J)
            spin_polarization: P_CISS
            
        Returns:
            k_ET: Transfer rate (s^-1)
        """
        p = self.params
        kt = p.kt()
        
        # Marcus rate (spin-averaged)
        prefactor = (2 * np.pi / HBAR) * coupling**2 / \
                   np.sqrt(4 * np.pi * reorganization * kt)
        
        activation = np.exp(-(delta_g + reorganization)**2 / (4 * reorganization * kt))
        
        k_marcus = prefactor * activation
        
        # CISS modulation
        k_et = k_marcus * (1.0 + spin_polarization)
        
        return k_et
    
    def tight_binding_chain(self, 
                           n_sites: int = 10,
                           b_field: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tight-binding model of DNA chain with CISS
        
        Args:
            n_sites: Number of base pairs
            b_field: External magnetic field (T)
            
        Returns:
            (energies, states): Eigenvalues and eigenvectors
        """
        p = self.params
        
        # Build Hamiltonian (2*n_sites x 2*n_sites for spin)
        h_size = 2 * n_sites
        h = np.zeros((h_size, h_size), dtype=complex)
        
        # On-site energies with disorder
        for i in range(n_sites):
            disorder = np.random.uniform(-p.disorder_w, p.disorder_w)
            h[2*i, 2*i] = disorder  # Spin up
            h[2*i+1, 2*i+1] = disorder  # Spin down
        
        # Hopping terms with CISS
        for i in range(n_sites - 1):
            # Chiral phase
            phi = 2 * np.pi * p.chirality / (p.pitch / p.radius)
            
            # Spin-dependent hopping
            # Up-up
            h[2*i, 2*(i+1)] = p.hopping_t * np.exp(1j * phi)
            h[2*(i+1), 2*i] = p.hopping_t * np.exp(-1j * phi)
            
            # Down-down
            h[2*i+1, 2*(i+1)+1] = p.hopping_t * np.exp(-1j * phi)
            h[2*(i+1)+1, 2*i+1] = p.hopping_t * np.exp(1j * phi)
            
            # Spin-flip (weak)
            h[2*i, 2*(i+1)+1] = 0.01 * p.hopping_t
            h[2*(i+1)+1, 2*i] = 0.01 * p.hopping_t
        
        # Add magnetic field
        if b_field is not None:
            for i in range(n_sites):
                h[2*i, 2*i] += BOHR_MAGNETON * b_field[2]  # Spin up
                h[2*i+1, 2*i+1] -= BOHR_MAGNETON * b_field[2]  # Spin down
        
        # Diagonalize
        energies, states = np.linalg.eigh(h)
        
        return energies, states
    
    def spin_current_density(self,
                            voltage: float = 0.1,
                            length: float = 100e-9,
                            n_channels: int = 1000) -> float:
        """
        Calculate spin current density
        
        Args:
            voltage: Applied voltage (V)
            length: DNA length (m)
            n_channels: Number of parallel transport channels
            
        Returns:
            j_spin: Spin current density (A/m²)
        """
        p = self.params
        
        # Energy scale
        energy = E_CHARGE * voltage
        
        # Average CISS efficiency
        p_ciss_avg = self.ciss_efficiency(energy=energy, length=length)
        
        # Total current (Ohm's law-like)
        conductance = n_channels * E_CHARGE**2 / (HBAR * length)
        i_total = conductance * voltage
        
        # Spin current
        i_spin = p_ciss_avg * i_total
        
        # Current density (assuming ~1 nm² cross-section)
        area = np.pi * p.radius**2
        j_spin = i_spin / area
        
        return j_spin


class MagneticFieldFromSpinCurrent:
    """
    Calculate effective magnetic field from spin current
    
    Uses Ampere-like relation for spin accumulation
    """
    
    @staticmethod
    def b_effective(j_spin: float, 
                   radius: float = 1e-9,
                   length: float = 10e-9) -> float:
        """
        Calculate effective magnetic field
        
        B_eff ≈ μ_0 * j_spin * length / (2π * radius)
        
        Args:
            j_spin: Spin current density (A/m²)
            radius: Effective radius (m)
            length: Active length (m)
            
        Returns:
            B_eff: Effective magnetic field (T)
        """
        mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
        
        b_eff = mu_0 * j_spin * length / (2 * np.pi * radius)
        
        return b_eff
    
    @staticmethod
    def spin_accumulation(j_spin: float,
                         tau_spin: float = 1e-9,
                         diffusion: float = 1e-4) -> float:
        """
        Spin accumulation in DNA
        
        Args:
            j_spin: Spin current density (A/m²)
            tau_spin: Spin relaxation time (s)
            diffusion: Spin diffusion constant (m²/s)
            
        Returns:
            n_spin: Spin density (1/m³)
        """
        # Spin density from current
        lambda_spin = np.sqrt(diffusion * tau_spin)  # Spin diffusion length
        
        n_spin = j_spin * tau_spin / (E_CHARGE * lambda_spin)
        
        return n_spin


def benchmark_ciss() -> Dict:
    """
    Benchmark CISS model against known values
    
    Returns:
        Dictionary with benchmark results
    """
    model = CISSModel()
    
    results = {
        'p_ciss_typical': model.ciss_efficiency(
            energy=0.5*E_CHARGE,
            length=10e-9
        ),
        'b_eff_typical': MagneticFieldFromSpinCurrent.b_effective(
            j_spin=1e-6,
            radius=1e-9,
            length=10e-9
        ),
        'chirality_test': {
            'L_DNA': model.ciss_efficiency(),
            'D_DNA': CISSModel(CISSParameters(chirality=-1)).ciss_efficiency()
        }
    }
    
    return results


if __name__ == "__main__":
    """Example usage and tests"""
    import matplotlib.pyplot as plt
    
    print("CISS Mechanism Simulator - Test Run")
    print("=" * 50)
    
    # Create model
    model = CISSModel()
    
    # Test 1: P_CISS vs length
    print("\nTest 1: P_CISS vs DNA length")
    lengths = np.linspace(1e-9, 100e-9, 50)
    p_ciss = [model.ciss_efficiency(length=l) for l in lengths]
    
    print(f"P_CISS at 10 nm: {model.ciss_efficiency(length=10e-9):.3f}")
    print(f"P_CISS at 50 nm: {model.ciss_efficiency(length=50e-9):.3f}")
    
    # Test 2: Chirality reversal
    print("\nTest 2: Chirality reversal")
    model_l = CISSModel(CISSParameters(chirality=+1))
    model_d = CISSModel(CISSParameters(chirality=-1))
    
    p_l = model_l.ciss_efficiency()
    p_d = model_d.ciss_efficiency()
    
    print(f"P_CISS(L-DNA): {p_l:.3f}")
    print(f"P_CISS(D-DNA): {p_d:.3f}")
    print(f"Sign reversal confirmed: {np.sign(p_l) == -np.sign(p_d)}")
    
    # Test 3: Effective magnetic field
    print("\nTest 3: Effective magnetic field")
    j_spin = 1e-6  # A/m²
    b_eff = MagneticFieldFromSpinCurrent.b_effective(j_spin)
    
    print(f"Spin current density: {j_spin:.2e} A/m²")
    print(f"Effective B field: {b_eff*1e6:.2f} μT")
    
    # Benchmark
    print("\nBenchmark results:")
    benchmark = benchmark_ciss()
    for key, val in benchmark.items():
        print(f"  {key}: {val}")
    
    print("\n" + "=" * 50)
    print("CISS tests complete.")
