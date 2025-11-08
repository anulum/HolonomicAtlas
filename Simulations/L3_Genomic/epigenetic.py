"""
Epigenetic State Dynamics - Ising Model
========================================

Implements epigenetic phase transitions and bistable memory.

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
from scipy.stats import entropy
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EpigeneticParameters:
    """Parameters for epigenetic Ising model"""
    
    # System size
    n_genes: int = 1000
    n_cells: int = 100
    
    # Ising model parameters
    j_coupling: float = 1.0  # Coupling strength (units of k_B*T)
    h_field: float = 0.0  # External field
    
    # Temperature
    temperature: float = 310.0  # K
    k_b: float = 1.380649e-23  # J/K
    
    # Critical temperature for methylation
    t_c: float = 310.0  # K (at body temperature)
    
    # Dynamics
    flip_rate: float = 1.0  # Attempt rate (1/s)
    
    def beta(self) -> float:
        """Inverse temperature"""
        return 1.0 / (self.k_b * self.temperature)
    
    def kt(self) -> float:
        """Thermal energy"""
        return self.k_b * self.temperature


class EpigeneticIsing:
    """
    Epigenetic State as Ising Model
    
    Hamiltonian:
    H = -J Σ_<ij> σ_i σ_j - h Σ_i σ_i - ΔS·∇T
    
    σ_i = +1 (methylated), -1 (unmethylated)
    """
    
    def __init__(self, params: Optional[EpigeneticParameters] = None):
        self.params = params or EpigeneticParameters()
        self._init_state()
    
    def _init_state(self):
        """Initialize random epigenetic state"""
        p = self.params
        
        # Spin state: +1 (methylated), -1 (unmethylated)
        self.spins = np.random.choice([-1, 1], size=(p.n_cells, p.n_genes))
        
        # Coupling matrix (simplified: nearest neighbor)
        self.coupling = self._build_coupling_matrix()
    
    def _build_coupling_matrix(self) -> np.ndarray:
        """Build gene-gene coupling matrix"""
        p = self.params
        
        # Simplified: nearest neighbor coupling on 1D chain
        coupling = np.zeros((p.n_genes, p.n_genes))
        
        for i in range(p.n_genes - 1):
            coupling[i, i+1] = p.j_coupling
            coupling[i+1, i] = p.j_coupling
        
        return coupling
    
    def energy(self, cell_idx: int = 0) -> float:
        """
        Compute Hamiltonian energy for one cell
        
        Args:
            cell_idx: Cell index
            
        Returns:
            Energy (units of k_B*T)
        """
        p = self.params
        spins = self.spins[cell_idx]
        
        # Interaction energy: -J Σ σ_i σ_j
        interaction = 0.0
        for i in range(p.n_genes):
            for j in range(p.n_genes):
                if self.coupling[i, j] != 0:
                    interaction -= self.coupling[i, j] * spins[i] * spins[j]
        
        # External field energy: -h Σ σ_i
        field_energy = -p.h_field * np.sum(spins)
        
        total = interaction + field_energy
        
        return total
    
    def local_field(self, cell_idx: int, gene_idx: int) -> float:
        """
        Compute local field acting on one spin
        
        h_local = J Σ_j σ_j + h
        
        Args:
            cell_idx: Cell index
            gene_idx: Gene index
            
        Returns:
            Local field
        """
        p = self.params
        spins = self.spins[cell_idx]
        
        # Sum over neighbors
        h_local = 0.0
        for j in range(p.n_genes):
            if self.coupling[gene_idx, j] != 0:
                h_local += self.coupling[gene_idx, j] * spins[j]
        
        h_local += p.h_field
        
        return h_local
    
    def metropolis_step(self, cell_idx: int = 0):
        """
        Perform one Metropolis Monte Carlo step
        
        Args:
            cell_idx: Cell to update
        """
        p = self.params
        
        # Random gene to flip
        gene_idx = np.random.randint(p.n_genes)
        
        # Current spin
        spin_i = self.spins[cell_idx, gene_idx]
        
        # Energy change if flipped
        h_local = self.local_field(cell_idx, gene_idx)
        delta_e = 2 * spin_i * h_local
        
        # Metropolis acceptance
        if delta_e < 0 or np.random.rand() < np.exp(-p.beta() * delta_e):
            self.spins[cell_idx, gene_idx] *= -1
    
    def glauber_dynamics(self, cell_idx: int, gene_idx: int, dt: float):
        """
        Glauber dynamics (continuous time)
        
        Flip rate: Γ = Γ_0 / (1 + exp(2βh_local σ_i))
        
        Args:
            cell_idx: Cell index
            gene_idx: Gene index
            dt: Time step (s)
        """
        p = self.params
        
        spin_i = self.spins[cell_idx, gene_idx]
        h_local = self.local_field(cell_idx, gene_idx)
        
        # Flip probability
        flip_prob = p.flip_rate * dt / (1.0 + np.exp(2 * p.beta() * h_local * spin_i))
        
        if np.random.rand() < flip_prob:
            self.spins[cell_idx, gene_idx] *= -1
    
    def simulate(self,
                 duration: float = 1.0,
                 dt: float = 1e-3,
                 method: str = 'glauber') -> Dict:
        """
        Simulate epigenetic dynamics
        
        Args:
            duration: Simulation duration (s)
            dt: Time step (s)
            method: 'metropolis' or 'glauber'
            
        Returns:
            Dictionary with simulation history
        """
        p = self.params
        
        n_steps = int(duration / dt)
        
        # History
        history = {
            'time': np.arange(0, duration, dt),
            'magnetization': np.zeros((n_steps, p.n_cells)),
            'energy': np.zeros((n_steps, p.n_cells)),
            'spins_snapshots': []
        }
        
        # Run simulation
        for step in range(n_steps):
            # Update all cells
            for cell_idx in range(p.n_cells):
                if method == 'metropolis':
                    # Multiple Metropolis steps per timestep
                    for _ in range(p.n_genes):
                        self.metropolis_step(cell_idx)
                elif method == 'glauber':
                    # Glauber dynamics
                    for gene_idx in range(p.n_genes):
                        self.glauber_dynamics(cell_idx, gene_idx, dt)
            
            # Record
            for cell_idx in range(p.n_cells):
                history['magnetization'][step, cell_idx] = np.mean(self.spins[cell_idx])
                history['energy'][step, cell_idx] = self.energy(cell_idx)
            
            # Save snapshots periodically
            if step % (n_steps // 10) == 0:
                history['spins_snapshots'].append(self.spins.copy())
        
        return history
    
    def phase_transition_scan(self, t_range: Tuple[float, float] = (250, 350), n_temps: int = 20) -> Dict:
        """
        Scan magnetization vs temperature to detect phase transition
        
        Args:
            t_range: Temperature range (K)
            n_temps: Number of temperature points
            
        Returns:
            Dictionary with T, magnetization, susceptibility
        """
        p = self.params
        
        temps = np.linspace(t_range[0], t_range[1], n_temps)
        magnetizations = np.zeros(n_temps)
        susceptibilities = np.zeros(n_temps)
        
        for i, temp in enumerate(temps):
            # Set temperature
            p.temperature = temp
            
            # Thermalize
            self._init_state()
            for _ in range(1000):  # Thermalization steps
                for cell_idx in range(min(10, p.n_cells)):  # Sample subset
                    self.metropolis_step(cell_idx)
            
            # Measure
            mag_samples = []
            for _ in range(100):  # Measurement steps
                for cell_idx in range(min(10, p.n_cells)):
                    self.metropolis_step(cell_idx)
                mag_samples.append(np.abs(np.mean(self.spins)))
            
            magnetizations[i] = np.mean(mag_samples)
            susceptibilities[i] = p.n_genes * (np.var(mag_samples) / p.kt())
        
        return {
            'temperatures': temps,
            'magnetization': magnetizations,
            'susceptibility': susceptibilities,
            't_c_estimate': temps[np.argmax(susceptibilities)]
        }
    
    def memory_capacity(self) -> int:
        """
        Estimate memory capacity
        
        Hopfield-like capacity: P_max ≈ 0.14 × N
        
        Returns:
            Maximum number of patterns
        """
        p = self.params
        capacity = int(0.14 * p.n_genes)
        return capacity
    
    def information_content(self, cell_idx: int = 0) -> float:
        """
        Compute Shannon entropy (information content)
        
        Args:
            cell_idx: Cell index
            
        Returns:
            Entropy (bits)
        """
        spins = self.spins[cell_idx]
        
        # Convert to binary (0, 1)
        binary = (spins + 1) // 2
        
        # Count patterns (simplified: histogram of local configurations)
        # Full information would be H = N bits for N independent spins
        # With correlations: H < N
        
        # Simplified: fraction of methylated sites
        p_meth = np.mean(binary)
        p_unmeth = 1 - p_meth
        
        if p_meth == 0 or p_unmeth == 0:
            return 0.0
        
        h = -p_meth * np.log2(p_meth) - p_unmeth * np.log2(p_unmeth)
        
        # Total information (bits)
        total_info = h * self.params.n_genes
        
        return total_info


if __name__ == "__main__":
    """Test epigenetic Ising model"""
    print("Epigenetic Ising Model - Test")
    print("=" * 60)
    
    # Create model
    epi = EpigeneticIsing()
    
    # Test 1: Basic simulation
    print("\nTest 1: Epigenetic dynamics simulation")
    history = epi.simulate(duration=1.0, dt=1e-2, method='glauber')
    
    mag_final = history['magnetization'][-1]
    print(f"  Initial magnetization: {history['magnetization'][0].mean():.3f}")
    print(f"  Final magnetization: {mag_final.mean():.3f}")
    
    # Test 2: Memory capacity
    print("\nTest 2: Memory capacity")
    capacity = epi.memory_capacity()
    print(f"  N_genes: {epi.params.n_genes}")
    print(f"  Estimated capacity: {capacity} patterns")
    print(f"  Capacity ratio: {capacity/epi.params.n_genes:.2%}")
    
    # Test 3: Information content
    print("\nTest 3: Information content")
    info = epi.information_content(cell_idx=0)
    print(f"  Shannon entropy: {info:.1f} bits")
    print(f"  Information per gene: {info/epi.params.n_genes:.2f} bits")
    
    # Test 4: Phase transition (simplified)
    print("\nTest 4: Phase transition detection")
    print("  (Running abbreviated scan...)")
    transition = epi.phase_transition_scan(t_range=(250, 350), n_temps=10)
    print(f"  Estimated T_c: {transition['t_c_estimate']:.1f} K")
    print(f"  Body temperature: {310} K")
    
    print("\n" + "=" * 60)
