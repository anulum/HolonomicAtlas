"""
Bioelectric Field Dynamics
===========================

Implements bioelectric morphogenetic field equations and pattern formation.

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class BioelectricParameters:
    """Parameters for bioelectric field simulation"""
    
    # Domain
    domain_size: float = 1e-3  # m (1 mm)
    n_points: int = 100
    
    # Field equation parameters
    lambda_screen: float = 1e-4  # Screening length (m)
    epsilon: float = 80.0  # Relative permittivity
    
    # Gap junction coupling
    g_gap: float = 1e-9  # Gap junction conductance (S)
    
    # Ion channel parameters
    g_na: float = 1e-8  # Sodium conductance (S)
    g_k: float = 3e-9  # Potassium conductance (S)
    g_cl: float = 1e-9  # Chloride conductance (S)
    
    e_na: float = 60e-3  # Sodium reversal (V)
    e_k: float = -90e-3  # Potassium reversal (V)
    e_cl: float = -60e-3  # Chloride reversal (V)
    
    # Pump parameters
    i_pump: float = -1e-12  # Pump current (A)
    
    # Membrane capacitance
    c_m: float = 1e-11  # F
    
    # Pattern parameters
    pattern_type: str = 'gradient'  # 'gradient', 'oscillating', 'domain'
    

class BioelectricField:
    """
    Bioelectric Field Simulator
    
    Solves:
    ∇²V - (1/λ²)V = -ρ/ε + I_source
    """
    
    def __init__(self, params: Optional[BioelectricParameters] = None):
        self.params = params or BioelectricParameters()
        self._build_operators()
    
    def _build_operators(self):
        """Build finite difference operators"""
        p = self.params
        
        # Spatial grid
        self.x = np.linspace(0, p.domain_size, p.n_points)
        self.dx = self.x[1] - self.x[0]
        
        # Laplacian (tridiagonal)
        diag_main = -2 * np.ones(p.n_points) / self.dx**2
        diag_off = np.ones(p.n_points-1) / self.dx**2
        
        # Screening term
        diag_main -= 1.0 / p.lambda_screen**2
        
        # Boundary conditions: Neumann (zero flux)
        self.laplacian = diags(
            [diag_off, diag_main, diag_off],
            [-1, 0, 1],
            shape=(p.n_points, p.n_points)
        ).tocsr()
    
    def solve_poisson(self, 
                     rho: np.ndarray,
                     i_source: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve Poisson equation for voltage
        
        ∇²V - (1/λ²)V = -ρ/ε + I_source
        
        Args:
            rho: Charge density (C/m³)
            i_source: Current source density (A/m³)
            
        Returns:
            V: Voltage field (V)
        """
        p = self.params
        
        # Right-hand side
        rhs = -rho / (8.854e-12 * p.epsilon)
        
        if i_source is not None:
            rhs += i_source
        
        # Solve sparse system
        v = spsolve(self.laplacian, rhs)
        
        return v
    
    def create_target_pattern(self, pattern_type: Optional[str] = None) -> np.ndarray:
        """
        Create V_target morphogenetic pattern
        
        Args:
            pattern_type: 'gradient', 'oscillating', 'domain', or 'custom'
            
        Returns:
            V_target: Target voltage pattern (V)
        """
        p = self.params
        pattern = pattern_type or p.pattern_type
        
        if pattern == 'gradient':
            # Linear anterior-posterior gradient
            v_target = -70e-3 + 30e-3 * (self.x / p.domain_size)
        
        elif pattern == 'oscillating':
            # Periodic pattern for segmentation
            v_target = -50e-3 + 20e-3 * np.sin(10 * 2*np.pi * self.x / p.domain_size)
        
        elif pattern == 'domain':
            # Step function for boundary
            v_target = np.where(self.x < p.domain_size/2, -70e-3, -40e-3)
        
        elif pattern == 'head_tail':
            # Specific head-tail patterning
            # Head: -50 mV, Tail: -20 mV
            sigmoid = 1.0 / (1.0 + np.exp(-20*(self.x/p.domain_size - 0.5)))
            v_target = -50e-3 + 30e-3 * sigmoid
        
        else:
            v_target = np.ones(p.n_points) * (-70e-3)
        
        return v_target
    
    def ode_system(self, 
                   t: float,
                   v: np.ndarray,
                   v_target: np.ndarray) -> np.ndarray:
        """
        Bioelectric field dynamics ODE
        
        dV/dt = -(I_ion + I_gap + I_pump) / C_m + feedback(V_target - V)
        
        Args:
            t: Time (s)
            v: Current voltage (V)
            v_target: Target pattern (V)
            
        Returns:
            dV/dt
        """
        p = self.params
        
        # Ion currents (Goldman-Hodgkin-Katz-like)
        i_na = p.g_na * (v - p.e_na)
        i_k = p.g_k * (v - p.e_k)
        i_cl = p.g_cl * (v - p.e_cl)
        i_ion = i_na + i_k + i_cl
        
        # Gap junction coupling (diffusion)
        i_gap = np.zeros_like(v)
        i_gap[1:-1] = p.g_gap * (v[2:] - 2*v[1:-1] + v[:-2]) / self.dx**2
        
        # Feedback toward target
        feedback_strength = 0.1  # 1/s
        feedback = feedback_strength * (v_target - v)
        
        # Total dynamics
        dv_dt = -(i_ion + i_gap + p.i_pump) / p.c_m + feedback
        
        return dv_dt
    
    def simulate(self,
                 duration: float = 1.0,
                 dt: float = 1e-4,
                 v_init: Optional[np.ndarray] = None,
                 pattern: str = 'gradient') -> Dict:
        """
        Simulate bioelectric field evolution
        
        Args:
            duration: Simulation duration (s)
            dt: Time step (s)
            v_init: Initial voltage, defaults to -70 mV
            pattern: Target pattern type
            
        Returns:
            Dictionary with simulation results
        """
        p = self.params
        
        # Initial condition
        if v_init is None:
            v_init = np.ones(p.n_points) * (-70e-3)
        
        # Target pattern
        v_target = self.create_target_pattern(pattern)
        
        # Time points
        t_span = (0, duration)
        t_eval = np.arange(0, duration, dt)
        
        # Solve ODE
        solution = solve_ivp(
            fun=lambda t, y: self.ode_system(t, y, v_target),
            t_span=t_span,
            y0=v_init,
            t_eval=t_eval,
            method='LSODA'
        )
        
        return {
            'time': solution.t,
            'voltage': solution.y,
            'v_target': v_target,
            'x': self.x,
            'success': solution.success
        }
    
    def compute_electric_field(self, v: np.ndarray) -> np.ndarray:
        """
        Compute electric field E = -∇V
        
        Args:
            v: Voltage (V)
            
        Returns:
            E: Electric field (V/m)
        """
        e_field = -np.gradient(v, self.dx)
        return e_field
    
    def pattern_memory_energy(self, v: np.ndarray, v_target: np.ndarray) -> float:
        """
        Compute Hopfield-like pattern memory energy
        
        E = -½ Σ J_ij V_i V_j - Σ h_i V_i
        
        Simplified: E = -½ |V - V_target|²
        
        Args:
            v: Current voltage
            v_target: Target pattern
            
        Returns:
            Energy
        """
        energy = -0.5 * np.sum((v - v_target)**2)
        return energy


def voltage_to_morphogen(v_field: np.ndarray,
                        e_field: np.ndarray,
                        dt: float = 1e-4,
                        diffusion: float = 1e-7,
                        mu_drift: float = 1e-8) -> np.ndarray:
    """
    Voltage-to-Morphogen (V2M) transduction
    
    ∂φ/∂t = D∇²φ - ∇·(μEφ) + source
    
    Args:
        v_field: Voltage field (V)
        e_field: Electric field (V/m)
        dt: Time step (s)
        diffusion: Diffusion coefficient (m²/s)
        mu_drift: Electrophoretic mobility (m²/V·s)
        
    Returns:
        dφ/dt: Morphogen concentration change
    """
    # Diffusion term
    laplacian = np.gradient(np.gradient(v_field))
    diffusion_term = diffusion * laplacian
    
    # Advection term (electrophoresis)
    # Simplified 1D: -∇·(μEφ) ≈ -μ(E·∇φ + φ·∇E)
    grad_morphogen = np.gradient(v_field)  # Placeholder: should be φ
    advection_term = -mu_drift * (e_field * grad_morphogen)
    
    # Total change
    dphi_dt = diffusion_term + advection_term
    
    return dphi_dt


if __name__ == "__main__":
    """Test bioelectric field simulation"""
    print("Bioelectric Field Simulator - Test")
    print("=" * 60)
    
    # Create simulator
    bio = BioelectricField()
    
    # Test 1: Pattern formation
    print("\nTest 1: Gradient pattern formation")
    results = bio.simulate(duration=1.0, pattern='gradient')
    
    v_final = results['voltage'][:, -1]
    v_target = results['v_target']
    
    print(f"  Target range: [{v_target.min()*1e3:.1f}, {v_target.max()*1e3:.1f}] mV")
    print(f"  Final range: [{v_final.min()*1e3:.1f}, {v_final.max()*1e3:.1f}] mV")
    print(f"  Pattern error: {np.sqrt(np.mean((v_final - v_target)**2))*1e3:.2f} mV")
    
    # Test 2: Different patterns
    print("\nTest 2: Multiple pattern types")
    patterns = ['gradient', 'oscillating', 'domain', 'head_tail']
    
    for pattern in patterns:
        v_target = bio.create_target_pattern(pattern)
        print(f"  {pattern}: range [{v_target.min()*1e3:.1f}, {v_target.max()*1e3:.1f}] mV")
    
    # Test 3: Electric field
    print("\nTest 3: Electric field calculation")
    e_field = bio.compute_electric_field(v_final)
    print(f"  E-field range: [{e_field.min():.2f}, {e_field.max():.2f}] V/m")
    print(f"  Expected range: 1-10 mV/mm = 1-10 V/m")
    
    print("\n" + "=" * 60)
