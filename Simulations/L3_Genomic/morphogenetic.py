"""
Morphogenetic Field Dynamics - PDE Solver
==========================================

Implements the V2M (Voltage-to-Morphogen) operator and morphogenetic field
equations with reaction-diffusion-advection dynamics.

Based on Chapter 10 of the SCPN Layer 3 manuscript.

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
from scipy.ndimage import laplace
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings


@dataclass
class MorphogeneticParameters:
    """Parameters for morphogenetic field simulation"""
    
    # Spatial domain
    domain_size: Tuple[float, float] = (1e-3, 1e-3)  # (Lx, Ly) in meters
    n_points: Tuple[int, int] = (100, 100)  # Grid resolution
    
    # Diffusion
    D: float = 5e-12  # Morphogen diffusion coefficient (m²/s)
    
    # Advection (electrophoresis)
    mu_e: float = 1e-8  # Electrophoretic mobility (m²/V·s)
    v0: float = 0.0  # Base advection velocity (m/s)
    v1: float = 1e-6  # Voltage coupling (m/V·s)
    
    # Reaction
    k_react: float = 1e-3  # Reaction rate (1/s)
    k_degrade: float = 1e-4  # Degradation rate (1/s)
    
    # Voltage reference
    V_ref: float = -70e-3  # Reference voltage (V)
    
    # Boundary conditions
    boundary: str = 'neumann'  # 'neumann', 'dirichlet', or 'periodic'
    
    # Stability parameters
    cfl_target: float = 0.4  # Target CFL number
    
    def __post_init__(self):
        """Compute derived quantities"""
        self.dx = self.domain_size[0] / self.n_points[0]
        self.dy = self.domain_size[1] / self.n_points[1]
        self.h = min(self.dx, self.dy)
        
        # Compute stable timestep
        self.dt_diffusion = 0.25 * self.h**2 / self.D
        self.dt_advection = self.cfl_target * self.h / (abs(self.v0) + abs(self.v1 * 0.1) + 1e-12)
        self.dt = min(self.dt_diffusion, self.dt_advection)
        
    def regime_peclet(self, v_max: float = None) -> float:
        """Compute Péclet number"""
        if v_max is None:
            v_max = abs(self.v0) + abs(self.v1 * 0.1)
        L = max(self.domain_size)
        return v_max * L / self.D
    
    def regime_damkohler(self) -> float:
        """Compute Damköhler number"""
        L = max(self.domain_size)
        return self.k_react * L**2 / self.D


class MorphogeneticPDE:
    """
    Morphogenetic Field PDE Solver
    
    Solves the Reaction-Diffusion-Advection equation:
    ∂φ/∂t = D∇²φ - ∇·(v(V)φ) + f(φ) - k_deg φ
    
    where v(V) = v₀ + v₁·V is the voltage-dependent advection velocity
    """
    
    def __init__(self, params: Optional[MorphogeneticParameters] = None):
        self.params = params or MorphogeneticParameters()
        self._build_operators()
        
    def _build_operators(self):
        """Build finite difference operators"""
        p = self.params
        nx, ny = p.n_points
        
        # Spatial grids
        self.x = np.linspace(0, p.domain_size[0], nx)
        self.y = np.linspace(0, p.domain_size[1], ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
    def laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """
        Compute 2D Laplacian using 5-point stencil
        
        Args:
            field: 2D field array
            
        Returns:
            ∇²field
        """
        p = self.params
        
        # 5-point stencil
        lap = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4.0 * field
        ) / p.h**2
        
        # Apply boundary conditions
        if p.boundary == 'neumann':
            # Zero flux at boundaries
            lap[0, :] = lap[1, :]
            lap[-1, :] = lap[-2, :]
            lap[:, 0] = lap[:, 1]
            lap[:, -1] = lap[:, -2]
        elif p.boundary == 'dirichlet':
            # Fixed value at boundaries
            lap[0, :] = 0
            lap[-1, :] = 0
            lap[:, 0] = 0
            lap[:, -1] = 0
        # Periodic is handled by np.roll automatically
        
        return lap
    
    def velocity_field(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity field from voltage
        
        v(V) = v₀ + v₁·(V - V_ref)
        
        Args:
            V: Voltage field (V)
            
        Returns:
            (vx, vy): Velocity components (m/s)
        """
        p = self.params
        
        # Electric field E = -∇V
        Ex = -np.gradient(V, p.dx, axis=0)
        Ey = -np.gradient(V, p.dy, axis=1)
        
        # Velocity from electrophoresis
        vx = p.v0 + p.mu_e * Ex
        vy = p.v0 + p.mu_e * Ey
        
        return vx, vy
    
    def upwind_divergence(self, 
                         phi: np.ndarray, 
                         vx: np.ndarray, 
                         vy: np.ndarray) -> np.ndarray:
        """
        Upwind scheme for advection term ∇·(vφ)
        
        Uses Godunov upwinding for stability
        
        Args:
            phi: Morphogen field
            vx, vy: Velocity components
            
        Returns:
            ∇·(vφ)
        """
        p = self.params
        
        # X-direction
        flux_x = np.zeros_like(phi)
        # Upwind for vx > 0
        mask_pos = vx > 0
        flux_x[mask_pos] = vx[mask_pos] * phi[mask_pos]
        # Upwind for vx < 0
        mask_neg = vx < 0
        flux_x[mask_neg] = vx[mask_neg] * np.roll(phi, -1, axis=0)[mask_neg]
        
        div_x = np.gradient(flux_x, p.dx, axis=0)
        
        # Y-direction
        flux_y = np.zeros_like(phi)
        mask_pos = vy > 0
        flux_y[mask_pos] = vy[mask_pos] * phi[mask_pos]
        mask_neg = vy < 0
        flux_y[mask_neg] = vy[mask_neg] * np.roll(phi, -1, axis=1)[mask_neg]
        
        div_y = np.gradient(flux_y, p.dy, axis=1)
        
        return div_x + div_y
    
    def reaction_term(self, phi: np.ndarray) -> np.ndarray:
        """
        Reaction term f(φ)
        
        Simple Hill-like production with saturation
        
        Args:
            phi: Morphogen concentration
            
        Returns:
            f(φ): Reaction rate
        """
        p = self.params
        
        # Hill-like: f(φ) = k * φⁿ / (Kⁿ + φⁿ)
        n = 2
        K = 0.5
        
        production = p.k_react * phi**n / (K**n + phi**n)
        degradation = p.k_degrade * phi
        
        return production - degradation
    
    def step(self, 
             phi: np.ndarray,
             V: np.ndarray,
             dt: Optional[float] = None) -> np.ndarray:
        """
        Perform one integration step
        
        Args:
            phi: Current morphogen field
            V: Voltage field
            dt: Time step (uses params.dt if None)
            
        Returns:
            phi_new: Updated morphogen field
        """
        p = self.params
        if dt is None:
            dt = p.dt
        
        # Check CFL condition
        vx, vy = self.velocity_field(V)
        v_max = max(np.abs(vx).max(), np.abs(vy).max())
        cfl = v_max * dt / p.h
        
        if cfl > 0.5:
            warnings.warn(f"CFL number {cfl:.3f} exceeds safe limit 0.5")
        
        # Compute terms
        diffusion = p.D * self.laplacian_2d(phi)
        advection = -self.upwind_divergence(phi, vx, vy)
        reaction = self.reaction_term(phi)
        
        # Forward Euler step
        dphi_dt = diffusion + advection + reaction
        phi_new = phi + dt * dphi_dt
        
        # Enforce positivity
        phi_new = np.maximum(phi_new, 0.0)
        
        return phi_new
    
    def simulate(self,
                 duration: float = 1.0,
                 V_field: Optional[np.ndarray] = None,
                 V_profile: Optional[Callable] = None,
                 phi_init: Optional[np.ndarray] = None,
                 record_interval: int = 10) -> Dict:
        """
        Run morphogenetic field simulation
        
        Args:
            duration: Simulation time (s)
            V_field: Static voltage field (V), or
            V_profile: Function V_profile(t, x, y) returning voltage
            phi_init: Initial morphogen distribution
            record_interval: Record every N steps
            
        Returns:
            Dictionary with simulation history
        """
        p = self.params
        
        # Initialize morphogen field
        if phi_init is None:
            # Default: small random perturbation
            phi = 0.1 + 0.01 * np.random.rand(*p.n_points)
        else:
            phi = phi_init.copy()
        
        # Initialize voltage field
        if V_field is None and V_profile is None:
            # Default: uniform resting potential
            V = np.ones(p.n_points) * p.V_ref
        elif V_field is not None:
            V = V_field.copy()
        
        # Time steps
        n_steps = int(duration / p.dt)
        n_records = n_steps // record_interval
        
        # Pre-allocate history
        history = {
            'time': np.zeros(n_records),
            'phi': np.zeros((n_records, *p.n_points)),
            'V': np.zeros((n_records, *p.n_points)),
            'mass': np.zeros(n_records),
            'center_of_mass': np.zeros((n_records, 2))
        }
        
        # Run simulation
        record_idx = 0
        for step_idx in range(n_steps):
            t = step_idx * p.dt
            
            # Update voltage if dynamic
            if V_profile is not None:
                V = V_profile(t, self.X, self.Y)
            
            # Integrate one step
            phi = self.step(phi, V)
            
            # Record
            if step_idx % record_interval == 0:
                history['time'][record_idx] = t
                history['phi'][record_idx] = phi
                history['V'][record_idx] = V
                history['mass'][record_idx] = np.sum(phi) * p.h**2
                
                # Center of mass
                com_x = np.sum(self.X * phi) / np.sum(phi)
                com_y = np.sum(self.Y * phi) / np.sum(phi)
                history['center_of_mass'][record_idx] = [com_x, com_y]
                
                record_idx += 1
                
                # Progress
                if step_idx % (n_steps // 10) == 0:
                    print(f"  Progress: {100*step_idx/n_steps:.0f}%")
        
        return history
    
    def steady_state_solver(self,
                           V_field: np.ndarray,
                           tolerance: float = 1e-6,
                           max_iter: int = 10000) -> np.ndarray:
        """
        Solve for steady-state morphogen distribution
        
        D∇²φ - ∇·(v(V)φ) + f(φ) = 0
        
        Args:
            V_field: Voltage field (V)
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            phi_steady: Steady-state morphogen field
        """
        p = self.params
        
        # Initial guess
        phi = 0.5 * np.ones(p.n_points)
        
        # Pseudo-time iteration
        dt_pseudo = 0.1 * p.dt
        
        for iteration in range(max_iter):
            phi_old = phi.copy()
            
            # Step
            phi = self.step(phi, V_field, dt=dt_pseudo)
            
            # Check convergence
            change = np.max(np.abs(phi - phi_old))
            if change < tolerance:
                print(f"Converged in {iteration} iterations")
                break
        else:
            warnings.warn(f"Did not converge after {max_iter} iterations")
        
        return phi


def create_gradient_voltage(params: MorphogeneticParameters,
                           V_min: float = -70e-3,
                           V_max: float = -40e-3,
                           direction: str = 'x') -> np.ndarray:
    """
    Create linear voltage gradient
    
    Args:
        params: Morphogenetic parameters
        V_min, V_max: Voltage range (V)
        direction: 'x' or 'y'
        
    Returns:
        V: Voltage field
    """
    if direction == 'x':
        x_norm = np.linspace(0, 1, params.n_points[0])
        y_norm = np.ones(params.n_points[1])
        X, Y = np.meshgrid(x_norm, y_norm, indexing='ij')
        V = V_min + (V_max - V_min) * X
    else:
        x_norm = np.ones(params.n_points[0])
        y_norm = np.linspace(0, 1, params.n_points[1])
        X, Y = np.meshgrid(x_norm, y_norm, indexing='ij')
        V = V_min + (V_max - V_min) * Y
    
    return V


def create_domain_voltage(params: MorphogeneticParameters,
                         V_left: float = -70e-3,
                         V_right: float = -40e-3,
                         boundary_pos: float = 0.5) -> np.ndarray:
    """
    Create step-function voltage (domain boundary)
    
    Args:
        params: Morphogenetic parameters
        V_left, V_right: Voltage values (V)
        boundary_pos: Boundary position (0-1)
        
    Returns:
        V: Voltage field
    """
    x = np.linspace(0, 1, params.n_points[0])
    y = np.ones(params.n_points[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    V = np.where(X < boundary_pos, V_left, V_right)
    
    return V


if __name__ == "__main__":
    """Test morphogenetic PDE solver"""
    print("Morphogenetic PDE Solver - Test")
    print("=" * 60)
    
    # Create solver
    params = MorphogeneticParameters(
        n_points=(50, 50),
        domain_size=(1e-3, 1e-3)
    )
    
    morph = MorphogeneticPDE(params)
    
    print(f"\nDomain: {params.domain_size[0]*1e3:.1f} mm × {params.domain_size[1]*1e3:.1f} mm")
    print(f"Grid: {params.n_points[0]} × {params.n_points[1]}")
    print(f"Resolution: {params.h*1e6:.2f} μm")
    print(f"Timestep: {params.dt*1e3:.3f} ms")
    print(f"Péclet number: {params.regime_peclet():.2f}")
    print(f"Damköhler number: {params.regime_damkohler():.2f}")
    
    # Test 1: Gradient-driven morphogenesis
    print("\nTest 1: Voltage gradient driving morphogen accumulation")
    V_gradient = create_gradient_voltage(params, V_min=-70e-3, V_max=-40e-3)
    
    results = morph.simulate(
        duration=2.0,
        V_field=V_gradient,
        record_interval=20
    )
    
    print(f"\nInitial mass: {results['mass'][0]:.6f}")
    print(f"Final mass: {results['mass'][-1]:.6f}")
    print(f"Center of mass shift: {(results['center_of_mass'][-1] - results['center_of_mass'][0])*1e6:.2f} μm")
    
    # Test 2: Domain boundary
    print("\nTest 2: Sharp boundary pattern formation")
    V_domain = create_domain_voltage(params, V_left=-70e-3, V_right=-40e-3)
    
    results_domain = morph.simulate(
        duration=1.0,
        V_field=V_domain,
        record_interval=10
    )
    
    print(f"Final morphogen range: [{results_domain['phi'][-1].min():.3f}, {results_domain['phi'][-1].max():.3f}]")
    
    print("\n" + "=" * 60)
