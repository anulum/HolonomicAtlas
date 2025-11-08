"""
Klein-Gordon Field Theory for Morphogenetic Patterns
=====================================================

Implements Klein-Gordon equation for quantum field theoretic description
of morphogenetic patterns.

Based on Chapter 11 of the SCPN Layer 3 manuscript.

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KGParameters:
    """Parameters for Klein-Gordon field simulation"""
    
    # Spatial domain (3D)
    domain_size: Tuple[float, float, float] = (1e-3, 1e-3, 1e-3)  # m
    n_points: Tuple[int, int, int] = (64, 64, 64)
    
    # Field parameters
    m_squared: float = 1e8  # Effective mass squared (1/m²)
    lambda_coupling: float = 0.1  # Self-interaction strength
    
    # Damping (for relaxation)
    gamma: float = 0.1  # Damping coefficient (1/s)
    
    # External potential coupling
    V_coupling: float = 1.0  # Coupling to bioelectric potential
    
    def __post_init__(self):
        """Compute derived quantities"""
        self.dx = self.domain_size[0] / self.n_points[0]
        self.dy = self.domain_size[1] / self.n_points[1]
        self.dz = self.domain_size[2] / self.n_points[2]
        self.h = min(self.dx, self.dy, self.dz)
        
        # Compute stable timestep
        # CFL: c*dt/h < 1, assuming c ~ 1 m/s
        self.dt = 0.4 * self.h / 1.0


class KleinGordonField:
    """
    Klein-Gordon Field Simulator
    
    Solves:
    ∂²φ/∂t² - ∇²φ + m²φ + λφ³ + γ∂φ/∂t = V_ext(x,t)
    
    This is the relativistic field equation with:
    - m²: Mass term (gap in spectrum)
    - λφ³: Self-interaction (anharmonic)
    - γ∂φ/∂t: Damping
    - V_ext: External bioelectric potential coupling
    """
    
    def __init__(self, params: Optional[KGParameters] = None):
        self.params = params or KGParameters()
        self._build_operators()
    
    def _build_operators(self):
        """Build differential operators"""
        p = self.params
        
        # Spatial grids
        self.x = np.linspace(0, p.domain_size[0], p.n_points[0])
        self.y = np.linspace(0, p.domain_size[1], p.n_points[1])
        self.z = np.linspace(0, p.domain_size[2], p.n_points[2])
        
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.grid = (X, Y, Z)
        
        # Fourier space wave vectors
        kx = 2*np.pi * fftfreq(p.n_points[0], p.dx)
        ky = 2*np.pi * fftfreq(p.n_points[1], p.dy)
        kz = 2*np.pi * fftfreq(p.n_points[2], p.dz)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        self.k_squared = KX**2 + KY**2 + KZ**2
    
    def laplacian_spectral(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian using spectral method
        
        Args:
            field: 3D field
            
        Returns:
            ∇²field
        """
        # FFT
        field_k = fftn(field)
        
        # Multiply by -k²
        lap_k = -self.k_squared * field_k
        
        # IFFT
        lap = np.real(ifftn(lap_k))
        
        return lap
    
    def kg_rhs(self, 
               phi: np.ndarray,
               phi_t: np.ndarray,
               V_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Right-hand side of Klein-Gordon equation
        
        System:
        dφ/dt = φ_t
        dφ_t/dt = ∇²φ - m²φ - λφ³ - γφ_t + V_ext
        
        Args:
            phi: Field value
            phi_t: Field time derivative
            V_ext: External potential (optional)
            
        Returns:
            (dphi_dt, dphi_t_dt)
        """
        p = self.params
        
        # Laplacian
        lap_phi = self.laplacian_spectral(phi)
        
        # Klein-Gordon equation
        # ∂²φ/∂t² = ∇²φ - m²φ - λφ³ - γ∂φ/∂t + V_ext
        
        dphi_dt = phi_t
        
        dphi_t_dt = (
            lap_phi
            - p.m_squared * phi
            - p.lambda_coupling * phi**3
            - p.gamma * phi_t
        )
        
        if V_ext is not None:
            dphi_t_dt += p.V_coupling * V_ext
        
        return dphi_dt, dphi_t_dt
    
    def step_rk4(self,
                 phi: np.ndarray,
                 phi_t: np.ndarray,
                 dt: float,
                 V_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 integration step
        
        Args:
            phi: Current field
            phi_t: Current field derivative
            dt: Time step
            V_ext: External potential
            
        Returns:
            (phi_new, phi_t_new)
        """
        # k1
        k1_phi, k1_phi_t = self.kg_rhs(phi, phi_t, V_ext)
        
        # k2
        k2_phi, k2_phi_t = self.kg_rhs(
            phi + 0.5*dt*k1_phi,
            phi_t + 0.5*dt*k1_phi_t,
            V_ext
        )
        
        # k3
        k3_phi, k3_phi_t = self.kg_rhs(
            phi + 0.5*dt*k2_phi,
            phi_t + 0.5*dt*k2_phi_t,
            V_ext
        )
        
        # k4
        k4_phi, k4_phi_t = self.kg_rhs(
            phi + dt*k3_phi,
            phi_t + dt*k3_phi_t,
            V_ext
        )
        
        # Update
        phi_new = phi + (dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
        phi_t_new = phi_t + (dt/6) * (k1_phi_t + 2*k2_phi_t + 2*k3_phi_t + k4_phi_t)
        
        return phi_new, phi_t_new
    
    def energy(self, phi: np.ndarray, phi_t: np.ndarray) -> Dict[str, float]:
        """
        Compute field energy components
        
        E = ∫ [½(∂φ/∂t)² + ½(∇φ)² + ½m²φ² + ¼λφ⁴] d³x
        
        Args:
            phi: Field
            phi_t: Field time derivative
            
        Returns:
            Dictionary with energy components
        """
        p = self.params
        
        # Volume element
        dV = p.dx * p.dy * p.dz
        
        # Kinetic energy
        E_kinetic = 0.5 * np.sum(phi_t**2) * dV
        
        # Gradient energy
        grad_phi = np.gradient(phi)
        E_gradient = 0.5 * np.sum(
            grad_phi[0]**2 + grad_phi[1]**2 + grad_phi[2]**2
        ) * dV
        
        # Mass energy
        E_mass = 0.5 * p.m_squared * np.sum(phi**2) * dV
        
        # Interaction energy
        E_interaction = 0.25 * p.lambda_coupling * np.sum(phi**4) * dV
        
        # Total
        E_total = E_kinetic + E_gradient + E_mass + E_interaction
        
        return {
            'kinetic': E_kinetic,
            'gradient': E_gradient,
            'mass': E_mass,
            'interaction': E_interaction,
            'total': E_total
        }
    
    def power_spectrum(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum P(k)
        
        Args:
            phi: Field
            
        Returns:
            (k_values, power_spectrum)
        """
        # FFT
        phi_k = fftn(phi)
        
        # Power spectrum
        P_k = np.abs(phi_k)**2
        
        # Radially average
        k_mag = np.sqrt(self.k_squared)
        k_max = k_mag.max()
        
        k_bins = np.linspace(0, k_max, 50)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        P_avg = np.zeros(len(k_centers))
        
        for i, (k_min, k_max) in enumerate(zip(k_bins[:-1], k_bins[1:])):
            mask = (k_mag >= k_min) & (k_mag < k_max)
            if np.any(mask):
                P_avg[i] = np.mean(P_k[mask])
        
        return k_centers, P_avg
    
    def estimate_m_squared(self, phi: np.ndarray) -> Dict:
        """
        Estimate m² from power spectrum
        
        For Klein-Gordon: P(k) ~ 1/(k² + m²)
        
        Args:
            phi: Field
            
        Returns:
            Dictionary with m² estimate and fit quality
        """
        k, P = self.power_spectrum(phi)
        
        # Remove k=0 (DC component)
        mask = k > 0
        k = k[mask]
        P = P[mask]
        
        # Fit: log(P) ~ -log(k² + m²)
        # => P ~ A/(k² + m²)
        
        from scipy.optimize import curve_fit
        
        def model(k, A, m_sq):
            return A / (k**2 + m_sq)
        
        try:
            popt, pcov = curve_fit(model, k, P, p0=[P[0], 1e8])
            A_fit, m_sq_fit = popt
            
            # R² metric
            P_pred = model(k, *popt)
            ss_res = np.sum((P - P_pred)**2)
            ss_tot = np.sum((P - np.mean(P))**2)
            r_squared = 1 - ss_res / ss_tot
            
            success = True
        except:
            m_sq_fit = np.nan
            r_squared = 0.0
            success = False
        
        return {
            'm_squared_estimate': m_sq_fit,
            'm_squared_true': self.params.m_squared,
            'r_squared': r_squared,
            'fit_success': success,
            'k': k,
            'P': P
        }
    
    def simulate(self,
                 duration: float = 1.0,
                 phi_init: Optional[np.ndarray] = None,
                 V_ext_profile: Optional[callable] = None,
                 record_interval: int = 10) -> Dict:
        """
        Run Klein-Gordon simulation
        
        Args:
            duration: Simulation time (s)
            phi_init: Initial field (random if None)
            V_ext_profile: Function V_ext(t, X, Y, Z)
            record_interval: Record every N steps
            
        Returns:
            Dictionary with simulation history
        """
        p = self.params
        
        # Initialize
        if phi_init is None:
            # Random perturbation
            phi = 0.01 * np.random.randn(*p.n_points)
        else:
            phi = phi_init.copy()
        
        phi_t = np.zeros_like(phi)
        
        # Time steps
        n_steps = int(duration / p.dt)
        n_records = n_steps // record_interval
        
        # Pre-allocate history
        history = {
            'time': np.zeros(n_records),
            'phi': np.zeros((n_records, *p.n_points)),
            'energy': np.zeros(n_records)
        }
        
        # Run
        record_idx = 0
        for step_idx in range(n_steps):
            t = step_idx * p.dt
            
            # External potential
            V_ext = None
            if V_ext_profile is not None:
                V_ext = V_ext_profile(t, *self.grid)
            
            # Integrate
            phi, phi_t = self.step_rk4(phi, phi_t, p.dt, V_ext)
            
            # Record
            if step_idx % record_interval == 0:
                history['time'][record_idx] = t
                history['phi'][record_idx] = phi
                history['energy'][record_idx] = self.energy(phi, phi_t)['total']
                
                record_idx += 1
                
                if step_idx % (n_steps // 10) == 0:
                    print(f"  Progress: {100*step_idx/n_steps:.0f}%")
        
        return history


if __name__ == "__main__":
    """Test Klein-Gordon solver"""
    print("Klein-Gordon Field Simulator - Test")
    print("=" * 60)
    
    # Create solver
    params = KGParameters(
        n_points=(32, 32, 32),
        m_squared=1e8
    )
    
    kg = KleinGordonField(params)
    
    print(f"\nDomain: {params.domain_size[0]*1e3:.1f} mm³")
    print(f"Grid: {params.n_points[0]}³")
    print(f"m²: {params.m_squared:.2e} 1/m²")
    print(f"dt: {params.dt*1e6:.2f} μs")
    
    # Test 1: Free evolution
    print("\nTest 1: Free field evolution")
    results = kg.simulate(duration=0.01, record_interval=5)
    
    print(f"  Initial energy: {results['energy'][0]:.6e}")
    print(f"  Final energy: {results['energy'][-1]:.6e}")
    print(f"  Energy conservation: {abs(results['energy'][-1] - results['energy'][0])/results['energy'][0]*100:.2f}%")
    
    # Test 2: Estimate m²
    print("\nTest 2: Mass parameter estimation")
    phi_final = results['phi'][-1]
    estimate = kg.estimate_m_squared(phi_final)
    
    if estimate['fit_success']:
        print(f"  True m²: {estimate['m_squared_true']:.2e}")
        print(f"  Estimated m²: {estimate['m_squared_estimate']:.2e}")
        print(f"  R²: {estimate['r_squared']:.4f}")
    else:
        print("  Fit failed")
    
    print("\n" + "=" * 60)
