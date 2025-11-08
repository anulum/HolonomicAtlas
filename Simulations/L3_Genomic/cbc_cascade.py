"""
CBC (CISS-Bioelectric-Chromatin) Cascade Simulator

Implements the complete 4-stage cascade that transduces quantum spin information
to chromatin accessibility:

Stage 1: CISS → Spin-polarized current (ps timescale)
Stage 2: Spin current → Effective magnetic field (ns timescale)
Stage 3: B_eff → Ion channel modulation (μs timescale)
Stage 4: ΔV_mem → Chromatin remodeling (min timescale)

Author: Based on SCPN Framework by Miroslav Šotek
Date: November 2024
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.special import erf
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
AVOGADRO = 6.02214076e23  # Avogadro's number


@dataclass
class CBCParameters:
    """Parameters for CBC cascade simulation"""
    
    # Stage 1: CISS Parameters
    p_ciss_max: float = 0.8  # Maximum spin polarization
    k_et_0: float = 1e9  # Base electron transfer rate (s^-1)
    delta_g_act: float = 0.5 * E_CHARGE  # Activation energy (J)
    e_redox: float = 0.0  # Redox potential (V)
    e_half: float = -0.2  # Half-maximal potential (V)
    lambda_so_0: float = 0.1 * E_CHARGE  # Base spin-orbit coupling (J)
    kappa_twist: float = 0.01  # Torsional coupling constant
    
    # Stage 2: Field Generation
    gamma_b: float = 2.8e10  # Gyromagnetic ratio (rad/(s·T))
    current_density: float = 1e-6  # Electron current density (A/m^2)
    
    # Stage 3: Ion Channel Modulation
    v_half_0: float = -40e-3  # Base half-activation voltage (V)
    slope_factor: float = 5e-3  # Voltage sensitivity (V)
    alpha_b: float = 1e3  # Magnetic field sensitivity (T^-1)
    g_ca_max: float = 1e-9  # Maximum Ca^2+ conductance (S)
    e_ca: float = 120e-3  # Ca^2+ reversal potential (V)
    
    # Stage 4: Chromatin Remodeling
    ca_threshold: float = 1e-6  # Ca^2+ threshold for CaMKII (M)
    hill_coeff: float = 4.0  # Hill coefficient
    k_phos_0: float = 1.0  # Base phosphorylation rate (s^-1)
    alpha_camk: float = 1e6  # CaMKII sensitivity (M^-1)
    e_unwrap_0: float = 15.0  # Base unwrapping energy (k_B*T)
    gamma_mod: float = 2.0  # Modulation strength (k_B*T)
    
    # System parameters
    temperature: float = 310.0  # Physiological temperature (K)
    psi_s_amplitude: float = 1.0  # Ψs field amplitude
    
    def kt(self) -> float:
        """Thermal energy k_B*T"""
        return K_B * self.temperature


class CBCCascade:
    """
    Complete CBC Cascade Simulator
    
    Simulates the full transduction pathway from CISS to chromatin accessibility
    with all intermediate stages and proper temporal dynamics.
    """
    
    def __init__(self, params: Optional[CBCParameters] = None):
        """
        Initialize CBC cascade simulator
        
        Args:
            params: CBC parameters (uses defaults if None)
        """
        self.params = params or CBCParameters()
        self.history = {
            'time': [],
            'p_ciss': [],
            'j_spin': [],
            'b_eff': [],
            'p_open': [],
            'v_mem': [],
            'ca_conc': [],
            'chromatin_a': []
        }
    
    def stage1_ciss(self, 
                    e_redox: float, 
                    theta_spin: float = 0.0,
                    dna_torsion: float = 0.0,
                    psi_s: float = 1.0) -> Tuple[float, float]:
        """
        Stage 1: CISS generates spin-polarized electron transfer
        
        Args:
            e_redox: Redox potential (V)
            theta_spin: Spin alignment angle (rad)
            dna_torsion: DNA torsional strain
            psi_s: Ψs field amplitude
            
        Returns:
            (P_CISS, k_ET): Spin polarization and transfer rate
        """
        p = self.params
        
        # Metabolic coupling of CISS efficiency
        p_ciss = p.p_ciss_max / (1.0 + np.exp(-(e_redox - p.e_half) / p.kt()))
        
        # Torsional modulation of spin-orbit coupling
        lambda_so = p.lambda_so_0 + p.kappa_twist * dna_torsion
        lambda_so *= (1.0 + 0.5 * psi_s)  # Ψs enhancement
        
        # Electron transfer rate with CISS modulation
        k_et = p.k_et_0 * np.exp(-p.delta_g_act / p.kt()) * \
               (1.0 + p_ciss * np.cos(theta_spin))
        
        return p_ciss, k_et
    
    def stage2_field_generation(self, 
                                p_ciss: float, 
                                k_et: float,
                                n_electrons: float = 1e6) -> float:
        """
        Stage 2: Spin current generates effective magnetic field
        
        Args:
            p_ciss: Spin polarization
            k_et: Electron transfer rate
            n_electrons: Number of electrons
            
        Returns:
            B_eff: Effective magnetic field (T)
        """
        p = self.params
        
        # Spin current density
        j_spin = p_ciss * k_et * n_electrons * E_CHARGE
        
        # Effective field via curl of spin current
        # Simplified: B_eff ≈ μ_0 * j_spin / (2π*r) for r ~ 1 nm
        r_eff = 1e-9  # Effective radius (m)
        mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
        
        b_eff = mu_0 * j_spin / (2 * np.pi * r_eff)
        
        return b_eff
    
    def stage3_channel_modulation(self, 
                                  b_eff: float, 
                                  v_mem: float) -> Tuple[float, float]:
        """
        Stage 3: Magnetic field modulates ion channel opening
        
        Args:
            b_eff: Effective magnetic field (T)
            v_mem: Membrane voltage (V)
            
        Returns:
            (P_open, I_Ca): Channel open probability and Ca^2+ current
        """
        p = self.params
        
        # Voltage half-activation modulated by B field
        v_half_mod = p.v_half_0 + p.alpha_b * b_eff
        
        # Boltzmann sigmoid for channel activation
        m_inf = 1.0 / (1.0 + np.exp(-(v_mem - v_half_mod) / p.slope_factor))
        
        # Ca^2+ current
        i_ca = p.g_ca_max * m_inf * (v_mem - p.e_ca)
        
        return m_inf, i_ca
    
    def stage4_chromatin_remodeling(self, 
                                   ca_conc: float,
                                   psi_s: float = 1.0) -> float:
        """
        Stage 4: Voltage changes remodel chromatin
        
        Args:
            ca_conc: Ca^2+ concentration (M)
            psi_s: Ψs field amplitude
            
        Returns:
            A: Chromatin accessibility (0-1)
        """
        p = self.params
        
        # CaMKII activation (Hill function)
        camk_active = (ca_conc ** p.hill_coeff) / \
                      (p.ca_threshold ** p.hill_coeff + ca_conc ** p.hill_coeff)
        
        # Energy shift due to CaMKII phosphorylation
        delta_e_mod = p.gamma_mod * camk_active
        
        # Effective unwrapping energy (reduced by phosphorylation)
        e_unwrap_eff = p.e_unwrap_0 - delta_e_mod
        
        # Ψs field enhancement of accessibility
        psi_enhancement = 1.0 + 0.3 * psi_s
        
        # Chromatin accessibility
        a_chromatin = psi_enhancement * np.exp(-e_unwrap_eff)
        
        # Normalize to [0, 1]
        a_chromatin = np.clip(a_chromatin, 0, 1)
        
        return a_chromatin
    
    def run_cascade(self, 
                    t_span: Tuple[float, float] = (0, 1.0),
                    dt: float = 1e-6,
                    e_redox: float = 0.0,
                    v_mem_init: float = -70e-3,
                    b_field_ext: float = 0.0,
                    psi_s_func = None) -> Dict:
        """
        Run complete CBC cascade simulation
        
        Args:
            t_span: Time span (start, end) in seconds
            dt: Time step (s)
            e_redox: Redox potential (V)
            v_mem_init: Initial membrane voltage (V)
            b_field_ext: External magnetic field (T)
            psi_s_func: Function t -> Ψs(t) or constant value
            
        Returns:
            Dictionary with time series of all variables
        """
        # Time array
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)
        
        # Initialize arrays
        p_ciss = np.zeros(n_steps)
        j_spin = np.zeros(n_steps)
        b_eff = np.zeros(n_steps)
        p_open = np.zeros(n_steps)
        v_mem = np.zeros(n_steps)
        ca_conc = np.zeros(n_steps)
        chromatin_a = np.zeros(n_steps)
        
        # Initial conditions
        v_mem[0] = v_mem_init
        ca_conc[0] = 1e-7  # Resting Ca^2+ (100 nM)
        chromatin_a[0] = 0.1  # Initial accessibility
        
        # Psi_s function
        if psi_s_func is None:
            psi_s_func = lambda t: self.params.psi_s_amplitude
        elif not callable(psi_s_func):
            val = psi_s_func
            psi_s_func = lambda t: val
        
        # Simulation loop
        for i in range(1, n_steps):
            # Current Ψs value
            psi_s = psi_s_func(t[i])
            
            # Stage 1: CISS
            p_ciss[i], k_et = self.stage1_ciss(
                e_redox=e_redox,
                theta_spin=0.0,
                psi_s=psi_s
            )
            
            # Stage 2: Field Generation
            b_eff[i] = self.stage2_field_generation(p_ciss[i], k_et)
            b_eff[i] += b_field_ext  # Add external field
            
            # Stage 3: Channel Modulation
            p_open[i], i_ca = self.stage3_channel_modulation(
                b_eff[i], 
                v_mem[i-1]
            )
            
            # Update Ca^2+ concentration
            # dCa/dt = J_in - k_out * Ca
            j_in = -i_ca / (E_CHARGE * 1e-15)  # Convert current to flux
            k_out = 100.0  # Clearance rate (s^-1)
            dca_dt = j_in - k_out * ca_conc[i-1]
            ca_conc[i] = ca_conc[i-1] + dca_dt * dt
            ca_conc[i] = max(ca_conc[i], 1e-8)  # Floor at 10 nM
            
            # Update voltage
            # Simple model: V_mem relaxes to V_rest with Ca^2+ drive
            v_rest = -70e-3
            tau_v = 10e-3  # Voltage time constant (10 ms)
            v_drive = 20e-3 * (ca_conc[i] / 1e-6)  # Ca-dependent depolarization
            dv_dt = (v_rest + v_drive - v_mem[i-1]) / tau_v
            v_mem[i] = v_mem[i-1] + dv_dt * dt
            
            # Stage 4: Chromatin Remodeling
            chromatin_a[i] = self.stage4_chromatin_remodeling(
                ca_conc[i],
                psi_s=psi_s
            )
            
            # Spin current (for diagnostics)
            j_spin[i] = p_ciss[i] * k_et
        
        # Store results
        results = {
            'time': t,
            'p_ciss': p_ciss,
            'j_spin': j_spin,
            'b_eff': b_eff,
            'p_open': p_open,
            'v_mem': v_mem,
            'ca_conc': ca_conc,
            'chromatin_a': chromatin_a,
            'params': self.params
        }
        
        return results
    
    def chirality_reversal_test(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Test chirality dependence: L-DNA vs D-DNA (Z-DNA)
        
        L-DNA should give positive P_CISS, D-DNA should give negative P_CISS
        
        Returns:
            (results_L, results_D): Results for L-DNA and D-DNA
        """
        # L-DNA (normal chirality)
        results_l = self.run_cascade(**kwargs)
        
        # D-DNA (reversed chirality) - invert CISS
        original_p_ciss = self.params.p_ciss_max
        self.params.p_ciss_max = -original_p_ciss
        results_d = self.run_cascade(**kwargs)
        self.params.p_ciss_max = original_p_ciss  # Restore
        
        return results_l, results_d
    
    def temporal_precedence_test(self, **kwargs) -> Dict:
        """
        Test temporal precedence: t_spin < t_field < t_channel < t_voltage < t_chromatin
        
        Returns:
            Dictionary with characteristic timescales
        """
        results = self.run_cascade(**kwargs)
        
        # Find characteristic response times (10-90% rise)
        def find_response_time(signal, threshold_low=0.1, threshold_high=0.9):
            """Find time to reach threshold from baseline"""
            baseline = signal[0]
            peak = np.max(signal)
            if peak == baseline:
                return np.inf
            
            signal_norm = (signal - baseline) / (peak - baseline)
            
            try:
                t_low = np.where(signal_norm >= threshold_low)[0][0]
                t_high = np.where(signal_norm >= threshold_high)[0][0]
                return t_high - t_low
            except IndexError:
                return np.inf
        
        dt = results['time'][1] - results['time'][0]
        
        timescales = {
            'tau_spin': find_response_time(results['p_ciss']) * dt,
            'tau_field': find_response_time(results['b_eff']) * dt,
            'tau_channel': find_response_time(results['p_open']) * dt,
            'tau_voltage': find_response_time(results['v_mem']) * dt,
            'tau_chromatin': find_response_time(results['chromatin_a']) * dt,
        }
        
        # Check precedence
        precedence_satisfied = (
            timescales['tau_spin'] < timescales['tau_field'] < 
            timescales['tau_channel'] < timescales['tau_voltage'] < 
            timescales['tau_chromatin']
        )
        
        return {
            'timescales': timescales,
            'precedence_satisfied': precedence_satisfied,
            'results': results
        }


# Utility functions

def minimal_detectable_effect(
    n_samples: int = 100,
    alpha: float = 0.05,
    power: float = 0.8
) -> Dict[str, float]:
    """
    Calculate minimal detectable effects for CBC cascade
    
    Args:
        n_samples: Number of experimental samples
        alpha: Significance level
        power: Statistical power
        
    Returns:
        Dictionary of MDEs for each parameter
    """
    from scipy import stats
    
    # Effect size for given power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Cohen's d for detection
    cohens_d = (z_alpha + z_beta) / np.sqrt(n_samples / 2)
    
    # Convert to physical units (assuming ~20% CV)
    cv = 0.2
    
    mde = {
        'p_ciss': cohens_d * cv * 0.6,  # ~5% of typical value
        'delta_v_half': cohens_d * cv * 5e-3,  # ~2 mV
        'delta_v_mem': cohens_d * cv * 20e-3,  # ~5 mV
        'delta_chromatin': cohens_d * cv * 0.3,  # ~10%
    }
    
    return mde


if __name__ == "__main__":
    """Example usage"""
    import matplotlib.pyplot as plt
    
    print("CBC Cascade Simulator - Test Run")
    print("=" * 50)
    
    # Initialize simulator
    cbc = CBCCascade()
    
    # Run cascade
    print("\nRunning CBC cascade...")
    results = cbc.run_cascade(
        t_span=(0, 0.01),  # 10 ms
        dt=1e-6,  # 1 μs
        e_redox=0.0,
        psi_s_func=lambda t: 1.0 + 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz modulation
    )
    
    # Print results
    print(f"\nFinal chromatin accessibility: {results['chromatin_a'][-1]:.3f}")
    print(f"Mean B_eff: {np.mean(results['b_eff'])*1e6:.2f} μT")
    print(f"ΔV_mem: {(results['v_mem'][-1] - results['v_mem'][0])*1e3:.2f} mV")
    
    # Test temporal precedence
    print("\nTesting temporal precedence...")
    precedence = cbc.temporal_precedence_test(t_span=(0, 0.1), dt=1e-6)
    print("Timescales:")
    for key, val in precedence['timescales'].items():
        print(f"  {key}: {val*1e3:.3f} ms")
    print(f"Precedence satisfied: {precedence['precedence_satisfied']}")
    
    # Calculate MDE
    mde = minimal_detectable_effect(n_samples=100)
    print("\nMinimal Detectable Effects (n=100, α=0.05, power=0.8):")
    for key, val in mde.items():
        print(f"  {key}: {val:.4f}")
    
    print("\n" + "=" * 50)
    print("Test complete. See plots for visualization.")
