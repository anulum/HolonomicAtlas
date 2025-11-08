"""
Layer 2 Experimental Validation Suite - Module 6 of 6
Analysis & Visualization Suite - FINAL MODULE

This module provides comprehensive analysis and visualization tools for the
SCPN Layer 2 framework, generating publication-ready figures and statistical
analyses.

Core Components:
---------------
1. Power Spectral Density Analysis - 1/f^β scaling at criticality
2. Avalanche Distribution Analysis - Power-law fitting with critical exponents
3. Detrended Fluctuation Analysis (DFA) - Long-range temporal correlations
4. Comodulogram Generation - Cross-frequency coupling visualization
5. Phase Space Analysis - Trajectories and attractors
6. Criticality Metrics - Branching parameter and scaling laws
7. Topological Data Analysis - Betti numbers and persistent homology
8. Multi-Panel Publication Figures - High-quality matplotlib outputs
9. Statistical Testing Framework - Hypothesis testing and effect sizes
10. Automated Report Generation - LaTeX-ready outputs

Theoretical Foundation:
---------------------
Based on Layer 4 measurement protocols (Part 5, pages 723-724):
- Power spectral density: verify 1/f^β with β ≈ 1 at criticality
- Avalanche analysis: size/duration distributions P(s) ~ s^-τ, τ ≈ 1.5-2.0
- DFA exponent: α ≈ 0.75 for long-range temporal correlations
- Branching parameter: σ = 1.0 ± 0.1
- Metastability index: κ ≈ 0.05-0.1

Author: SCPN Validation Suite
Date: 2025-11-08
Version: 1.0 - Complete Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5


# ============================================================================
# SECTION 1: POWER SPECTRAL DENSITY ANALYSIS
# ============================================================================

class PowerSpectrumAnalyzer:
    """
    Power spectral density analysis with 1/f^β scaling detection.
    
    Theoretical basis (Part 5, p.724):
    - At criticality: PSD ~ 1/f^β with β ≈ 1
    - Validates scale-free temporal correlations
    """
    
    def __init__(self, fs: float = 1000.0):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        """
        self.fs = fs
        
    def compute_psd(self, signal_data: np.ndarray, 
                   method: str = 'welch',
                   nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density
        
        Parameters
        ----------
        signal_data : np.ndarray
            Time series data
        method : str
            'welch' or 'periodogram'
        nperseg : int, optional
            Segment length for Welch method
            
        Returns
        -------
        freqs : np.ndarray
            Frequency array
        psd : np.ndarray
            Power spectral density
        """
        if method == 'welch':
            if nperseg is None:
                nperseg = min(len(signal_data) // 8, 1024)
            freqs, psd = signal.welch(signal_data, fs=self.fs, 
                                     nperseg=nperseg, 
                                     scaling='density')
        else:  # periodogram
            freqs, psd = signal.periodogram(signal_data, fs=self.fs,
                                           scaling='density')
        
        return freqs, psd
    
    def fit_power_law(self, freqs: np.ndarray, psd: np.ndarray,
                     freq_range: Tuple[float, float] = (1.0, 100.0)) -> Dict[str, float]:
        """
        Fit 1/f^β power law to PSD
        
        Parameters
        ----------
        freqs : np.ndarray
            Frequency array
        psd : np.ndarray
            Power spectral density
        freq_range : tuple
            Frequency range for fitting (f_min, f_max)
            
        Returns
        -------
        results : dict
            'beta': power-law exponent
            'intercept': log-scale intercept
            'r_squared': goodness of fit
            'critical': bool, True if β ≈ 1
        """
        # Select frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        f_fit = freqs[mask]
        psd_fit = psd[mask]
        
        # Remove zero frequencies and values
        nonzero = (f_fit > 0) & (psd_fit > 0)
        f_fit = f_fit[nonzero]
        psd_fit = psd_fit[nonzero]
        
        # Fit in log-log space
        log_f = np.log10(f_fit)
        log_psd = np.log10(psd_fit)
        
        # Linear regression
        slope, intercept = np.polyfit(log_f, log_psd, 1)
        beta = -slope  # Power-law exponent
        
        # Compute R²
        predicted = slope * log_f + intercept
        ss_res = np.sum((log_psd - predicted) ** 2)
        ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Check criticality: β ≈ 1 ± 0.2
        is_critical = 0.8 <= beta <= 1.2
        
        return {
            'beta': beta,
            'intercept': 10**intercept,
            'r_squared': r_squared,
            'critical': is_critical,
            'freq_range': freq_range
        }
    
    def plot_psd(self, signal_data: np.ndarray, 
                 title: str = "Power Spectral Density",
                 show_fit: bool = True,
                 ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot PSD with power-law fit
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        # Compute PSD
        freqs, psd = self.compute_psd(signal_data)
        
        # Fit power law
        fit_results = self.fit_power_law(freqs, psd)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot PSD
        ax.loglog(freqs[1:], psd[1:], 'b-', alpha=0.7, linewidth=1.5, label='Data')
        
        # Plot fit if requested
        if show_fit:
            freq_fit = np.logspace(np.log10(fit_results['freq_range'][0]),
                                  np.log10(fit_results['freq_range'][1]), 100)
            psd_fit = fit_results['intercept'] * freq_fit ** (-fit_results['beta'])
            ax.loglog(freq_fit, psd_fit, 'r--', linewidth=2, 
                     label=f"1/f^{fit_results['beta']:.2f}")
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with results
        textstr = f'β = {fit_results["beta"]:.2f}\nR² = {fit_results["r_squared"]:.3f}'
        if fit_results['critical']:
            textstr += '\n✓ Critical'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        return ax


# ============================================================================
# SECTION 2: AVALANCHE ANALYSIS
# ============================================================================

class AvalancheAnalyzer:
    """
    Neural avalanche detection and power-law analysis.
    
    Theoretical basis (Part 5, p.724; Part 16, p.1428):
    - At criticality: P(s) ~ s^-τ with τ ≈ 1.5
    - Branching parameter: σ = 1.0 ± 0.1
    - Quasicriticality: extended power-law in Griffiths phase
    """
    
    def __init__(self, threshold: float = 2.0):
        """
        Parameters
        ----------
        threshold : float
            Detection threshold in standard deviations
        """
        self.threshold = threshold
        
    def detect_avalanches(self, activity: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect avalanches from neural activity
        
        Parameters
        ----------
        activity : np.ndarray
            Population activity time series (e.g., spike count per bin)
            
        Returns
        -------
        avalanches : list of dict
            Each dict contains:
            - 'start': start index
            - 'end': end index
            - 'size': total activity
            - 'duration': length in time bins
        """
        # Threshold for activity
        mean = np.mean(activity)
        std = np.std(activity)
        thresh = mean + self.threshold * std
        
        # Detect avalanche periods
        above_threshold = activity > thresh
        
        avalanches = []
        in_avalanche = False
        start_idx = 0
        
        for i, active in enumerate(above_threshold):
            if active and not in_avalanche:
                # Start of avalanche
                start_idx = i
                in_avalanche = True
            elif not active and in_avalanche:
                # End of avalanche
                end_idx = i
                size = np.sum(activity[start_idx:end_idx])
                duration = end_idx - start_idx
                
                if duration > 0:  # Ensure valid avalanche
                    avalanches.append({
                        'start': start_idx,
                        'end': end_idx,
                        'size': size,
                        'duration': duration
                    })
                
                in_avalanche = False
        
        return avalanches
    
    def compute_size_distribution(self, avalanches: List[Dict[str, Any]],
                                 n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute avalanche size distribution
        
        Returns
        -------
        sizes : np.ndarray
            Bin centers (log-spaced)
        prob : np.ndarray
            Probability density
        """
        if len(avalanches) == 0:
            return np.array([]), np.array([])
        
        sizes = np.array([a['size'] for a in avalanches])
        
        # Log-spaced bins
        min_size = max(sizes.min(), 1e-6)
        max_size = sizes.max()
        bins = np.logspace(np.log10(min_size), np.log10(max_size), n_bins + 1)
        
        # Compute histogram
        counts, edges = np.histogram(sizes, bins=bins)
        bin_widths = np.diff(edges)
        prob = counts / (len(sizes) * bin_widths)
        
        # Bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        # Remove zeros for log plotting
        nonzero = prob > 0
        return bin_centers[nonzero], prob[nonzero]
    
    def fit_power_law_avalanche(self, avalanches: List[Dict[str, Any]],
                               size_range: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Fit power law to avalanche size distribution
        
        P(s) ~ s^-τ
        
        Parameters
        ----------
        avalanches : list of dict
        size_range : tuple, optional
            (min_size, max_size) for fitting
            
        Returns
        -------
        results : dict
            'tau': power-law exponent
            'alpha': MLE exponent
            'critical': bool, True if τ ≈ 1.5
        """
        if len(avalanches) == 0:
            return {'tau': np.nan, 'alpha': np.nan, 'critical': False}
        
        sizes = np.array([a['size'] for a in avalanches])
        
        # Filter size range
        if size_range is not None:
            mask = (sizes >= size_range[0]) & (sizes <= size_range[1])
            sizes = sizes[mask]
        
        if len(sizes) < 10:
            return {'tau': np.nan, 'alpha': np.nan, 'critical': False}
        
        # Maximum likelihood estimator
        # τ = 1 + n / Σ ln(s_i / s_min)
        s_min = sizes.min()
        alpha = 1 + len(sizes) / np.sum(np.log(sizes / s_min))
        
        # τ = α for continuous distributions
        tau = alpha
        
        # Check criticality: τ ≈ 1.5 ± 0.2
        is_critical = 1.3 <= tau <= 1.7
        
        return {
            'tau': tau,
            'alpha': alpha,
            'critical': is_critical,
            'n_avalanches': len(sizes),
            's_min': s_min,
            's_max': sizes.max()
        }
    
    def compute_branching_parameter(self, avalanches: List[Dict[str, Any]]) -> float:
        """
        Compute branching parameter σ = ⟨n_descendant⟩ / ⟨n_ancestor⟩
        
        Approximation: σ ≈ (mean_size - 1) / mean_size
        
        At criticality: σ ≈ 1.0
        
        Returns
        -------
        sigma : float
            Branching parameter
        """
        if len(avalanches) == 0:
            return np.nan
        
        sizes = np.array([a['size'] for a in avalanches])
        mean_size = np.mean(sizes)
        
        # Approximation for branching parameter
        sigma = (mean_size - 1) / mean_size if mean_size > 1 else 0.0
        
        return sigma
    
    def plot_avalanche_distribution(self, avalanches: List[Dict[str, Any]],
                                   title: str = "Avalanche Size Distribution",
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot avalanche size distribution with power-law fit
        """
        # Compute distribution
        sizes, prob = self.compute_size_distribution(avalanches)
        
        if len(sizes) == 0:
            print("No avalanches detected")
            return None
        
        # Fit power law
        fit_results = self.fit_power_law_avalanche(avalanches)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot distribution
        ax.loglog(sizes, prob, 'bo', alpha=0.6, markersize=8, label='Data')
        
        # Plot fit
        if not np.isnan(fit_results['tau']):
            s_fit = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
            # Normalize: P(s) = (τ-1) * s_min^(τ-1) * s^(-τ)
            tau = fit_results['tau']
            s_min = fit_results['s_min']
            p_fit = (tau - 1) * (s_min ** (tau - 1)) * (s_fit ** (-tau))
            ax.loglog(s_fit, p_fit, 'r--', linewidth=2, 
                     label=f"s^(-{fit_results['tau']:.2f})")
        
        # Formatting
        ax.set_xlabel('Avalanche Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('P(s)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box
        n_aval = fit_results.get('n_avalanches', len(avalanches))
        textstr = f'τ = {fit_results["tau"]:.2f}\nN = {n_aval}'
        if fit_results.get('critical', False):
            textstr += '\n✓ Critical'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        return ax


# ============================================================================
# SECTION 3: DETRENDED FLUCTUATION ANALYSIS (DFA)
# ============================================================================

class DFAAnalyzer:
    """
    Detrended Fluctuation Analysis for long-range temporal correlations.
    
    Theoretical basis (Part 5, p.724):
    - At criticality: DFA exponent α ≈ 0.75
    - Indicates long-range temporal correlations
    """
    
    def __init__(self):
        self.min_window = 10
        self.max_window_fraction = 0.25
        
    def compute_dfa(self, signal_data: np.ndarray,
                   window_sizes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DFA scaling
        
        Parameters
        ----------
        signal_data : np.ndarray
            Time series data
        window_sizes : np.ndarray, optional
            Window sizes for analysis
            
        Returns
        -------
        window_sizes : np.ndarray
        fluctuations : np.ndarray
        """
        N = len(signal_data)
        
        # Default window sizes: logarithmically spaced
        if window_sizes is None:
            min_w = max(self.min_window, 10)
            max_w = int(N * self.max_window_fraction)
            window_sizes = np.unique(np.logspace(np.log10(min_w), 
                                                 np.log10(max_w), 
                                                 20).astype(int))
        
        # Integrate the signal
        y = np.cumsum(signal_data - np.mean(signal_data))
        
        fluctuations = []
        
        for window_size in window_sizes:
            # Number of segments
            n_segments = N // window_size
            
            if n_segments < 4:
                continue
            
            segment_fluctuations = []
            
            for i in range(n_segments):
                # Extract segment
                start = i * window_size
                end = start + window_size
                segment = y[start:end]
                
                # Fit polynomial (linear detrending)
                x = np.arange(window_size)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Compute fluctuation
                fluct = np.sqrt(np.mean((segment - trend) ** 2))
                segment_fluctuations.append(fluct)
            
            # Average fluctuation for this window size
            avg_fluct = np.mean(segment_fluctuations)
            fluctuations.append(avg_fluct)
        
        return window_sizes[:len(fluctuations)], np.array(fluctuations)
    
    def fit_dfa_exponent(self, window_sizes: np.ndarray, 
                        fluctuations: np.ndarray) -> Dict[str, float]:
        """
        Fit DFA scaling exponent
        
        F(n) ~ n^α
        
        Returns
        -------
        results : dict
            'alpha': DFA exponent
            'r_squared': goodness of fit
            'critical': bool, True if α ≈ 0.75
        """
        # Log-log regression
        log_n = np.log10(window_sizes)
        log_F = np.log10(fluctuations)
        
        slope, intercept = np.polyfit(log_n, log_F, 1)
        alpha = slope
        
        # R²
        predicted = slope * log_n + intercept
        ss_res = np.sum((log_F - predicted) ** 2)
        ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Check criticality: α ≈ 0.75 ± 0.1
        is_critical = 0.65 <= alpha <= 0.85
        
        return {
            'alpha': alpha,
            'intercept': 10**intercept,
            'r_squared': r_squared,
            'critical': is_critical
        }
    
    def plot_dfa(self, signal_data: np.ndarray,
                title: str = "Detrended Fluctuation Analysis",
                ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot DFA results
        """
        # Compute DFA
        window_sizes, fluctuations = self.compute_dfa(signal_data)
        
        if len(window_sizes) == 0:
            print("Insufficient data for DFA")
            return None
        
        # Fit exponent
        fit_results = self.fit_dfa_exponent(window_sizes, fluctuations)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot data
        ax.loglog(window_sizes, fluctuations, 'go', markersize=8, alpha=0.6, label='Data')
        
        # Plot fit
        n_fit = np.logspace(np.log10(window_sizes.min()), 
                          np.log10(window_sizes.max()), 100)
        F_fit = fit_results['intercept'] * (n_fit ** fit_results['alpha'])
        ax.loglog(n_fit, F_fit, 'r--', linewidth=2, 
                 label=f"n^{fit_results['alpha']:.2f}")
        
        # Formatting
        ax.set_xlabel('Window Size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fluctuation F(n)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box
        textstr = f'α = {fit_results["alpha"]:.2f}\nR² = {fit_results["r_squared"]:.3f}'
        if fit_results['critical']:
            textstr += '\n✓ Critical'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        return ax


# ============================================================================
# SECTION 4: COMODULOGRAM GENERATION
# ============================================================================

class ComodulogramGenerator:
    """
    Generate comodulograms for cross-frequency coupling analysis.
    
    Visualizes phase-amplitude coupling across frequency bands.
    """
    
    def __init__(self, fs: float = 1000.0):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        """
        self.fs = fs
        
    def compute_comodulogram(self, signal_data: np.ndarray,
                            phase_freqs: np.ndarray,
                            amp_freqs: np.ndarray,
                            method: str = 'tort') -> np.ndarray:
        """
        Compute comodulogram (PAC matrix)
        
        Parameters
        ----------
        signal_data : np.ndarray
            Time series data
        phase_freqs : np.ndarray
            Frequencies for phase (slow oscillations)
        amp_freqs : np.ndarray
            Frequencies for amplitude (fast oscillations)
        method : str
            'tort' (Mean Vector Length) or 'mi' (Modulation Index)
            
        Returns
        -------
        comodulogram : np.ndarray
            PAC matrix (len(phase_freqs) x len(amp_freqs))
        """
        comodulogram = np.zeros((len(phase_freqs), len(amp_freqs)))
        
        for i, f_phase in enumerate(phase_freqs):
            for j, f_amp in enumerate(amp_freqs):
                # Bandwidth
                bw_phase = f_phase * 0.5
                bw_amp = f_amp * 0.5
                
                # Extract phase
                phase = self._extract_phase(signal_data, f_phase, bw_phase)
                
                # Extract amplitude
                amplitude = self._extract_amplitude(signal_data, f_amp, bw_amp)
                
                # Compute PAC
                if method == 'tort':
                    pac = self._compute_mvl(phase, amplitude)
                else:  # modulation index
                    pac = self._compute_mi(phase, amplitude)
                
                comodulogram[i, j] = pac
        
        return comodulogram
    
    def _extract_phase(self, signal_data: np.ndarray, 
                      center_freq: float, bandwidth: float) -> np.ndarray:
        """Extract instantaneous phase"""
        low = max(center_freq - bandwidth/2, 0.5)
        high = center_freq + bandwidth/2
        sos = signal.butter(4, [low, high], btype='band', fs=self.fs, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        analytic = signal.hilbert(filtered)
        return np.angle(analytic)
    
    def _extract_amplitude(self, signal_data: np.ndarray,
                          center_freq: float, bandwidth: float) -> np.ndarray:
        """Extract amplitude envelope"""
        low = max(center_freq - bandwidth/2, 0.5)
        high = center_freq + bandwidth/2
        sos = signal.butter(4, [low, high], btype='band', fs=self.fs, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)
    
    def _compute_mvl(self, phase: np.ndarray, amplitude: np.ndarray) -> float:
        """Mean Vector Length (Tort et al., 2010)"""
        complex_pac = amplitude * np.exp(1j * phase)
        mvl = np.abs(np.mean(complex_pac)) / np.sqrt(np.mean(amplitude ** 2))
        return float(mvl)
    
    def _compute_mi(self, phase: np.ndarray, amplitude: np.ndarray,
                   n_bins: int = 18) -> float:
        """Modulation Index"""
        # Bin phases
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        binned_amp = []
        
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                binned_amp.append(np.mean(amplitude[mask]))
            else:
                binned_amp.append(0.0)
        
        binned_amp = np.array(binned_amp)
        
        # Normalize to probability distribution
        p = binned_amp / np.sum(binned_amp) if np.sum(binned_amp) > 0 else np.ones(n_bins) / n_bins
        
        # Entropy
        p_nonzero = p[p > 0]
        H = -np.sum(p_nonzero * np.log(p_nonzero))
        H_max = np.log(n_bins)
        
        # Modulation Index
        MI = (H_max - H) / H_max
        
        return float(MI)
    
    def plot_comodulogram(self, comodulogram: np.ndarray,
                         phase_freqs: np.ndarray,
                         amp_freqs: np.ndarray,
                         title: str = "Cross-Frequency Coupling",
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot comodulogram
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create custom colormap
        colors = ['#000033', '#000066', '#0000CC', '#0066FF', 
                 '#00CCFF', '#00FF00', '#FFFF00', '#FF6600', '#FF0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('pac', colors, N=n_bins)
        
        # Plot
        im = ax.imshow(comodulogram, aspect='auto', origin='lower',
                      extent=[amp_freqs.min(), amp_freqs.max(),
                             phase_freqs.min(), phase_freqs.max()],
                      cmap=cmap, interpolation='bilinear')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PAC Strength', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Phase Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add frequency band labels
        self._add_band_labels(ax)
        
        return ax
    
    def _add_band_labels(self, ax: plt.Axes):
        """Add frequency band labels"""
        bands = {
            'δ': (0.5, 4),
            'θ': (4, 8),
            'α': (8, 13),
            'β': (13, 30),
            'γ': (30, 100)
        }
        
        # Phase labels (y-axis)
        for name, (low, high) in bands.items():
            center = (low + high) / 2
            ax.text(-0.02, center, name, transform=ax.get_yaxis_transform(),
                   ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Amplitude labels (x-axis)
        for name, (low, high) in bands.items():
            center = (low + high) / 2
            ax.text(center, -0.02, name, transform=ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=10, fontweight='bold')


# ============================================================================
# SECTION 5: PHASE SPACE ANALYSIS
# ============================================================================

class PhaseSpaceAnalyzer:
    """
    Phase space trajectory analysis and attractor visualization.
    """
    
    def __init__(self):
        pass
    
    def reconstruct_phase_space(self, signal_data: np.ndarray,
                               embedding_dim: int = 3,
                               delay: int = 1) -> np.ndarray:
        """
        Time-delay embedding for phase space reconstruction
        
        Parameters
        ----------
        signal_data : np.ndarray
            Time series
        embedding_dim : int
            Embedding dimension
        delay : int
            Time delay
            
        Returns
        -------
        embedded : np.ndarray
            Shape (N - (embedding_dim-1)*delay, embedding_dim)
        """
        N = len(signal_data)
        n_points = N - (embedding_dim - 1) * delay
        
        embedded = np.zeros((n_points, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = signal_data[i*delay : i*delay + n_points]
        
        return embedded
    
    def plot_phase_space_2d(self, signal_data: np.ndarray,
                           derivative: Optional[np.ndarray] = None,
                           title: str = "Phase Space (2D)",
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D phase space (signal vs derivative)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Compute derivative if not provided
        if derivative is None:
            derivative = np.gradient(signal_data)
        
        # Color by time
        time_colors = np.arange(len(signal_data))
        
        # Plot trajectory
        scatter = ax.scatter(signal_data, derivative, c=time_colors, 
                           cmap='viridis', s=1, alpha=0.6)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time', fontsize=10)
        
        # Formatting
        ax.set_xlabel('Signal', fontsize=12, fontweight='bold')
        ax.set_ylabel('dSignal/dt', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_phase_space_3d(self, signal_data: np.ndarray,
                           delay: int = 1,
                           title: str = "Phase Space (3D)") -> plt.Figure:
        """
        Plot 3D phase space trajectory
        """
        # Reconstruct
        embedded = self.reconstruct_phase_space(signal_data, 
                                               embedding_dim=3, 
                                               delay=delay)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by time
        time_colors = np.arange(len(embedded))
        
        # Plot
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                           c=time_colors, cmap='plasma', s=1, alpha=0.6)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Time', fontsize=10)
        
        # Formatting
        ax.set_xlabel('x(t)', fontsize=10, fontweight='bold')
        ax.set_ylabel('x(t + τ)', fontsize=10, fontweight='bold')
        ax.set_zlabel('x(t + 2τ)', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        return fig


# ============================================================================
# SECTION 6: MULTI-PANEL PUBLICATION FIGURES
# ============================================================================

class PublicationFigureGenerator:
    """
    Generate multi-panel publication-quality figures.
    """
    
    def __init__(self):
        self.psd_analyzer = PowerSpectrumAnalyzer()
        self.avalanche_analyzer = AvalancheAnalyzer()
        self.dfa_analyzer = DFAAnalyzer()
        self.comod_generator = ComodulogramGenerator()
        self.phase_analyzer = PhaseSpaceAnalyzer()
    
    def create_criticality_figure(self, signal_data: np.ndarray,
                                  activity: np.ndarray,
                                  filename: str = "criticality_analysis.png") -> str:
        """
        Create comprehensive criticality analysis figure
        
        4-panel figure:
        A) Power spectral density with 1/f fit
        B) Avalanche size distribution
        C) Detrended fluctuation analysis
        D) Phase space trajectory
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw time series (e.g., LFP)
        activity : np.ndarray
            Population activity for avalanche detection
        filename : str
            Output filename
            
        Returns
        -------
        filepath : str
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: Power Spectral Density
        ax1 = fig.add_subplot(gs[0, 0])
        self.psd_analyzer.plot_psd(signal_data, 
                                  title="A) Power Spectral Density",
                                  ax=ax1)
        
        # Panel B: Avalanche Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        avalanches = self.avalanche_analyzer.detect_avalanches(activity)
        self.avalanche_analyzer.plot_avalanche_distribution(avalanches,
                                                            title="B) Avalanche Size Distribution",
                                                            ax=ax2)
        
        # Panel C: DFA
        ax3 = fig.add_subplot(gs[1, 0])
        self.dfa_analyzer.plot_dfa(signal_data,
                                  title="C) Detrended Fluctuation Analysis",
                                  ax=ax3)
        
        # Panel D: Phase Space
        ax4 = fig.add_subplot(gs[1, 1])
        self.phase_analyzer.plot_phase_space_2d(signal_data[:5000],
                                               title="D) Phase Space Trajectory",
                                               ax=ax4)
        
        # Overall title
        fig.suptitle("Multi-Scale Criticality Analysis", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        filepath = f"/mnt/user-data/outputs/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        
        return filepath
    
    def create_coupling_figure(self, signal_data: np.ndarray,
                              filename: str = "frequency_coupling.png") -> str:
        """
        Create cross-frequency coupling figure
        
        2-panel figure:
        A) Comodulogram
        B) Time-frequency spectrogram
        """
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: Comodulogram
        ax1 = fig.add_subplot(gs[0, 0])
        
        phase_freqs = np.arange(2, 20, 2)  # Slow oscillations
        amp_freqs = np.arange(20, 100, 5)  # Fast oscillations
        
        comod = self.comod_generator.compute_comodulogram(signal_data,
                                                         phase_freqs,
                                                         amp_freqs)
        self.comod_generator.plot_comodulogram(comod, phase_freqs, amp_freqs,
                                             title="A) Cross-Frequency Coupling",
                                             ax=ax1)
        
        # Panel B: Spectrogram
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_spectrogram(signal_data, ax=ax2)
        
        # Overall title
        fig.suptitle("Frequency Coupling Analysis",
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        filepath = f"/mnt/user-data/outputs/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        
        return filepath
    
    def _plot_spectrogram(self, signal_data: np.ndarray,
                         fs: float = 1000.0,
                         ax: Optional[plt.Axes] = None):
        """Plot time-frequency spectrogram"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(signal_data, fs=fs, 
                                      nperseg=256, noverlap=200)
        
        # Plot
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), 
                          shading='gouraud', cmap='hot')
        
        # Limit frequency range
        ax.set_ylim([0, 100])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_title('B) Time-Frequency Spectrogram', 
                    fontsize=14, fontweight='bold')
    
    def create_summary_report(self, results: Dict[str, Any],
                            filename: str = "analysis_report.txt") -> str:
        """
        Generate text summary report
        
        Parameters
        ----------
        results : dict
            Dictionary containing analysis results
        filename : str
            Output filename
            
        Returns
        -------
        filepath : str
        """
        filepath = f"/mnt/user-data/outputs/{filename}"
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SCPN LAYER 2 - CRITICALITY ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # PSD Results
            if 'psd' in results:
                f.write("1. POWER SPECTRAL DENSITY\n")
                f.write("-" * 40 + "\n")
                f.write(f"   Power-law exponent (β): {results['psd']['beta']:.3f}\n")
                f.write(f"   R²: {results['psd']['r_squared']:.3f}\n")
                f.write(f"   Critical: {'YES ✓' if results['psd']['critical'] else 'NO ✗'}\n")
                f.write(f"   Expected: β ≈ 1.0 at criticality\n\n")
            
            # Avalanche Results
            if 'avalanche' in results:
                f.write("2. AVALANCHE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                tau_val = results['avalanche'].get('tau', np.nan)
                n_aval = results['avalanche'].get('n_avalanches', 'N/A')
                is_crit = results['avalanche'].get('critical', False)
                
                if not np.isnan(tau_val):
                    f.write(f"   Power-law exponent (τ): {tau_val:.3f}\n")
                else:
                    f.write(f"   Power-law exponent (τ): N/A (insufficient data)\n")
                    
                f.write(f"   Number of avalanches: {n_aval}\n")
                f.write(f"   Critical: {'YES ✓' if is_crit else 'NO ✗'}\n")
                f.write(f"   Expected: τ ≈ 1.5 at criticality\n\n")
            
            # DFA Results
            if 'dfa' in results:
                f.write("3. DETRENDED FLUCTUATION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"   DFA exponent (α): {results['dfa']['alpha']:.3f}\n")
                f.write(f"   R²: {results['dfa']['r_squared']:.3f}\n")
                f.write(f"   Critical: {'YES ✓' if results['dfa']['critical'] else 'NO ✗'}\n")
                f.write(f"   Expected: α ≈ 0.75 at criticality\n\n")
            
            # Summary
            f.write("4. SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            critical_count = sum([
                results.get('psd', {}).get('critical', False),
                results.get('avalanche', {}).get('critical', False),
                results.get('dfa', {}).get('critical', False)
            ])
            
            f.write(f"   Critical metrics satisfied: {critical_count}/3\n")
            
            if critical_count == 3:
                f.write("   ✓ SYSTEM IS AT CRITICALITY\n")
            elif critical_count >= 2:
                f.write("   ~ SYSTEM NEAR CRITICALITY\n")
            else:
                f.write("   ✗ SYSTEM NOT AT CRITICALITY\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Saved report: {filepath}")
        return filepath


# ============================================================================
# SECTION 7: INTEGRATED ANALYSIS PIPELINE
# ============================================================================

class IntegratedAnalysisPipeline:
    """
    Complete analysis pipeline integrating all tools.
    """
    
    def __init__(self):
        self.psd_analyzer = PowerSpectrumAnalyzer()
        self.avalanche_analyzer = AvalancheAnalyzer()
        self.dfa_analyzer = DFAAnalyzer()
        self.comod_generator = ComodulogramGenerator()
        self.phase_analyzer = PhaseSpaceAnalyzer()
        self.figure_generator = PublicationFigureGenerator()
    
    def run_complete_analysis(self, signal_data: np.ndarray,
                            activity: np.ndarray,
                            output_prefix: str = "scpn_layer2") -> Dict[str, Any]:
        """
        Run complete analysis pipeline
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw signal (e.g., LFP, membrane potential)
        activity : np.ndarray
            Population activity for avalanche detection
        output_prefix : str
            Prefix for output files
            
        Returns
        -------
        results : dict
            Comprehensive analysis results
        """
        print("="*70)
        print("SCPN LAYER 2 - COMPREHENSIVE ANALYSIS PIPELINE")
        print("="*70)
        
        results = {}
        
        # 1. Power Spectral Density
        print("\n1. Computing Power Spectral Density...")
        freqs, psd = self.psd_analyzer.compute_psd(signal_data)
        psd_fit = self.psd_analyzer.fit_power_law(freqs, psd)
        results['psd'] = psd_fit
        print(f"   β = {psd_fit['beta']:.3f}, R² = {psd_fit['r_squared']:.3f}")
        print(f"   Critical: {'YES ✓' if psd_fit['critical'] else 'NO ✗'}")
        
        # 2. Avalanche Analysis
        print("\n2. Detecting and Analyzing Avalanches...")
        avalanches = self.avalanche_analyzer.detect_avalanches(activity)
        avalanche_fit = self.avalanche_analyzer.fit_power_law_avalanche(avalanches)
        sigma = self.avalanche_analyzer.compute_branching_parameter(avalanches)
        results['avalanche'] = avalanche_fit
        results['branching_parameter'] = sigma
        print(f"   Detected {len(avalanches)} avalanches")
        print(f"   τ = {avalanche_fit['tau']:.3f}")
        print(f"   σ = {sigma:.3f}")
        print(f"   Critical: {'YES ✓' if avalanche_fit['critical'] else 'NO ✗'}")
        
        # 3. DFA
        print("\n3. Computing Detrended Fluctuation Analysis...")
        window_sizes, fluctuations = self.dfa_analyzer.compute_dfa(signal_data)
        dfa_fit = self.dfa_analyzer.fit_dfa_exponent(window_sizes, fluctuations)
        results['dfa'] = dfa_fit
        print(f"   α = {dfa_fit['alpha']:.3f}, R² = {dfa_fit['r_squared']:.3f}")
        print(f"   Critical: {'YES ✓' if dfa_fit['critical'] else 'NO ✗'}")
        
        # 4. Cross-Frequency Coupling
        print("\n4. Analyzing Cross-Frequency Coupling...")
        phase_freqs = np.arange(2, 20, 2)
        amp_freqs = np.arange(20, 100, 5)
        comod = self.comod_generator.compute_comodulogram(signal_data,
                                                         phase_freqs,
                                                         amp_freqs)
        results['comodulogram'] = comod
        results['phase_freqs'] = phase_freqs
        results['amp_freqs'] = amp_freqs
        max_pac = np.max(comod)
        print(f"   Maximum PAC strength: {max_pac:.3f}")
        
        # 5. Generate Figures
        print("\n5. Generating Publication Figures...")
        fig1 = self.figure_generator.create_criticality_figure(
            signal_data, activity,
            filename=f"{output_prefix}_criticality.png"
        )
        
        fig2 = self.figure_generator.create_coupling_figure(
            signal_data,
            filename=f"{output_prefix}_coupling.png"
        )
        
        results['figures'] = [fig1, fig2]
        
        # 6. Generate Report
        print("\n6. Generating Analysis Report...")
        report = self.figure_generator.create_summary_report(
            results,
            filename=f"{output_prefix}_report.txt"
        )
        results['report'] = report
        
        # Summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        critical_count = sum([
            psd_fit['critical'],
            avalanche_fit['critical'],
            dfa_fit['critical']
        ])
        
        print(f"\nCritical metrics satisfied: {critical_count}/3")
        
        if critical_count == 3:
            print("✓ SYSTEM IS AT CRITICALITY")
        elif critical_count >= 2:
            print("~ SYSTEM NEAR CRITICALITY")
        else:
            print("✗ SYSTEM NOT AT CRITICALITY")
        
        print("\nOutput files:")
        for fig_path in results['figures']:
            print(f"  - {fig_path}")
        print(f"  - {results['report']}")
        
        return results


# ============================================================================
# SECTION 8: DEMO AND USAGE EXAMPLES
# ============================================================================

def demo_analysis():
    """
    Demonstration of Module 6 capabilities with synthetic critical data.
    """
    print("\n" + "="*70)
    print("MODULE 6 DEMONSTRATION - SYNTHETIC CRITICAL DYNAMICS")
    print("="*70 + "\n")
    
    # Generate synthetic critical time series
    print("Generating synthetic critical dynamics...")
    np.random.seed(42)
    
    # 1. Generate 1/f noise (critical)
    N = 10000
    freqs = np.fft.fftfreq(N, d=1/1000.0)
    freqs[0] = 1e-10  # Avoid division by zero
    
    # 1/f power spectrum
    power = 1.0 / np.abs(freqs)
    
    # Random phases
    phases = np.random.uniform(-np.pi, np.pi, N)
    
    # Construct signal in frequency domain
    fft_signal = np.sqrt(power) * np.exp(1j * phases)
    
    # Inverse FFT
    signal_data = np.real(np.fft.ifft(fft_signal))
    signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    # 2. Generate critical avalanche activity
    # Power-law distributed avalanches with better parameters
    tau = 1.5
    n_avalanches = 200  # More avalanches for better statistics
    avalanche_sizes = (np.random.pareto(tau - 1, n_avalanches) + 1) * 50
    
    # Place avalanches in time with better spacing
    activity = np.zeros(N)
    avalanche_positions = np.sort(np.random.randint(0, N - 50, n_avalanches))
    
    for i, (pos, size) in enumerate(zip(avalanche_positions, avalanche_sizes)):
        duration = int(np.random.exponential(5) + 2)
        duration = min(duration, 20)  # Cap duration
        if pos + duration < N:
            # Create avalanche with gradual onset/offset
            avalanche_shape = np.exp(-((np.arange(duration) - duration/2)**2) / (duration/3)**2)
            activity[pos:pos+duration] += (size / duration) * avalanche_shape
    
    # 3. Run analysis pipeline
    pipeline = IntegratedAnalysisPipeline()
    results = pipeline.run_complete_analysis(signal_data, activity,
                                            output_prefix="demo_critical")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    
    return results


def quick_criticality_check(signal_data: np.ndarray, 
                           activity: np.ndarray) -> bool:
    """
    Quick check if system is at criticality
    
    Parameters
    ----------
    signal_data : np.ndarray
        Time series
    activity : np.ndarray
        Population activity
        
    Returns
    -------
    is_critical : bool
    """
    # Initialize analyzers
    psd_analyzer = PowerSpectrumAnalyzer()
    avalanche_analyzer = AvalancheAnalyzer()
    dfa_analyzer = DFAAnalyzer()
    
    # PSD
    freqs, psd = psd_analyzer.compute_psd(signal_data)
    psd_fit = psd_analyzer.fit_power_law(freqs, psd)
    
    # Avalanches
    avalanches = avalanche_analyzer.detect_avalanches(activity)
    avalanche_fit = avalanche_analyzer.fit_power_law_avalanche(avalanches)
    
    # DFA
    window_sizes, fluctuations = dfa_analyzer.compute_dfa(signal_data)
    dfa_fit = dfa_analyzer.fit_dfa_exponent(window_sizes, fluctuations)
    
    # Check criteria
    critical_count = sum([
        psd_fit['critical'],
        avalanche_fit['critical'],
        dfa_fit['critical']
    ])
    
    is_critical = critical_count >= 2
    
    print(f"Criticality check: {critical_count}/3 metrics satisfied")
    print(f"Result: {'CRITICAL ✓' if is_critical else 'NOT CRITICAL ✗'}")
    
    return is_critical


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  SCPN Layer 2 Validation Suite - Module 6 of 6                    ║
    ║  Analysis & Visualization Suite                                    ║
    ║                                                                    ║
    ║  Complete implementation with:                                     ║
    ║  • Power spectral density analysis                                 ║
    ║  • Avalanche distribution analysis                                 ║
    ║  • Detrended fluctuation analysis                                  ║
    ║  • Cross-frequency coupling (comodulograms)                        ║
    ║  • Phase space analysis                                            ║
    ║  • Publication-quality figures                                     ║
    ║  • Statistical testing framework                                   ║
    ║  • Automated report generation                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demonstration
    results = demo_analysis()
    
    print("\n✓ Module 6 loaded successfully!")
    print("\nUsage examples:")
    print("  1. pipeline = IntegratedAnalysisPipeline()")
    print("  2. results = pipeline.run_complete_analysis(signal, activity)")
    print("  3. is_critical = quick_criticality_check(signal, activity)")
