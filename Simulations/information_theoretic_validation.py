"""
Information-Theoretic Measures and Experimental Validation
SCPN Layer 4 - Consciousness Quantification Framework
"""

import numpy as np
import scipy as sp
from scipy import stats, signal, optimize
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional
import itertools
from dataclasses import dataclass
import warnings

class InformationTheoreticMeasures:
    """
    Quantify consciousness-relevant information metrics
    """
    
    def __init__(self, n_elements: int, n_states: int = 2):
        self.n_elements = n_elements
        self.n_states = n_states
        self.epsilon = 1e-10  # Small value to avoid log(0)
        
    def entropy(self, distribution: np.ndarray) -> float:
        """
        Calculate Shannon entropy: H = -∑ p(x) log p(x)
        """
        # Normalize to probability distribution
        p = distribution / (np.sum(distribution) + self.epsilon)
        
        # Remove zero probabilities
        p = p[p > self.epsilon]
        
        return -np.sum(p * np.log2(p + self.epsilon))
    
    def mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Discretize continuous variables
        X_discrete = self._discretize(X)
        Y_discrete = self._discretize(Y)
        
        # Calculate joint distribution
        joint_hist = np.histogram2d(X_discrete, Y_discrete, bins=self.n_states)[0]
        joint_prob = joint_hist / np.sum(joint_hist)
        
        # Marginal distributions
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)
        
        # Calculate MI
        mi = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_prob[i, j] > self.epsilon:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (p_x[i] * p_y[j] + self.epsilon) + self.epsilon
                    )
        
        return max(0, mi)  # Ensure non-negative
    
    def transfer_entropy(self, X: np.ndarray, Y: np.ndarray, delay: int = 1) -> float:
        """
        Calculate transfer entropy: TE(X→Y) = ∑ p(y_{t+1}, y_t, x_t) log[p(y_{t+1}|y_t, x_t)/p(y_{t+1}|y_t)]
        Measures information flow from X to Y
        """
        if len(X) != len(Y):
            raise ValueError("X and Y must have same length")
        
        # Create time-delayed versions
        if len(X) <= delay + 1:
            return 0.0
            
        Y_future = Y[delay + 1:]
        Y_past = Y[delay:-1]
        X_past = X[delay:-1]
        
        # Discretize
        Y_future_d = self._discretize(Y_future)
        Y_past_d = self._discretize(Y_past)
        X_past_d = self._discretize(X_past)
        
        # Calculate conditional probabilities
        te = 0
        n_samples = len(Y_future_d)
        
        for i in range(n_samples):
            # p(y_{t+1}, y_t, x_t)
            joint_count = np.sum((Y_future_d == Y_future_d[i]) & 
                               (Y_past_d == Y_past_d[i]) & 
                               (X_past_d == X_past_d[i]))
            p_joint = joint_count / n_samples
            
            # p(y_{t+1} | y_t, x_t)
            condition_count = np.sum((Y_past_d == Y_past_d[i]) & 
                                    (X_past_d == X_past_d[i]))
            if condition_count > 0:
                p_cond_with_x = joint_count / condition_count
            else:
                p_cond_with_x = 0
            
            # p(y_{t+1} | y_t)
            past_count = np.sum(Y_past_d == Y_past_d[i])
            future_given_past = np.sum((Y_future_d == Y_future_d[i]) & 
                                      (Y_past_d == Y_past_d[i]))
            if past_count > 0:
                p_cond_without_x = future_given_past / past_count
            else:
                p_cond_without_x = 0
            
            if p_joint > self.epsilon and p_cond_with_x > self.epsilon and p_cond_without_x > self.epsilon:
                te += p_joint * np.log2(p_cond_with_x / p_cond_without_x)
        
        return max(0, te / n_samples)  # Normalize and ensure non-negative
    
    def integrated_information(self, system_state: np.ndarray, 
                             connectivity: np.ndarray) -> float:
        """
        Calculate Integrated Information (Φ)
        Simplified version of IIT 3.0
        Φ = min_partition[I(past; future | partition)]
        """
        n = len(system_state)
        
        if n < 2:
            return 0.0
        
        # Generate all possible bipartitions
        min_information = float('inf')
        
        for partition_size in range(1, n // 2 + 1):
            for partition in itertools.combinations(range(n), partition_size):
                partition_A = list(partition)
                partition_B = [i for i in range(n) if i not in partition_A]
                
                # Calculate effective information for this partition
                ei = self._effective_information(
                    system_state, 
                    connectivity, 
                    partition_A, 
                    partition_B
                )
                
                min_information = min(min_information, ei)
        
        # Φ is the information above the minimum partition
        whole_system_info = self._system_information(system_state, connectivity)
        phi = whole_system_info - min_information
        
        return max(0, phi)
    
    def _effective_information(self, state: np.ndarray, 
                              connectivity: np.ndarray,
                              partition_A: List[int], 
                              partition_B: List[int]) -> float:
        """Calculate effective information across partition"""
        # Extract sub-states
        state_A = state[partition_A]
        state_B = state[partition_B]
        
        # Calculate information flow between partitions
        info_A_to_B = 0
        info_B_to_A = 0
        
        # Use connectivity to weight information flow
        for i in partition_A:
            for j in partition_B:
                if connectivity[i, j] > 0:
                    info_A_to_B += connectivity[i, j] * self.entropy(state[i:i+1])
        
        for i in partition_B:
            for j in partition_A:
                if connectivity[i, j] > 0:
                    info_B_to_A += connectivity[i, j] * self.entropy(state[i:i+1])
        
        return (info_A_to_B + info_B_to_A) / 2
    
    def _system_information(self, state: np.ndarray, connectivity: np.ndarray) -> float:
        """Calculate total system information"""
        # Entropy of the whole system
        system_entropy = self.entropy(state)
        
        # Weighted by connectivity
        total_connectivity = np.sum(connectivity)
        if total_connectivity > 0:
            return system_entropy * (1 + total_connectivity / len(state))
        return system_entropy
    
    def complexity_measures(self, time_series: np.ndarray) -> Dict:
        """
        Calculate various complexity measures
        C = H(system) - ∑ᵢ H(subsystemᵢ)
        """
        measures = {}
        
        # Lempel-Ziv complexity
        measures['lempel_ziv'] = self._lempel_ziv_complexity(time_series)
        
        # Approximate entropy
        measures['approximate_entropy'] = self._approximate_entropy(time_series)
        
        # Sample entropy
        measures['sample_entropy'] = self._sample_entropy(time_series)
        
        # Permutation entropy
        measures['permutation_entropy'] = self._permutation_entropy(time_series)
        
        # Neural complexity (Tononi et al.)
        if len(time_series.shape) > 1:
            measures['neural_complexity'] = self._neural_complexity(time_series)
        
        return measures
    
    def _discretize(self, data: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
        """Discretize continuous data for information calculations"""
        if n_bins is None:
            n_bins = self.n_states
        
        # Use quantile-based discretization for better resolution
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(data, quantiles)
        bins[0] -= 1e-10  # Ensure lowest value is included
        
        return np.digitize(data, bins) - 1
    
    def _lempel_ziv_complexity(self, sequence: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity"""
        # Binarize the sequence
        binary = (sequence > np.median(sequence)).astype(int)
        
        n = len(binary)
        complexity = 0
        i = 0
        
        while i < n:
            # Find the longest substring starting at i that hasn't been seen
            j = i + 1
            while j <= n:
                substring = tuple(binary[i:j])
                # Check if substring exists in previous part
                found = False
                for k in range(i):
                    if k + len(substring) <= i:
                        if tuple(binary[k:k+len(substring)]) == substring:
                            found = True
                            break
                
                if not found and j < n:
                    j += 1
                else:
                    break
            
            complexity += 1
            i = j
        
        # Normalize by theoretical maximum
        max_complexity = n / np.log2(n) if n > 1 else 1
        return complexity / max_complexity
    
    def _approximate_entropy(self, U: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy"""
        N = len(U)
        
        def _maxdist(xi, xj, m):
            return max(abs(xi[k] - xj[k]) for k in range(m))
        
        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append(U[i:i + m])
            
            C = []
            for i in range(N - m + 1):
                template = patterns[i]
                matching = sum(1 for j in range(N - m + 1) 
                             if _maxdist(template, patterns[j], m) <= r * np.std(U))
                C.append(matching / (N - m + 1))
            
            return sum(np.log(c) for c in C if c > 0) / (N - m + 1)
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _sample_entropy(self, U: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (more robust than ApEn)"""
        N = len(U)
        
        def _maxdist(xi, xj, m):
            return max(abs(xi[k] - xj[k]) for k in range(m))
        
        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append(U[i:i + m])
            
            matches = 0
            comparisons = 0
            
            for i in range(N - m + 1):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(patterns[i], patterns[j], m) <= r * np.std(U):
                        matches += 1
                    comparisons += 1
            
            return matches / comparisons if comparisons > 0 else 0
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m1 == 0:
            return float('inf')
        return -np.log(phi_m1 / phi_m) if phi_m > 0 else 0
    
    def _permutation_entropy(self, time_series: np.ndarray, order: int = 3) -> float:
        """Calculate permutation entropy"""
        n = len(time_series)
        if n < order:
            return 0
        
        # Get all permutations of given order
        permutations = list(itertools.permutations(range(order)))
        perm_count = {perm: 0 for perm in permutations}
        
        # Count occurrence of each permutation pattern
        for i in range(n - order + 1):
            segment = time_series[i:i + order]
            sorted_indices = tuple(np.argsort(segment))
            if sorted_indices in perm_count:
                perm_count[sorted_indices] += 1
        
        # Calculate entropy
        total = sum(perm_count.values())
        if total == 0:
            return 0
        
        probs = np.array(list(perm_count.values())) / total
        probs = probs[probs > 0]
        
        return -np.sum(probs * np.log2(probs))
    
    def _neural_complexity(self, multi_channel_data: np.ndarray) -> float:
        """
        Neural complexity as defined by Tononi et al.
        C = H(system) - average(H(subsystems))
        """
        n_channels, n_samples = multi_channel_data.shape
        
        # System entropy
        system_entropy = self.entropy(multi_channel_data.flatten())
        
        # Average subsystem entropy
        subsystem_entropies = []
        for k in range(1, n_channels):
            for subset in itertools.combinations(range(n_channels), k):
                subset_data = multi_channel_data[list(subset), :].flatten()
                subsystem_entropies.append(self.entropy(subset_data))
        
        avg_subsystem_entropy = np.mean(subsystem_entropies) if subsystem_entropies else 0
        
        return system_entropy - avg_subsystem_entropy


@dataclass
class ExperimentalProtocol:
    """Data structure for experimental protocols"""
    name: str
    description: str
    parameters: Dict
    measurements: List[str]
    controls: List[str]
    expected_outcomes: Dict


class ExperimentalValidation:
    """
    Experimental validation framework for tissue synchronization theory
    """
    
    def __init__(self):
        self.protocols = []
        self.results = {}
        
    def schumann_resonance_protocol(self) -> ExperimentalProtocol:
        """
        Protocol for testing Schumann resonance coupling
        """
        protocol = ExperimentalProtocol(
            name="Schumann Resonance Entrainment",
            description="Measure EEG phase coherence at Schumann frequencies",
            parameters={
                'frequencies': [7.83, 14.3, 20.8, 27.3, 33.8],  # Hz
                'recording_duration': 3600,  # seconds
                'sampling_rate': 1000,  # Hz
                'electrode_positions': ['Fz', 'Cz', 'Pz', 'Oz'],
                'shielding_conditions': ['unshielded', 'faraday_cage', 'mu_metal']
            },
            measurements=[
                'EEG_phase_coherence',
                'power_spectral_density',
                'phase_amplitude_coupling',
                'geomagnetic_indices'
            ],
            controls=[
                'Time-matched recordings in Faraday cage',
                'Sham frequency stimulation',
                'Subject blinding to condition'
            ],
            expected_outcomes={
                'phase_coherence_increase': '>20% at 7.83 Hz',
                'correlation_with_Kp_index': 'r > 0.3',
                'shielding_effect': 'Reduced coherence in Faraday cage'
            }
        )
        
        return protocol
    
    def lunar_phase_protocol(self) -> ExperimentalProtocol:
        """
        Protocol for lunar phase effects on biological rhythms
        """
        protocol = ExperimentalProtocol(
            name="Lunar Phase Biological Coupling",
            description="Track cell division rates and metabolic rhythms across lunar cycle",
            parameters={
                'duration': 60,  # days (2 lunar cycles)
                'cell_types': ['fibroblasts', 'neural_progenitors', 'hepatocytes'],
                'measurements_per_day': 4,
                'culture_conditions': 'constant darkness, 37°C',
                'metabolic_markers': ['ATP/ADP', 'NAD+/NADH', 'glucose_uptake']
            },
            measurements=[
                'cell_division_rate',
                'circadian_gene_expression',
                'metabolic_oscillations',
                'calcium_transients'
            ],
            controls=[
                'Random phase-shifted light exposure',
                'Gravitational shielding with rotating platform',
                'Temperature and humidity variations'
            ],
            expected_outcomes={
                'division_rate_modulation': '15-25% variation with lunar phase',
                'peak_activity': 'Full moon ± 2 days',
                'metabolic_coupling': 'ATP/ADP ratio follows lunar cycle'
            }
        )
        
        return protocol
    
    def biophoton_emission_protocol(self) -> ExperimentalProtocol:
        """
        Protocol for measuring ultra-weak biophoton emissions
        """
        protocol = ExperimentalProtocol(
            name="Biophoton Emission Spectroscopy",
            description="Quantify spontaneous photon emissions from living tissue",
            parameters={
                'detector': 'Photomultiplier tube (Hamamatsu R943-02)',
                'spectral_range': [200, 800],  # nm
                'integration_time': 100,  # seconds
                'temperature': 37,  # Celsius
                'sample_types': ['neural_tissue', 'cardiac_tissue', 'liver_tissue'],
                'metabolic_states': ['resting', 'active', 'stressed', 'dying']
            },
            measurements=[
                'photon_count_rate',
                'spectral_distribution',
                'temporal_patterns',
                'correlation_with_metabolism'
            ],
            controls=[
                'Dead tissue baseline',
                'Chemical luminescence blanks',
                'Dark noise measurements',
                'Temperature-matched water'
            ],
            expected_outcomes={
                'emission_rate': '10-1000 photons/sec/cm²',
                'spectral_peaks': '420-450nm (stressed), 620-650nm (healthy)',
                'metabolic_correlation': 'r > 0.7 with ATP production',
                'coherence': 'Delayed luminescence shows coherent properties'
            }
        )
        
        return protocol
    
    def multi_scale_recording_protocol(self) -> ExperimentalProtocol:
        """
        Protocol for simultaneous multi-scale recordings
        """
        protocol = ExperimentalProtocol(
            name="Multi-Scale Synchronization Mapping",
            description="Record from subcellular to tissue level simultaneously",
            parameters={
                'recording_levels': {
                    'subcellular': 'Calcium imaging (GCaMP6)',
                    'cellular': 'Patch-clamp array',
                    'network': 'Multi-electrode array (MEA)',
                    'tissue': 'Local field potential (LFP)',
                    'whole_brain': 'EEG/MEG'
                },
                'sampling_rates': {
                    'calcium': 100,  # Hz
                    'patch': 20000,  # Hz
                    'MEA': 30000,  # Hz
                    'LFP': 1000,  # Hz
                    'EEG': 1000  # Hz
                },
                'duration': 3600,  # seconds
                'preparation': 'Organotypic slice culture'
            },
            measurements=[
                'cross_scale_coherence',
                'information_flow_direction',
                'avalanche_propagation',
                'phase_transitions'
            ],
            controls=[
                'Pharmacological decoupling (gap junction blockers)',
                'Temperature variations',
                'Computational null models'
            ],
            expected_outcomes={
                'scale_invariance': 'Power law scaling across 4 orders of magnitude',
                'information_flow': 'Bidirectional between scales',
                'critical_dynamics': 'Branching ratio σ = 1.0 ± 0.1',
                'coherence_decay': 'Log-linear with scale separation'
            }
        )
        
        return protocol
    
    def validate_criticality(self, neural_data: np.ndarray) -> Dict:
        """
        Validate critical brain dynamics
        """
        validation = {}
        
        # 1. Avalanche size distribution
        avalanche_sizes = self._detect_avalanches(neural_data)
        
        # Fit power law
        if len(avalanche_sizes) > 50:
            tau, p_value = self._fit_power_law(avalanche_sizes)
            validation['avalanche_exponent'] = tau
            validation['power_law_p_value'] = p_value
            validation['is_power_law'] = p_value > 0.1
        
        # 2. Branching ratio
        branching_ratio = self._calculate_branching_ratio(neural_data)
        validation['branching_ratio'] = branching_ratio
        validation['is_critical'] = 0.95 < branching_ratio < 1.05
        
        # 3. Correlation length
        correlation_length = self._calculate_correlation_length(neural_data)
        validation['correlation_length'] = correlation_length
        validation['long_range_correlations'] = correlation_length > len(neural_data) * 0.1
        
        # 4. Susceptibility peak
        susceptibility = self._calculate_susceptibility(neural_data)
        validation['susceptibility'] = susceptibility
        validation['susceptibility_divergence'] = susceptibility > np.mean(susceptibility) + 2 * np.std(susceptibility)
        
        return validation
    
    def _detect_avalanches(self, data: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Detect neuronal avalanches"""
        # Threshold crossings
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        binary = (data > threshold * np.std(data, axis=1, keepdims=True)).astype(int)
        
        avalanches = []
        in_avalanche = False
        current_size = 0
        
        for t in range(binary.shape[1]):
            active = np.sum(binary[:, t])
            
            if active > 0:
                in_avalanche = True
                current_size += active
            elif in_avalanche:
                avalanches.append(current_size)
                current_size = 0
                in_avalanche = False
        
        return avalanches
    
    def _fit_power_law(self, data: List[int]) -> Tuple[float, float]:
        """Fit power law distribution and return exponent and p-value"""
        # Use maximum likelihood estimation
        data = np.array(data)
        x_min = np.min(data)
        
        # MLE for power law exponent
        n = len(data)
        tau = 1 + n / np.sum(np.log(data / x_min))
        
        # Kolmogorov-Smirnov test
        # Generate theoretical distribution
        x_theory = np.arange(x_min, np.max(data) + 1)
        p_theory = x_theory ** (-tau)
        p_theory /= np.sum(p_theory)
        
        # Empirical distribution
        hist, bins = np.histogram(data, bins=len(x_theory))
        p_empirical = hist / np.sum(hist)
        
        # KS statistic
        ks_stat, p_value = stats.ks_2samp(
            np.repeat(x_theory, (p_theory * n).astype(int)),
            data
        )
        
        return tau, p_value
    
    def _calculate_branching_ratio(self, data: np.ndarray) -> float:
        """Calculate branching ratio"""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        threshold = np.std(data)
        active = (data > threshold).astype(int)
        
        ratios = []
        for t in range(active.shape[1] - 1):
            n_active = np.sum(active[:, t])
            n_next = np.sum(active[:, t + 1])
            
            if n_active > 0:
                ratios.append(n_next / n_active)
        
        return np.mean(ratios) if ratios else 0.0
    
    def _calculate_correlation_length(self, data: np.ndarray) -> float:
        """Calculate spatial correlation length"""
        if len(data.shape) == 1:
            return 1.0
        
        n_channels, n_samples = data.shape
        
        # Spatial correlation function
        correlations = []
        
        for t in range(n_samples):
            snapshot = data[:, t]
            
            # Calculate pairwise correlations
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    distance = abs(i - j)
                    correlation = np.corrcoef(snapshot[i], snapshot[j])[0, 1]
                    correlations.append((distance, correlation))
        
        if not correlations:
            return 1.0
        
        # Fit exponential decay
        distances, corrs = zip(*correlations)
        distances = np.array(distances)
        corrs = np.abs(corrs)
        
        # Find decay length
        try:
            popt, _ = optimize.curve_fit(
                lambda x, xi: np.exp(-x / xi),
                distances,
                corrs,
                p0=[1.0]
            )
            return popt[0]
        except:
            return 1.0
    
    def _calculate_susceptibility(self, data: np.ndarray) -> float:
        """Calculate system susceptibility (response to perturbation)"""
        # Variance is related to susceptibility via fluctuation-dissipation theorem
        return np.var(data)
    
    def statistical_validation(self, experimental_data: Dict, 
                             theoretical_predictions: Dict) -> Dict:
        """
        Statistical validation of experimental results against theory
        """
        validation_results = {}
        
        for key in experimental_data:
            if key in theoretical_predictions:
                exp_val = experimental_data[key]
                theory_val = theoretical_predictions[key]
                
                # Perform appropriate statistical test
                if isinstance(exp_val, (list, np.ndarray)) and isinstance(theory_val, (list, np.ndarray)):
                    # Compare distributions
                    statistic, p_value = stats.ks_2samp(exp_val, theory_val)
                    validation_results[key] = {
                        'test': 'Kolmogorov-Smirnov',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                elif isinstance(exp_val, (int, float)) and isinstance(theory_val, (int, float)):
                    # Compare values with tolerance
                    relative_error = abs(exp_val - theory_val) / abs(theory_val)
                    validation_results[key] = {
                        'experimental': exp_val,
                        'theoretical': theory_val,
                        'relative_error': relative_error,
                        'validated': relative_error < 0.2  # 20% tolerance
                    }
        
        return validation_results


# Example usage and testing
if __name__ == "__main__":
    print("Testing Information-Theoretic Measures...")
    
    # Initialize measures
    itm = InformationTheoreticMeasures(n_elements=10)
    
    # Generate test data
    np.random.seed(42)
    test_signal = np.random.randn(1000)
    test_signal2 = 0.5 * test_signal + 0.5 * np.random.randn(1000)
    
    # Calculate measures
    entropy = itm.entropy(np.abs(test_signal))
    mi = itm.mutual_information(test_signal, test_signal2)
    te = itm.transfer_entropy(test_signal[:500], test_signal2[:500])
    
    print(f"Entropy: {entropy:.3f} bits")
    print(f"Mutual Information: {mi:.3f} bits")
    print(f"Transfer Entropy: {te:.3f} bits")
    
    # Test integrated information
    connectivity = np.random.rand(10, 10)
    connectivity = (connectivity + connectivity.T) / 2  # Symmetrize
    np.fill_diagonal(connectivity, 0)
    
    system_state = np.random.randn(10)
    phi = itm.integrated_information(system_state, connectivity)
    print(f"Integrated Information (Φ): {phi:.3f}")
    
    # Complexity measures
    complexity = itm.complexity_measures(test_signal)
    print(f"Lempel-Ziv Complexity: {complexity['lempel_ziv']:.3f}")
    print(f"Approximate Entropy: {complexity['approximate_entropy']:.3f}")
    print(f"Sample Entropy: {complexity['sample_entropy']:.3f}")
    
    print("\nTesting Experimental Validation Framework...")
    
    # Initialize validation
    exp_val = ExperimentalValidation()
    
    # Get protocols
    schumann_protocol = exp_val.schumann_resonance_protocol()
    print(f"\nProtocol: {schumann_protocol.name}")
    print(f"Expected outcome: {schumann_protocol.expected_outcomes}")
    
    # Validate criticality
    test_neural_data = np.random.randn(8, 1000)
    # Add some structure for more realistic testing
    for i in range(8):
        test_neural_data[i] += np.sin(2 * np.pi * 10 * np.arange(1000) / 1000)
    
    criticality_validation = exp_val.validate_criticality(test_neural_data)
    print(f"\nCriticality Validation:")
    print(f"  Branching ratio: {criticality_validation['branching_ratio']:.3f}")
    print(f"  Is critical: {criticality_validation['is_critical']}")
    print(f"  Long-range correlations: {criticality_validation['long_range_correlations']}")
    
    # Statistical validation example
    exp_data = {'branching_ratio': 0.98, 'avalanche_exponent': 1.48}
    theory_pred = {'branching_ratio': 1.0, 'avalanche_exponent': 1.5}
    
    stat_validation = exp_val.statistical_validation(exp_data, theory_pred)
    print(f"\nStatistical Validation:")
    for key, result in stat_validation.items():
        print(f"  {key}: {result}")
