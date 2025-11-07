"""
Information-Theoretic Measures and Experimental Validation for SCPN Layer 4.

This module provides a framework for quantifying consciousness-relevant information
metrics and validating the tissue synchronization theory against experimental data.
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
    Calculates consciousness-relevant information-theoretic metrics.

    This class includes methods for calculating Shannon entropy, mutual information,
    transfer entropy, integrated information (Φ), and various complexity measures.

    Attributes:
        n_elements (int): The number of elements in the system.
        n_states (int): The number of discrete states for discretization.
    """
    
    def __init__(self, n_elements: int, n_states: int = 2):
        """
        Initializes the InformationTheoreticMeasures calculator.

        Args:
            n_elements (int): The number of elements (e.g., neurons) in the system.
            n_states (int): The number of discrete states to use for quantization.
        """
        self.n_elements = n_elements
        self.n_states = n_states
        self.epsilon = 1e-10
        
    def entropy(self, distribution: np.ndarray) -> float:
        """
        Calculates the Shannon entropy of a given distribution.

        Args:
            distribution (np.ndarray): An array of counts or probabilities.

        Returns:
            float: The Shannon entropy in bits.
        """
        p = distribution / (np.sum(distribution) + self.epsilon)
        p = p[p > self.epsilon]
        return -np.sum(p * np.log2(p + self.epsilon))
    
    def mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculates the mutual information between two variables X and Y.

        Args:
            X (np.ndarray): The first time series.
            Y (np.ndarray): The second time series.

        Returns:
            float: The mutual information in bits.
        """
        X_discrete, Y_discrete = self._discretize(X), self._discretize(Y)
        joint_hist = np.histogram2d(X_discrete, Y_discrete, bins=self.n_states)[0]
        joint_prob = joint_hist / np.sum(joint_hist)
        p_x, p_y = np.sum(joint_prob, axis=1), np.sum(joint_prob, axis=0)
        
        mi = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_prob[i, j] > self.epsilon:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (p_x[i] * p_y[j] + self.epsilon) + self.epsilon)
        return max(0, mi)
    
    def transfer_entropy(self, X: np.ndarray, Y: np.ndarray, delay: int = 1) -> float:
        """
        Calculates the transfer entropy from variable X to variable Y.

        Args:
            X (np.ndarray): The source time series.
            Y (np.ndarray): The target time series.
            delay (int): The time delay for the transfer.

        Returns:
            float: The transfer entropy in bits.
        """
        if len(X) <= delay + 1: return 0.0
        
        Y_future, Y_past, X_past = Y[delay + 1:], Y[delay:-1], X[delay:-1]
        Y_future_d, Y_past_d, X_past_d = self._discretize(Y_future), self._discretize(Y_past), self._discretize(X_past)
        
        te, n_samples = 0, len(Y_future_d)
        for i in range(n_samples):
            joint_count = np.sum((Y_future_d == Y_future_d[i]) & (Y_past_d == Y_past_d[i]) & (X_past_d == X_past_d[i]))
            p_joint = joint_count / n_samples
            
            condition_count = np.sum((Y_past_d == Y_past_d[i]) & (X_past_d == X_past_d[i]))
            p_cond_with_x = joint_count / condition_count if condition_count > 0 else 0
            
            past_count = np.sum(Y_past_d == Y_past_d[i])
            future_given_past = np.sum((Y_future_d == Y_future_d[i]) & (Y_past_d == Y_past_d[i]))
            p_cond_without_x = future_given_past / past_count if past_count > 0 else 0
            
            if p_joint > self.epsilon and p_cond_with_x > self.epsilon and p_cond_without_x > self.epsilon:
                te += p_joint * np.log2(p_cond_with_x / p_cond_without_x)
        return max(0, te / n_samples)
    
    def integrated_information(self, system_state: np.ndarray, connectivity: np.ndarray) -> float:
        """
        Calculates a simplified version of Integrated Information (Φ).

        Args:
            system_state (np.ndarray): The current state of the system's elements.
            connectivity (np.ndarray): The connectivity matrix of the system.

        Returns:
            float: The integrated information in bits.
        """
        n = len(system_state)
        if n < 2: return 0.0
        
        min_information = float('inf')
        for partition_size in range(1, n // 2 + 1):
            for partition in itertools.combinations(range(n), partition_size):
                partition_A, partition_B = list(partition), [i for i in range(n) if i not in partition]
                ei = self._effective_information(system_state, connectivity, partition_A, partition_B)
                min_information = min(min_information, ei)
        
        whole_system_info = self._system_information(system_state, connectivity)
        return max(0, whole_system_info - min_information)
    
    def _effective_information(self, state: np.ndarray, connectivity: np.ndarray, partition_A: List[int], partition_B: List[int]) -> float:
        """Calculates the effective information across a partition."""
        info_A_to_B, info_B_to_A = 0, 0
        for i in partition_A:
            for j in partition_B:
                if connectivity[i, j] > 0: info_A_to_B += connectivity[i, j] * self.entropy(state[i:i+1])
        for i in partition_B:
            for j in partition_A:
                if connectivity[i, j] > 0: info_B_to_A += connectivity[i, j] * self.entropy(state[i:i+1])
        return (info_A_to_B + info_B_to_A) / 2
    
    def _system_information(self, state: np.ndarray, connectivity: np.ndarray) -> float:
        """Calculates the total information of the system."""
        system_entropy = self.entropy(state)
        total_connectivity = np.sum(connectivity)
        return system_entropy * (1 + total_connectivity / len(state)) if total_connectivity > 0 else system_entropy
    
    def complexity_measures(self, time_series: np.ndarray) -> Dict:
        """
        Calculates a suite of complexity measures for a time series.

        Args:
            time_series (np.ndarray): The time series to analyze.

        Returns:
            Dict: A dictionary of complexity measures.
        """
        measures = {
            'lempel_ziv': self._lempel_ziv_complexity(time_series),
            'approximate_entropy': self._approximate_entropy(time_series),
            'sample_entropy': self._sample_entropy(time_series),
            'permutation_entropy': self._permutation_entropy(time_series)
        }
        if len(time_series.shape) > 1:
            measures['neural_complexity'] = self._neural_complexity(time_series)
        return measures
    
    def _discretize(self, data: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
        """Discretizes continuous data using quantile-based binning."""
        n_bins = n_bins if n_bins is not None else self.n_states
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(data, quantiles)
        bins[0] -= 1e-10
        return np.digitize(data, bins) - 1
    
    def _lempel_ziv_complexity(self, sequence: np.ndarray) -> float:
        """Calculates Lempel-Ziv complexity, normalized."""
        binary = (sequence > np.median(sequence)).astype(int)
        n, complexity, i = len(binary), 0, 0
        while i < n:
            j = i + 1
            while j <= n:
                substring = tuple(binary[i:j])
                found = False
                for k in range(i):
                    if k + len(substring) <= i and tuple(binary[k:k+len(substring)]) == substring:
                        found = True
                        break
                if not found and j < n: j += 1
                else: break
            complexity += 1
            i = j
        max_complexity = n / np.log2(n) if n > 1 else 1
        return complexity / max_complexity
    
    def _approximate_entropy(self, U: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculates Approximate Entropy (ApEn)."""
        N = len(U)
        def _maxdist(xi, xj, m): return max(abs(xi[k] - xj[k]) for k in range(m))
        def _phi(m):
            patterns = [U[i:i + m] for i in range(N - m + 1)]
            C = [sum(1 for j in range(N - m + 1) if _maxdist(patterns[i], patterns[j], m) <= r * np.std(U)) / (N - m + 1) for i in range(N - m + 1)]
            return sum(np.log(c) for c in C if c > 0) / (N - m + 1)
        try: return _phi(m) - _phi(m + 1)
        except: return 0.0
    
    def _sample_entropy(self, U: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculates Sample Entropy (SampEn)."""
        N = len(U)
        def _maxdist(xi, xj, m): return max(abs(xi[k] - xj[k]) for k in range(m))
        def _phi(m):
            patterns = [U[i:i + m] for i in range(N - m + 1)]
            matches, comparisons = 0, 0
            for i in range(N - m + 1):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(patterns[i], patterns[j], m) <= r * np.std(U): matches += 1
                    comparisons += 1
            return matches / comparisons if comparisons > 0 else 0
        phi_m, phi_m1 = _phi(m), _phi(m + 1)
        if phi_m1 == 0: return float('inf')
        return -np.log(phi_m1 / phi_m) if phi_m > 0 else 0
    
    def _permutation_entropy(self, time_series: np.ndarray, order: int = 3) -> float:
        """Calculates Permutation Entropy."""
        n = len(time_series)
        if n < order: return 0
        permutations = list(itertools.permutations(range(order)))
        perm_count = {perm: 0 for perm in permutations}
        for i in range(n - order + 1):
            sorted_indices = tuple(np.argsort(time_series[i:i + order]))
            if sorted_indices in perm_count: perm_count[sorted_indices] += 1
        total = sum(perm_count.values())
        if total == 0: return 0
        probs = np.array(list(perm_count.values())) / total
        return -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    
    def _neural_complexity(self, multi_channel_data: np.ndarray) -> float:
        """Calculates neural complexity as defined by Tononi et al."""
        n_channels, _ = multi_channel_data.shape
        system_entropy = self.entropy(multi_channel_data.flatten())
        subsystem_entropies = [self.entropy(multi_channel_data[list(subset), :].flatten()) for k in range(1, n_channels) for subset in itertools.combinations(range(n_channels), k)]
        return system_entropy - (np.mean(subsystem_entropies) if subsystem_entropies else 0)

@dataclass
class ExperimentalProtocol:
    """A data structure for defining experimental protocols."""
    name: str
    description: str
    parameters: Dict
    measurements: List[str]
    controls: List[str]
    expected_outcomes: Dict

class ExperimentalValidation:
    """
    A framework for validating the tissue synchronization theory experimentally.

    This class provides methods to define experimental protocols and validate
    simulation results against the predictions of critical brain dynamics.
    """
    
    def __init__(self):
        """Initializes the ExperimentalValidation framework."""
        self.protocols, self.results = [], {}
        
    def schumann_resonance_protocol(self) -> ExperimentalProtocol:
        """Defines a protocol for testing Schumann resonance coupling."""
        return ExperimentalProtocol(
            name="Schumann Resonance Entrainment",
            description="Measure EEG phase coherence at Schumann frequencies.",
            parameters={'frequencies': [7.83, 14.3, 20.8], 'shielding_conditions': ['unshielded', 'faraday_cage']},
            measurements=['EEG_phase_coherence', 'power_spectral_density'],
            controls=['Time-matched recordings in Faraday cage', 'Sham stimulation'],
            expected_outcomes={'phase_coherence_increase': '>20% at 7.83 Hz', 'shielding_effect': 'Reduced coherence'}
        )
    
    def lunar_phase_protocol(self) -> ExperimentalProtocol:
        """Defines a protocol for testing lunar phase effects on biological rhythms."""
        return ExperimentalProtocol(
            name="Lunar Phase Biological Coupling",
            description="Track cell division rates across the lunar cycle.",
            parameters={'duration': 60, 'cell_types': ['fibroblasts', 'neural_progenitors']},
            measurements=['cell_division_rate', 'circadian_gene_expression'],
            controls=['Random phase-shifted light exposure', 'Gravitational shielding'],
            expected_outcomes={'division_rate_modulation': '15-25% variation', 'peak_activity': 'Full moon ± 2 days'}
        )
    
    def biophoton_emission_protocol(self) -> ExperimentalProtocol:
        """Defines a protocol for measuring ultra-weak biophoton emissions."""
        return ExperimentalProtocol(
            name="Biophoton Emission Spectroscopy",
            description="Quantify spontaneous photon emissions from living tissue.",
            parameters={'spectral_range': [200, 800], 'sample_types': ['neural_tissue', 'cardiac_tissue']},
            measurements=['photon_count_rate', 'spectral_distribution'],
            controls=['Dead tissue baseline', 'Dark noise measurements'],
            expected_outcomes={'emission_rate': '10-1000 photons/sec/cm²', 'spectral_peaks': '420-450nm (stressed)'}
        )
    
    def multi_scale_recording_protocol(self) -> ExperimentalProtocol:
        """Defines a protocol for simultaneous multi-scale recordings."""
        return ExperimentalProtocol(
            name="Multi-Scale Synchronization Mapping",
            description="Record from subcellular to tissue level simultaneously.",
            parameters={'recording_levels': {'subcellular': 'Calcium imaging', 'cellular': 'Patch-clamp', 'network': 'MEA'}},
            measurements=['cross_scale_coherence', 'information_flow_direction'],
            controls=['Pharmacological decoupling', 'Computational null models'],
            expected_outcomes={'scale_invariance': 'Power law scaling', 'information_flow': 'Bidirectional'}
        )
    
    def validate_criticality(self, neural_data: np.ndarray) -> Dict:
        """
        Validates the presence of critical brain dynamics in neural data.

        Args:
            neural_data (np.ndarray): An array of neural signals.

        Returns:
            Dict: A dictionary of criticality validation results.
        """
        validation = {}
        avalanche_sizes = self._detect_avalanches(neural_data)
        if len(avalanche_sizes) > 50:
            tau, p_value = self._fit_power_law(avalanche_sizes)
            validation.update({'avalanche_exponent': tau, 'power_law_p_value': p_value, 'is_power_law': p_value > 0.1})
        branching_ratio = self._calculate_branching_ratio(neural_data)
        validation.update({'branching_ratio': branching_ratio, 'is_critical': 0.95 < branching_ratio < 1.05})
        correlation_length = self._calculate_correlation_length(neural_data)
        validation.update({'correlation_length': correlation_length, 'long_range_correlations': correlation_length > len(neural_data) * 0.1})
        susceptibility = self._calculate_susceptibility(neural_data)
        validation.update({'susceptibility': susceptibility, 'susceptibility_divergence': susceptibility > np.mean(susceptibility) + 2 * np.std(susceptibility)})
        return validation
    
    def _detect_avalanches(self, data: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Detects neuronal avalanches in the data."""
        data = data.reshape(1, -1) if len(data.shape) == 1 else data
        binary = (data > threshold * np.std(data, axis=1, keepdims=True)).astype(int)
        avalanches, in_avalanche, current_size = [], False, 0
        for t in range(binary.shape[1]):
            active = np.sum(binary[:, t])
            if active > 0: in_avalanche, current_size = True, current_size + active
            elif in_avalanche: avalanches.append(current_size); current_size, in_avalanche = 0, False
        return avalanches
    
    def _fit_power_law(self, data: List[int]) -> Tuple[float, float]:
        """Fits a power-law distribution to the data."""
        data, x_min = np.array(data), np.min(data)
        n = len(data)
        tau = 1 + n / np.sum(np.log(data / x_min))
        x_theory = np.arange(x_min, np.max(data) + 1)
        p_theory = x_theory ** (-tau)
        p_theory /= np.sum(p_theory)
        _, p_value = stats.ks_2samp(np.repeat(x_theory, (p_theory * n).astype(int)), data)
        return tau, p_value
    
    def _calculate_branching_ratio(self, data: np.ndarray) -> float:
        """Calculates the branching ratio of neural activity."""
        data = data.reshape(1, -1) if len(data.shape) == 1 else data
        active = (data > np.std(data)).astype(int)
        ratios = [np.sum(active[:, t + 1]) / np.sum(active[:, t]) for t in range(active.shape[1] - 1) if np.sum(active[:, t]) > 0]
        return np.mean(ratios) if ratios else 0.0
    
    def _calculate_correlation_length(self, data: np.ndarray) -> float:
        """Calculates the spatial correlation length."""
        if len(data.shape) == 1: return 1.0
        n_channels, n_samples = data.shape
        correlations = [(abs(i - j), np.corrcoef(data[i, t], data[j, t])[0, 1]) for t in range(n_samples) for i in range(n_channels) for j in range(i + 1, n_channels)]
        if not correlations: return 1.0
        distances, corrs = zip(*correlations)
        try:
            popt, _ = optimize.curve_fit(lambda x, xi: np.exp(-x / xi), np.array(distances), np.abs(corrs), p0=[1.0])
            return popt[0]
        except: return 1.0
    
    def _calculate_susceptibility(self, data: np.ndarray) -> float:
        """Calculates the system's susceptibility."""
        return np.var(data)
    
    def statistical_validation(self, experimental_data: Dict, theoretical_predictions: Dict) -> Dict:
        """
        Performs statistical validation of experimental data against theoretical predictions.

        Args:
            experimental_data (Dict): A dictionary of experimental results.
            theoretical_predictions (Dict): A dictionary of theoretical predictions.

        Returns:
            Dict: A dictionary of validation results.
        """
        validation_results = {}
        for key in experimental_data:
            if key in theoretical_predictions:
                exp_val, theory_val = experimental_data[key], theoretical_predictions[key]
                if isinstance(exp_val, (list, np.ndarray)) and isinstance(theory_val, (list, np.ndarray)):
                    statistic, p_value = stats.ks_2samp(exp_val, theory_val)
                    validation_results[key] = {'test': 'Kolmogorov-Smirnov', 'statistic': statistic, 'p_value': p_value, 'significant': p_value < 0.05}
                elif isinstance(exp_val, (int, float)) and isinstance(theory_val, (int, float)):
                    relative_error = abs(exp_val - theory_val) / abs(theory_val)
                    validation_results[key] = {'experimental': exp_val, 'theoretical': theory_val, 'relative_error': relative_error, 'validated': relative_error < 0.2}
        return validation_results

if __name__ == "__main__":
    print("Testing Information-Theoretic Measures...")
    itm = InformationTheoreticMeasures(n_elements=10)
    np.random.seed(42)
    test_signal, test_signal2 = np.random.randn(1000), 0.5 * np.random.randn(1000)
    print(f"Entropy: {itm.entropy(np.abs(test_signal)):.3f} bits")
    print(f"Mutual Information: {itm.mutual_information(test_signal, test_signal2):.3f} bits")
    print(f"Transfer Entropy: {itm.transfer_entropy(test_signal[:500], test_signal2[:500]):.3f} bits")
    
    connectivity = np.random.rand(10, 10)
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)
    print(f"Integrated Information (Φ): {itm.integrated_information(np.random.randn(10), connectivity):.3f}")
    
    complexity = itm.complexity_measures(test_signal)
    print(f"Lempel-Ziv Complexity: {complexity['lempel_ziv']:.3f}")
    
    print("\nTesting Experimental Validation Framework...")
    exp_val = ExperimentalValidation()
    schumann_protocol = exp_val.schumann_resonance_protocol()
    print(f"\nProtocol: {schumann_protocol.name}")
    print(f"Expected outcome: {schumann_protocol.expected_outcomes}")
    
    test_neural_data = np.random.randn(8, 1000)
    for i in range(8): test_neural_data[i] += np.sin(2 * np.pi * 10 * np.arange(1000) / 1000)
    
    criticality_validation = exp_val.validate_criticality(test_neural_data)
    print(f"\nCriticality Validation:")
    print(f"  Branching ratio: {criticality_validation['branching_ratio']:.3f}")
    
    exp_data = {'branching_ratio': 0.98, 'avalanche_exponent': 1.48}
    theory_pred = {'branching_ratio': 1.0, 'avalanche_exponent': 1.5}
    stat_validation = exp_val.statistical_validation(exp_data, theory_pred)
    print(f"\nStatistical Validation:")
    for key, result in stat_validation.items(): print(f"  {key}: {result}")
