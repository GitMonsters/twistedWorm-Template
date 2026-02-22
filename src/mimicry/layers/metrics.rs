//! Quality Metrics for the 8-Layer Multiplicative Integration System (Phase 5D).
//!
//! This module provides nuanced metrics that go beyond raw confidence values,
//! measuring the *quality* and *health* of layer processing results. These
//! metrics help distinguish between genuinely high-quality results and those
//! that merely saturated to the confidence cap.
//!
//! # Metrics
//!
//! - **Information Quality**: Shannon entropy of layer confidences — measures
//!   how much information the confidence distribution carries. High entropy means
//!   layers have diverse opinions; low entropy means they are all similar.
//!
//! - **Activation Diversity**: Fraction of layers with confidence above a threshold.
//!   Distinguishes results where all layers participated from those driven by
//!   one or two dominant layers.
//!
//! - **Convergence Speed**: Normalized measure of how quickly iterative processing
//!   reached convergence (0.0 = used all iterations, 1.0 = converged in one).
//!
//! - **Layer Utilization Balance**: Gini coefficient of layer confidences.
//!   0.0 = perfectly balanced (all layers equal), 1.0 = maximally imbalanced
//!   (one layer dominates).
//!
//! - **Effective Amplification Ratio**: Ratio of combined output confidence to
//!   the arithmetic mean of raw layer inputs, adjusted for cap effects.
//!   Values > 1.0 indicate the compounding system added value.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::layer::Layer;
use super::stack::StackProcessResult;

// =============================================================================
// QualityMetrics
// =============================================================================

/// A complete set of quality metrics computed from a `StackProcessResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Shannon entropy of the layer confidence distribution (bits).
    /// Higher values indicate more diverse layer opinions.
    pub information_quality: f32,

    /// Fraction of layers with confidence above the activation threshold.
    /// Range: [0.0, 1.0]. Higher is better for coverage.
    pub activation_diversity: f32,

    /// Normalized convergence speed.
    /// Range: [0.0, 1.0]. 1.0 = instant convergence, 0.0 = did not converge.
    pub convergence_speed: f32,

    /// Gini coefficient of layer confidences.
    /// Range: [0.0, 1.0]. 0.0 = perfectly balanced, 1.0 = one layer dominates.
    pub utilization_balance: f32,

    /// Effective amplification ratio: combined_confidence / arithmetic_mean.
    /// Values > 1.0 mean the compounding system amplified beyond simple averaging.
    pub effective_amplification: f32,

    /// Overall quality score combining all metrics (0.0 - 1.0).
    pub overall_quality: f32,

    /// Number of active layers (confidence > 0).
    pub active_layers: usize,

    /// Total layers in the result.
    pub total_layers: usize,

    /// Whether the result appears to have saturated at the confidence cap.
    pub appears_saturated: bool,
}

/// Configuration for the quality metrics computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricsConfig {
    /// Minimum confidence for a layer to be considered "active" for diversity.
    pub activation_threshold: f32,

    /// Weight of information_quality in the overall score.
    pub info_quality_weight: f32,

    /// Weight of activation_diversity in the overall score.
    pub diversity_weight: f32,

    /// Weight of convergence_speed in the overall score.
    pub convergence_weight: f32,

    /// Weight of utilization_balance in the overall score (inverted: lower Gini = higher score).
    pub balance_weight: f32,

    /// Weight of effective_amplification in the overall score.
    pub amplification_weight: f32,

    /// Saturation detection threshold: if combined_confidence is within this
    /// fraction of the effective cap, the result is considered saturated.
    pub saturation_threshold: f32,
}

impl Default for QualityMetricsConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.1,
            info_quality_weight: 0.20,
            diversity_weight: 0.20,
            convergence_weight: 0.15,
            balance_weight: 0.25,
            amplification_weight: 0.20,
            saturation_threshold: 0.95,
        }
    }
}

// =============================================================================
// MetricsAnalyzer
// =============================================================================

/// Stateful analyzer that computes quality metrics from processing results
/// and tracks aggregate statistics over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAnalyzer {
    config: QualityMetricsConfig,
    /// Total number of results analyzed.
    pub total_analyzed: u64,
    /// Running sum of overall quality scores.
    quality_sum: f64,
    /// Running sum of information quality.
    info_quality_sum: f64,
    /// Running sum of activation diversity.
    diversity_sum: f64,
    /// Running sum of convergence speed.
    convergence_sum: f64,
    /// Running sum of utilization balance (Gini).
    balance_sum: f64,
    /// Running sum of effective amplification.
    amplification_sum: f64,
    /// Count of saturated results.
    saturation_count: u64,
    /// Best overall quality observed.
    best_quality: f32,
    /// Worst overall quality observed.
    worst_quality: f32,
    /// Per-layer average confidences (running sum).
    layer_confidence_sums: HashMap<Layer, f64>,
}

impl MetricsAnalyzer {
    /// Create a new analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(QualityMetricsConfig::default())
    }

    /// Create a new analyzer with custom configuration.
    pub fn with_config(config: QualityMetricsConfig) -> Self {
        Self {
            config,
            total_analyzed: 0,
            quality_sum: 0.0,
            info_quality_sum: 0.0,
            diversity_sum: 0.0,
            convergence_sum: 0.0,
            balance_sum: 0.0,
            amplification_sum: 0.0,
            saturation_count: 0,
            best_quality: 0.0,
            worst_quality: 1.0,
            layer_confidence_sums: HashMap::new(),
        }
    }

    /// Analyze a `StackProcessResult` and return quality metrics.
    pub fn analyze(&mut self, result: &StackProcessResult) -> QualityMetrics {
        let confidences: Vec<f32> = result.layer_confidences.values().copied().collect();
        let total_layers = confidences.len();

        // --- Information Quality (Shannon entropy) ---
        let information_quality = self.compute_shannon_entropy(&confidences);

        // --- Activation Diversity ---
        let active_layers = confidences
            .iter()
            .filter(|&&c| c > self.config.activation_threshold)
            .count();
        let activation_diversity = if total_layers > 0 {
            active_layers as f32 / total_layers as f32
        } else {
            0.0
        };

        // --- Convergence Speed ---
        let convergence_speed = self.compute_convergence_speed(result);

        // --- Layer Utilization Balance (Gini coefficient) ---
        let utilization_balance = self.compute_gini_coefficient(&confidences);

        // --- Effective Amplification Ratio ---
        let arithmetic_mean = if total_layers > 0 {
            confidences.iter().sum::<f32>() / total_layers as f32
        } else {
            0.0
        };
        let effective_amplification = if arithmetic_mean > 0.001 {
            result.combined_confidence / arithmetic_mean
        } else {
            1.0
        };

        // --- Saturation Detection ---
        let cap = if result.effective_cap > 0.0 {
            result.effective_cap
        } else {
            2.0 // fallback to default max_confidence
        };
        let appears_saturated =
            result.combined_confidence >= cap * self.config.saturation_threshold;

        // --- Overall Quality Score ---
        let overall_quality = self.compute_overall_quality(
            information_quality,
            activation_diversity,
            convergence_speed,
            utilization_balance,
            effective_amplification,
        );

        // Update running statistics
        self.total_analyzed += 1;
        self.quality_sum += overall_quality as f64;
        self.info_quality_sum += information_quality as f64;
        self.diversity_sum += activation_diversity as f64;
        self.convergence_sum += convergence_speed as f64;
        self.balance_sum += utilization_balance as f64;
        self.amplification_sum += effective_amplification as f64;

        if appears_saturated {
            self.saturation_count += 1;
        }
        if overall_quality > self.best_quality {
            self.best_quality = overall_quality;
        }
        if overall_quality < self.worst_quality {
            self.worst_quality = overall_quality;
        }

        // Update per-layer tracking
        for (layer, &conf) in &result.layer_confidences {
            *self.layer_confidence_sums.entry(*layer).or_insert(0.0) += conf as f64;
        }

        QualityMetrics {
            information_quality,
            activation_diversity,
            convergence_speed,
            utilization_balance,
            effective_amplification,
            overall_quality,
            active_layers,
            total_layers,
            appears_saturated,
        }
    }

    /// Compute Shannon entropy of confidences (in bits).
    ///
    /// We normalize confidences to form a probability distribution, then compute
    /// H = -Σ p_i * log2(p_i). Maximum entropy for 8 layers = log2(8) = 3.0 bits.
    fn compute_shannon_entropy(&self, confidences: &[f32]) -> f32 {
        if confidences.is_empty() {
            return 0.0;
        }

        let sum: f32 = confidences.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        let mut entropy: f32 = 0.0;
        for &c in confidences {
            if c > 0.0 {
                let p = c / sum;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] by dividing by max possible entropy
        let max_entropy = (confidences.len() as f32).log2();
        if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Compute convergence speed as a normalized value.
    ///
    /// 1.0 = converged in 1 iteration, 0.0 = did not converge (used all iterations).
    /// If max_iterations is 0 (no iterative processing), returns 0.5 (neutral).
    fn compute_convergence_speed(&self, result: &StackProcessResult) -> f32 {
        if result.iterations == 0 {
            // No iterative processing was performed
            return 0.5;
        }

        if result.converged {
            // Faster convergence = better. Assume typical max is ~10 iterations.
            // Speed = 1.0 - (iterations_used - 1) / max_scale
            let max_scale = 10.0_f32;
            (1.0 - (result.iterations as f32 - 1.0) / max_scale).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Compute the Gini coefficient of layer confidences.
    ///
    /// The Gini coefficient measures inequality in a distribution:
    /// - 0.0 = perfect equality (all layers have equal confidence)
    /// - 1.0 = maximum inequality (one layer has all the confidence)
    ///
    /// Formula: G = (2 * Σ(i * x_i)) / (n * Σ x_i) - (n + 1) / n
    fn compute_gini_coefficient(&self, confidences: &[f32]) -> f32 {
        let n = confidences.len();
        if n <= 1 {
            return 0.0;
        }

        let sum: f32 = confidences.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        // Sort ascending for Gini computation
        let mut sorted = confidences.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let weighted_sum: f32 = sorted
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as f32 + 1.0) * x)
            .sum();

        let gini = (2.0 * weighted_sum) / (n as f32 * sum) - (n as f32 + 1.0) / n as f32;
        gini.clamp(0.0, 1.0)
    }

    /// Compute the overall quality score from individual metrics.
    fn compute_overall_quality(
        &self,
        info_quality: f32,
        diversity: f32,
        convergence: f32,
        gini: f32,
        amplification: f32,
    ) -> f32 {
        let w = &self.config;

        // Invert Gini: lower Gini = better balance
        let balance_score = 1.0 - gini;

        // Normalize amplification: cap contribution at 2.0x (diminishing returns)
        let amp_score = ((amplification - 1.0).max(0.0) / 1.0).min(1.0);

        let weighted = info_quality * w.info_quality_weight
            + diversity * w.diversity_weight
            + convergence * w.convergence_weight
            + balance_score * w.balance_weight
            + amp_score * w.amplification_weight;

        let total_weight = w.info_quality_weight
            + w.diversity_weight
            + w.convergence_weight
            + w.balance_weight
            + w.amplification_weight;

        if total_weight > 0.0 {
            (weighted / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    // =========================================================================
    // Aggregate statistics
    // =========================================================================

    /// Get the average overall quality across all analyzed results.
    pub fn average_quality(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        (self.quality_sum / self.total_analyzed as f64) as f32
    }

    /// Get the average information quality.
    pub fn average_info_quality(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        (self.info_quality_sum / self.total_analyzed as f64) as f32
    }

    /// Get the average activation diversity.
    pub fn average_diversity(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        (self.diversity_sum / self.total_analyzed as f64) as f32
    }

    /// Get the average convergence speed.
    pub fn average_convergence_speed(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        (self.convergence_sum / self.total_analyzed as f64) as f32
    }

    /// Get the average Gini coefficient (utilization balance).
    pub fn average_gini(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        (self.balance_sum / self.total_analyzed as f64) as f32
    }

    /// Get the average effective amplification ratio.
    pub fn average_amplification(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 1.0;
        }
        (self.amplification_sum / self.total_analyzed as f64) as f32
    }

    /// Get the saturation rate (fraction of results that hit the cap).
    pub fn saturation_rate(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        self.saturation_count as f32 / self.total_analyzed as f32
    }

    /// Get the best overall quality score observed.
    pub fn best_quality(&self) -> f32 {
        self.best_quality
    }

    /// Get the worst overall quality score observed.
    pub fn worst_quality(&self) -> f32 {
        if self.total_analyzed == 0 {
            return 0.0;
        }
        self.worst_quality
    }

    /// Get per-layer average confidences.
    pub fn layer_averages(&self) -> HashMap<Layer, f32> {
        if self.total_analyzed == 0 {
            return HashMap::new();
        }
        self.layer_confidence_sums
            .iter()
            .map(|(layer, &sum)| (*layer, (sum / self.total_analyzed as f64) as f32))
            .collect()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        format!(
            "=== QUALITY METRICS SUMMARY ===\n\
             Total Analyzed: {}\n\
             \n\
             Average Scores:\n\
               Overall Quality:      {:.4}\n\
               Information Quality:  {:.4}\n\
               Activation Diversity: {:.4}\n\
               Convergence Speed:    {:.4}\n\
               Utilization Balance:  {:.4} (Gini, lower=better)\n\
               Effective Amplify:    {:.4}x\n\
             \n\
             Extremes:\n\
               Best Quality:         {:.4}\n\
               Worst Quality:        {:.4}\n\
               Saturation Rate:      {:.1}%\n\
             \n\
             Layer Averages:\n{}",
            self.total_analyzed,
            self.average_quality(),
            self.average_info_quality(),
            self.average_diversity(),
            self.average_convergence_speed(),
            self.average_gini(),
            self.average_amplification(),
            self.best_quality(),
            self.worst_quality(),
            self.saturation_rate() * 100.0,
            self.format_layer_averages(),
        )
    }

    /// Format per-layer averages for display.
    fn format_layer_averages(&self) -> String {
        let averages = self.layer_averages();
        if averages.is_empty() {
            return "  No layer data.".to_string();
        }

        let mut lines = Vec::new();
        for layer in Layer::all() {
            if let Some(&avg) = averages.get(layer) {
                lines.push(format!(
                    "  L{} {}: {:.4}",
                    layer.number(),
                    layer.name(),
                    avg
                ));
            }
        }
        lines.join("\n")
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.total_analyzed = 0;
        self.quality_sum = 0.0;
        self.info_quality_sum = 0.0;
        self.diversity_sum = 0.0;
        self.convergence_sum = 0.0;
        self.balance_sum = 0.0;
        self.amplification_sum = 0.0;
        self.saturation_count = 0;
        self.best_quality = 0.0;
        self.worst_quality = 1.0;
        self.layer_confidence_sums.clear();
    }
}

impl Default for MetricsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Standalone convenience functions
// =============================================================================

/// Compute quality metrics for a single result using default configuration.
pub fn compute_quality(result: &StackProcessResult) -> QualityMetrics {
    let mut analyzer = MetricsAnalyzer::new();
    analyzer.analyze(result)
}

/// Compare quality metrics between two results, returning (metrics_a, metrics_b, delta).
///
/// The delta is computed as (a - b) for each numeric field, so positive delta
/// means result_a scored higher on that metric.
pub fn compare_quality(
    result_a: &StackProcessResult,
    result_b: &StackProcessResult,
) -> QualityComparison {
    let mut analyzer = MetricsAnalyzer::new();
    let metrics_a = analyzer.analyze(result_a);

    let mut analyzer_b = MetricsAnalyzer::new();
    let metrics_b = analyzer_b.analyze(result_b);

    let delta = QualityDelta {
        information_quality: metrics_a.information_quality - metrics_b.information_quality,
        activation_diversity: metrics_a.activation_diversity - metrics_b.activation_diversity,
        convergence_speed: metrics_a.convergence_speed - metrics_b.convergence_speed,
        utilization_balance: metrics_a.utilization_balance - metrics_b.utilization_balance,
        effective_amplification: metrics_a.effective_amplification
            - metrics_b.effective_amplification,
        overall_quality: metrics_a.overall_quality - metrics_b.overall_quality,
    };

    // Count wins per metric (positive delta = A wins)
    let mut a_wins = 0u32;
    let mut b_wins = 0u32;

    // For info quality, diversity, convergence, amplification: higher is better
    if delta.information_quality > 0.001 {
        a_wins += 1;
    } else if delta.information_quality < -0.001 {
        b_wins += 1;
    }
    if delta.activation_diversity > 0.001 {
        a_wins += 1;
    } else if delta.activation_diversity < -0.001 {
        b_wins += 1;
    }
    if delta.convergence_speed > 0.001 {
        a_wins += 1;
    } else if delta.convergence_speed < -0.001 {
        b_wins += 1;
    }
    if delta.effective_amplification > 0.001 {
        a_wins += 1;
    } else if delta.effective_amplification < -0.001 {
        b_wins += 1;
    }
    // For utilization_balance (Gini): LOWER is better, so negative delta = A wins
    if delta.utilization_balance < -0.001 {
        a_wins += 1;
    } else if delta.utilization_balance > 0.001 {
        b_wins += 1;
    }

    QualityComparison {
        metrics_a,
        metrics_b,
        delta,
        a_metric_wins: a_wins,
        b_metric_wins: b_wins,
    }
}

/// Result of comparing quality metrics between two processing results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityComparison {
    pub metrics_a: QualityMetrics,
    pub metrics_b: QualityMetrics,
    pub delta: QualityDelta,
    /// Number of individual metrics where A scored better.
    pub a_metric_wins: u32,
    /// Number of individual metrics where B scored better.
    pub b_metric_wins: u32,
}

/// Per-metric deltas (A - B).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDelta {
    pub information_quality: f32,
    pub activation_diversity: f32,
    pub convergence_speed: f32,
    pub utilization_balance: f32,
    pub effective_amplification: f32,
    pub overall_quality: f32,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(
        confidences: &[(Layer, f32)],
        combined: f32,
        iterations: u32,
        converged: bool,
    ) -> StackProcessResult {
        let mut result = StackProcessResult::empty();
        for &(layer, conf) in confidences {
            result.layer_confidences.insert(layer, conf);
        }
        result.combined_confidence = combined;
        result.iterations = iterations;
        result.converged = converged;
        result.effective_cap = 2.0;
        result
    }

    fn all_layers_uniform(conf: f32) -> Vec<(Layer, f32)> {
        Layer::all().iter().map(|&l| (l, conf)).collect()
    }

    // -------------------------------------------------------------------------
    // Shannon Entropy
    // -------------------------------------------------------------------------

    #[test]
    fn test_entropy_uniform_distribution() {
        // All layers equal => maximum entropy => normalized to 1.0
        let result = make_result(&all_layers_uniform(0.5), 0.5, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.information_quality - 1.0).abs() < 0.01,
            "Uniform distribution should have max entropy, got {}",
            metrics.information_quality
        );
    }

    #[test]
    fn test_entropy_single_layer() {
        // Only one layer active => minimum entropy
        let result = make_result(&[(Layer::BasePhysics, 1.0)], 1.0, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            metrics.information_quality < 0.01,
            "Single layer should have zero entropy, got {}",
            metrics.information_quality
        );
    }

    #[test]
    fn test_entropy_skewed_distribution() {
        // One layer dominates => low entropy
        let mut confs: Vec<(Layer, f32)> = Layer::all().iter().map(|&l| (l, 0.01)).collect();
        confs[0].1 = 1.0; // L1 dominates
        let result = make_result(&confs, 0.5, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            metrics.information_quality < 0.5,
            "Skewed distribution should have low entropy, got {}",
            metrics.information_quality
        );
    }

    // -------------------------------------------------------------------------
    // Activation Diversity
    // -------------------------------------------------------------------------

    #[test]
    fn test_diversity_all_active() {
        let result = make_result(&all_layers_uniform(0.5), 0.5, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.activation_diversity - 1.0).abs() < 0.01,
            "All layers active should give diversity=1.0, got {}",
            metrics.activation_diversity
        );
    }

    #[test]
    fn test_diversity_half_active() {
        let confs: Vec<(Layer, f32)> = Layer::all()
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, if i < 4 { 0.5 } else { 0.01 }))
            .collect();
        let result = make_result(&confs, 0.4, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.activation_diversity - 0.5).abs() < 0.01,
            "Half active should give diversity=0.5, got {}",
            metrics.activation_diversity
        );
    }

    #[test]
    fn test_diversity_none_active() {
        let confs: Vec<(Layer, f32)> = Layer::all().iter().map(|&l| (l, 0.01)).collect();
        let result = make_result(&confs, 0.01, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            metrics.activation_diversity < 0.01,
            "All below threshold should give diversity~0, got {}",
            metrics.activation_diversity
        );
    }

    // -------------------------------------------------------------------------
    // Convergence Speed
    // -------------------------------------------------------------------------

    #[test]
    fn test_convergence_fast() {
        // Converged in 1 iteration
        let result = make_result(&all_layers_uniform(0.5), 0.5, 1, true);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.convergence_speed - 1.0).abs() < 0.01,
            "Converged in 1 iter should give speed=1.0, got {}",
            metrics.convergence_speed
        );
    }

    #[test]
    fn test_convergence_slow() {
        // Did not converge
        let result = make_result(&all_layers_uniform(0.5), 0.5, 10, false);
        let metrics = compute_quality(&result);
        assert!(
            metrics.convergence_speed < 0.01,
            "Non-convergence should give speed~0, got {}",
            metrics.convergence_speed
        );
    }

    #[test]
    fn test_convergence_no_iterations() {
        // No iterative processing (forward-only)
        let result = make_result(&all_layers_uniform(0.5), 0.5, 0, false);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.convergence_speed - 0.5).abs() < 0.01,
            "No iterations should give neutral speed=0.5, got {}",
            metrics.convergence_speed
        );
    }

    // -------------------------------------------------------------------------
    // Gini Coefficient (Utilization Balance)
    // -------------------------------------------------------------------------

    #[test]
    fn test_gini_uniform() {
        // All equal => Gini = 0
        let result = make_result(&all_layers_uniform(0.5), 0.5, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            metrics.utilization_balance < 0.01,
            "Uniform should have Gini~0, got {}",
            metrics.utilization_balance
        );
    }

    #[test]
    fn test_gini_imbalanced() {
        // Very skewed
        let mut confs: Vec<(Layer, f32)> = Layer::all().iter().map(|&l| (l, 0.0)).collect();
        confs[0].1 = 1.0; // Only L1 has confidence
        let result = make_result(&confs, 0.5, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            metrics.utilization_balance > 0.7,
            "One-layer-dominates should have high Gini, got {}",
            metrics.utilization_balance
        );
    }

    // -------------------------------------------------------------------------
    // Effective Amplification
    // -------------------------------------------------------------------------

    #[test]
    fn test_amplification_no_boost() {
        // combined_confidence == arithmetic mean => ratio = 1.0
        let confs = all_layers_uniform(0.5);
        let result = make_result(&confs, 0.5, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.effective_amplification - 1.0).abs() < 0.01,
            "No boost should give amplification=1.0, got {}",
            metrics.effective_amplification
        );
    }

    #[test]
    fn test_amplification_boosted() {
        // combined_confidence > arithmetic mean
        let confs = all_layers_uniform(0.5);
        let result = make_result(&confs, 1.0, 3, true);
        let metrics = compute_quality(&result);
        assert!(
            (metrics.effective_amplification - 2.0).abs() < 0.01,
            "2x boost should give amplification=2.0, got {}",
            metrics.effective_amplification
        );
    }

    // -------------------------------------------------------------------------
    // Saturation Detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_saturation_detected() {
        // combined_confidence at cap
        let confs = all_layers_uniform(2.0);
        let mut result = make_result(&confs, 2.0, 3, true);
        result.effective_cap = 2.0;
        let metrics = compute_quality(&result);
        assert!(
            metrics.appears_saturated,
            "At cap should be detected as saturated"
        );
    }

    #[test]
    fn test_saturation_not_detected() {
        let confs = all_layers_uniform(0.5);
        let mut result = make_result(&confs, 0.5, 3, true);
        result.effective_cap = 2.0;
        let metrics = compute_quality(&result);
        assert!(
            !metrics.appears_saturated,
            "Well below cap should not be saturated"
        );
    }

    // -------------------------------------------------------------------------
    // Overall Quality
    // -------------------------------------------------------------------------

    #[test]
    fn test_overall_quality_good_result() {
        // Uniform confidences, converged fast, some amplification
        let confs = all_layers_uniform(0.7);
        let result = make_result(&confs, 1.0, 2, true);
        let metrics = compute_quality(&result);
        assert!(
            metrics.overall_quality > 0.5,
            "Good result should have quality > 0.5, got {}",
            metrics.overall_quality
        );
    }

    #[test]
    fn test_overall_quality_poor_result() {
        // Only one layer, no convergence, no amplification
        let result = make_result(&[(Layer::BasePhysics, 0.1)], 0.1, 5, false);
        let metrics = compute_quality(&result);
        assert!(
            metrics.overall_quality < 0.5,
            "Poor result should have quality < 0.5, got {}",
            metrics.overall_quality
        );
    }

    // -------------------------------------------------------------------------
    // MetricsAnalyzer aggregate stats
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyzer_accumulation() {
        let mut analyzer = MetricsAnalyzer::new();

        let r1 = make_result(&all_layers_uniform(0.5), 0.6, 2, true);
        let r2 = make_result(&all_layers_uniform(0.7), 0.9, 3, true);
        let r3 = make_result(&all_layers_uniform(2.0), 2.0, 5, true);

        let m1 = analyzer.analyze(&r1);
        let m2 = analyzer.analyze(&r2);
        let m3 = analyzer.analyze(&r3);

        assert_eq!(analyzer.total_analyzed, 3);
        assert!(analyzer.saturation_count >= 1, "r3 should saturate");

        let avg_q = analyzer.average_quality();
        let expected_avg = (m1.overall_quality + m2.overall_quality + m3.overall_quality) / 3.0;
        assert!(
            (avg_q - expected_avg).abs() < 0.01,
            "Average quality should match: {} vs {}",
            avg_q,
            expected_avg
        );

        assert!(analyzer.best_quality() >= m1.overall_quality);
        assert!(analyzer.best_quality() >= m2.overall_quality);
        assert!(
            analyzer.worst_quality() <= m1.overall_quality
                || analyzer.worst_quality() <= m2.overall_quality
        );
    }

    #[test]
    fn test_analyzer_reset() {
        let mut analyzer = MetricsAnalyzer::new();
        let r = make_result(&all_layers_uniform(0.5), 0.5, 2, true);
        analyzer.analyze(&r);
        assert_eq!(analyzer.total_analyzed, 1);

        analyzer.reset();
        assert_eq!(analyzer.total_analyzed, 0);
        assert_eq!(analyzer.average_quality(), 0.0);
        assert!(analyzer.layer_averages().is_empty());
    }

    #[test]
    fn test_analyzer_layer_averages() {
        let mut analyzer = MetricsAnalyzer::new();

        let r1 = make_result(
            &[(Layer::BasePhysics, 0.4), (Layer::CrossDomain, 0.8)],
            0.6,
            2,
            true,
        );
        let r2 = make_result(
            &[(Layer::BasePhysics, 0.6), (Layer::CrossDomain, 0.6)],
            0.6,
            2,
            true,
        );

        analyzer.analyze(&r1);
        analyzer.analyze(&r2);

        let avgs = analyzer.layer_averages();
        assert!((avgs[&Layer::BasePhysics] - 0.5).abs() < 0.01);
        assert!((avgs[&Layer::CrossDomain] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_analyzer_saturation_rate() {
        let mut analyzer = MetricsAnalyzer::new();

        // 2 saturated, 2 not
        let sat = make_result(&all_layers_uniform(2.0), 2.0, 3, true);
        let not_sat = make_result(&all_layers_uniform(0.5), 0.5, 3, true);

        analyzer.analyze(&sat);
        analyzer.analyze(&sat);
        analyzer.analyze(&not_sat);
        analyzer.analyze(&not_sat);

        assert!((analyzer.saturation_rate() - 0.5).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------

    #[test]
    fn test_compare_quality() {
        let good = make_result(&all_layers_uniform(0.7), 1.0, 2, true);
        let poor = make_result(&[(Layer::BasePhysics, 0.1)], 0.1, 5, false);

        let comparison = compare_quality(&good, &poor);
        assert!(
            comparison.delta.overall_quality > 0.0,
            "Good result should score higher overall"
        );
        assert!(
            comparison.a_metric_wins > comparison.b_metric_wins,
            "Good result should win more metrics"
        );
    }

    #[test]
    fn test_compare_quality_equal() {
        let r = make_result(&all_layers_uniform(0.5), 0.5, 3, true);
        let comparison = compare_quality(&r, &r);
        assert!(
            comparison.delta.overall_quality.abs() < 0.01,
            "Same result should have near-zero delta"
        );
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------

    #[test]
    fn test_summary_format() {
        let mut analyzer = MetricsAnalyzer::new();
        let r = make_result(&all_layers_uniform(0.5), 0.5, 3, true);
        analyzer.analyze(&r);

        let summary = analyzer.summary();
        assert!(summary.contains("Total Analyzed: 1"));
        assert!(summary.contains("Overall Quality"));
        assert!(summary.contains("Information Quality"));
    }

    // -------------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_result() {
        let result = StackProcessResult::empty();
        let metrics = compute_quality(&result);
        assert_eq!(metrics.active_layers, 0);
        assert_eq!(metrics.total_layers, 0);
        assert_eq!(metrics.information_quality, 0.0);
        assert_eq!(metrics.activation_diversity, 0.0);
    }

    #[test]
    fn test_zero_confidence_layers() {
        let confs = all_layers_uniform(0.0);
        let result = make_result(&confs, 0.0, 0, false);
        let metrics = compute_quality(&result);
        assert_eq!(metrics.information_quality, 0.0);
        assert_eq!(metrics.activation_diversity, 0.0);
        assert_eq!(metrics.utilization_balance, 0.0);
    }

    #[test]
    fn test_custom_config() {
        let config = QualityMetricsConfig {
            activation_threshold: 0.5, // Higher threshold
            ..Default::default()
        };
        let mut analyzer = MetricsAnalyzer::with_config(config);

        // All layers at 0.3 — below the 0.5 threshold
        let confs = all_layers_uniform(0.3);
        let result = make_result(&confs, 0.3, 3, true);
        let metrics = analyzer.analyze(&result);

        assert!(
            metrics.activation_diversity < 0.01,
            "All below custom threshold should give diversity~0, got {}",
            metrics.activation_diversity
        );
    }
}
