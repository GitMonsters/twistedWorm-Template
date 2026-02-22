//! Distribution-based Resonance for inter-layer coupling.
//!
//! Instead of treating resonance as a fixed scalar, this module models
//! each layer's confidence as a distribution (mean + variance). High
//! variance reduces effective resonance, penalizing uncertain layers.
//!
//! # Core Equation
//!
//! ```text
//! resonance(i,j) = sqrt(μᵢ × μⱼ) × (1 − min(sqrt(σᵢ² + σⱼ²), 1.0) × 0.5)
//! ```
//!
//! Where:
//! - `μᵢ`, `μⱼ` = mean confidence of layers i and j
//! - `σᵢ²`, `σⱼ²` = variance of confidence for layers i and j
//! - The variance penalty term reduces resonance when either layer is uncertain

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mimicry::layers::layer::Layer;

/// Confidence distribution for a single layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceDistribution {
    /// Mean confidence (μ).
    pub mean: f32,
    /// Variance of confidence (σ²).
    pub variance: f32,
    /// Number of samples used to compute this distribution.
    pub sample_count: u64,
}

impl ConfidenceDistribution {
    /// Create a new confidence distribution.
    pub fn new(mean: f32, variance: f32) -> Self {
        Self {
            mean: mean.max(0.0),
            variance: variance.max(0.0),
            sample_count: 1,
        }
    }

    /// Create a point distribution (zero variance).
    pub fn point(mean: f32) -> Self {
        Self::new(mean, 0.0)
    }

    /// Create from a set of confidence samples.
    pub fn from_samples(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self::new(0.0, 0.0);
        }

        let n = samples.len() as f32;
        let mean = samples.iter().sum::<f32>() / n;
        let variance = if samples.len() > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0)
        } else {
            0.0
        };

        Self {
            mean,
            variance,
            sample_count: samples.len() as u64,
        }
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f32 {
        self.variance.sqrt()
    }

    /// Get the coefficient of variation (std_dev / mean).
    /// Returns 0 if mean is 0.
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mean > 0.0 {
            self.std_dev() / self.mean
        } else {
            0.0
        }
    }

    /// Update the distribution with a new sample using Welford's online algorithm.
    pub fn update(&mut self, sample: f32) {
        self.sample_count += 1;
        let n = self.sample_count as f32;
        let delta = sample - self.mean;
        self.mean += delta / n;
        let delta2 = sample - self.mean;
        // Online variance: M2 approach
        // We track variance directly, so approximate with EMA-style update
        if n > 1.0 {
            self.variance = self.variance * ((n - 2.0) / (n - 1.0)) + (delta * delta2) / n;
        }
    }

    /// Merge two distributions (assuming independence).
    pub fn merge(&self, other: &ConfidenceDistribution) -> ConfidenceDistribution {
        let total = self.sample_count + other.sample_count;
        if total == 0 {
            return ConfidenceDistribution::new(0.0, 0.0);
        }

        let w1 = self.sample_count as f32 / total as f32;
        let w2 = other.sample_count as f32 / total as f32;

        let combined_mean = self.mean * w1 + other.mean * w2;
        // Combined variance includes within-group and between-group variance
        let combined_variance = w1 * (self.variance + (self.mean - combined_mean).powi(2))
            + w2 * (other.variance + (other.mean - combined_mean).powi(2));

        ConfidenceDistribution {
            mean: combined_mean,
            variance: combined_variance,
            sample_count: total,
        }
    }
}

impl Default for ConfidenceDistribution {
    fn default() -> Self {
        Self::point(0.5)
    }
}

/// Compute distribution-based resonance between two layers.
///
/// ```text
/// resonance(i,j) = sqrt(μᵢ × μⱼ) × (1 − min(sqrt(σᵢ² + σⱼ²), 1.0) × 0.5)
/// ```
pub fn distribution_resonance(
    dist_i: &ConfidenceDistribution,
    dist_j: &ConfidenceDistribution,
) -> f32 {
    let mean_product = dist_i.mean * dist_j.mean;

    // Geometric mean of confidences
    let mean_term = if mean_product > 0.0 {
        mean_product.sqrt()
    } else {
        0.0
    };

    // Variance penalty
    let combined_variance = dist_i.variance + dist_j.variance;
    let variance_penalty = combined_variance.sqrt().min(1.0) * 0.5;

    mean_term * (1.0 - variance_penalty)
}

/// Compute distribution-based resonance from raw values.
///
/// Convenience function that takes means and variances directly.
pub fn distribution_resonance_raw(
    mean_i: f32,
    variance_i: f32,
    mean_j: f32,
    variance_j: f32,
) -> f32 {
    let dist_i = ConfidenceDistribution::new(mean_i, variance_i);
    let dist_j = ConfidenceDistribution::new(mean_j, variance_j);
    distribution_resonance(&dist_i, &dist_j)
}

/// Configuration for the distribution resonance system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionResonanceConfig {
    /// Decay factor for distribution updates (EMA-style).
    pub ema_decay: f32,
    /// Minimum samples before variance is trusted.
    pub min_samples_for_variance: u64,
    /// Default variance for layers with insufficient samples.
    pub default_variance: f32,
    /// Whether to use distribution resonance (false = fallback to scalar).
    pub enabled: bool,
}

impl Default for DistributionResonanceConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.95,
            min_samples_for_variance: 3,
            default_variance: 0.01,
            enabled: true,
        }
    }
}

/// The Distribution Resonance System.
///
/// Tracks per-layer confidence distributions and computes pairwise
/// resonance using the distribution-based formula.
pub struct DistributionResonanceSystem {
    /// Configuration.
    config: DistributionResonanceConfig,
    /// Per-layer confidence distributions.
    distributions: HashMap<u8, ConfidenceDistribution>,
}

impl DistributionResonanceSystem {
    /// Create a new distribution resonance system.
    pub fn new(config: DistributionResonanceConfig) -> Self {
        let mut distributions = HashMap::new();

        // Initialize all 8 layers with default distributions
        for layer in Layer::all() {
            distributions.insert(
                layer.number(),
                ConfidenceDistribution::new(0.5, config.default_variance),
            );
        }

        Self {
            config,
            distributions,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DistributionResonanceConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistributionResonanceConfig {
        &self.config
    }

    /// Update a layer's distribution with a new confidence observation.
    pub fn observe(&mut self, layer: Layer, confidence: f32) {
        let num = layer.number();
        if let Some(dist) = self.distributions.get_mut(&num) {
            dist.update(confidence);
        }
    }

    /// Compute resonance between two layers using their current distributions.
    pub fn resonance(&self, layer_i: Layer, layer_j: Layer) -> f32 {
        if !self.config.enabled {
            // Fallback to simple geometric mean (no variance penalty)
            let mean_i = self
                .distributions
                .get(&layer_i.number())
                .map(|d| d.mean)
                .unwrap_or(0.5);
            let mean_j = self
                .distributions
                .get(&layer_j.number())
                .map(|d| d.mean)
                .unwrap_or(0.5);
            return (mean_i * mean_j).sqrt();
        }

        let dist_i = self.get_effective_distribution(layer_i);
        let dist_j = self.get_effective_distribution(layer_j);

        distribution_resonance(&dist_i, &dist_j)
    }

    /// Compute all pairwise resonances for a set of layers.
    pub fn pairwise_resonances(&self, layers: &[Layer]) -> HashMap<(u8, u8), f32> {
        let mut result = HashMap::new();

        for (idx, &layer_i) in layers.iter().enumerate() {
            for &layer_j in &layers[idx + 1..] {
                let res = self.resonance(layer_i, layer_j);
                result.insert((layer_i.number(), layer_j.number()), res);
            }
        }

        result
    }

    /// Get the distribution for a layer, using defaults for insufficient samples.
    fn get_effective_distribution(&self, layer: Layer) -> ConfidenceDistribution {
        let num = layer.number();
        match self.distributions.get(&num) {
            Some(dist) if dist.sample_count >= self.config.min_samples_for_variance => *dist,
            Some(dist) => {
                // Insufficient samples: use mean but default variance
                ConfidenceDistribution::new(dist.mean, self.config.default_variance)
            }
            None => ConfidenceDistribution::new(0.5, self.config.default_variance),
        }
    }

    /// Get a layer's current distribution.
    pub fn get_distribution(&self, layer: Layer) -> Option<&ConfidenceDistribution> {
        self.distributions.get(&layer.number())
    }

    /// Get all distributions.
    pub fn all_distributions(&self) -> &HashMap<u8, ConfidenceDistribution> {
        &self.distributions
    }

    /// Reset all distributions to defaults.
    pub fn reset(&mut self) {
        for dist in self.distributions.values_mut() {
            *dist = ConfidenceDistribution::new(0.5, self.config.default_variance);
        }
    }
}

impl Default for DistributionResonanceSystem {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_distribution_creation() {
        let dist = ConfidenceDistribution::new(0.8, 0.05);
        assert_eq!(dist.mean, 0.8);
        assert_eq!(dist.variance, 0.05);
        assert_eq!(dist.sample_count, 1);
    }

    #[test]
    fn test_point_distribution() {
        let dist = ConfidenceDistribution::point(0.9);
        assert_eq!(dist.mean, 0.9);
        assert_eq!(dist.variance, 0.0);
    }

    #[test]
    fn test_from_samples() {
        let samples = vec![0.8, 0.82, 0.78, 0.81, 0.79];
        let dist = ConfidenceDistribution::from_samples(&samples);

        assert!((dist.mean - 0.8).abs() < 0.01);
        assert!(dist.variance > 0.0);
        assert!(dist.variance < 0.01); // Low variance for close samples
        assert_eq!(dist.sample_count, 5);
    }

    #[test]
    fn test_from_empty_samples() {
        let dist = ConfidenceDistribution::from_samples(&[]);
        assert_eq!(dist.mean, 0.0);
        assert_eq!(dist.variance, 0.0);
    }

    #[test]
    fn test_distribution_update() {
        let mut dist = ConfidenceDistribution::point(0.8);
        dist.update(0.82);
        dist.update(0.78);

        assert!((dist.mean - 0.8).abs() < 0.01);
        assert!(dist.variance > 0.0); // Should have some variance now
        assert_eq!(dist.sample_count, 3);
    }

    #[test]
    fn test_distribution_merge() {
        let a = ConfidenceDistribution::from_samples(&[0.8, 0.82, 0.78]);
        let b = ConfidenceDistribution::from_samples(&[0.7, 0.72, 0.68]);

        let merged = a.merge(&b);

        // Mean should be between the two group means
        assert!(merged.mean > 0.68 && merged.mean < 0.82);
        assert_eq!(merged.sample_count, 6);
    }

    // ---- Core resonance formula tests ----

    #[test]
    fn test_resonance_zero_variance() {
        // With zero variance, resonance = sqrt(μᵢ × μⱼ)
        let dist_i = ConfidenceDistribution::point(0.8);
        let dist_j = ConfidenceDistribution::point(0.8);

        let res = distribution_resonance(&dist_i, &dist_j);
        // sqrt(0.8 * 0.8) * (1 - 0) = 0.8
        assert!((res - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_resonance_high_variance_reduces() {
        let low_var_i = ConfidenceDistribution::new(0.8, 0.01);
        let low_var_j = ConfidenceDistribution::new(0.8, 0.01);
        let high_var_i = ConfidenceDistribution::new(0.8, 0.5);
        let high_var_j = ConfidenceDistribution::new(0.8, 0.5);

        let res_low = distribution_resonance(&low_var_i, &low_var_j);
        let res_high = distribution_resonance(&high_var_i, &high_var_j);

        // High variance should reduce resonance
        assert!(res_high < res_low);
    }

    #[test]
    fn test_resonance_asymmetric() {
        // One confident layer, one uncertain
        let confident = ConfidenceDistribution::new(0.9, 0.01);
        let uncertain = ConfidenceDistribution::new(0.9, 0.5);

        let res = distribution_resonance(&confident, &uncertain);

        // Should be less than pure confident pair
        let res_both_confident = distribution_resonance(&confident, &confident);
        assert!(res < res_both_confident);
    }

    #[test]
    fn test_resonance_zero_mean() {
        let zero = ConfidenceDistribution::point(0.0);
        let normal = ConfidenceDistribution::point(0.8);

        let res = distribution_resonance(&zero, &normal);
        assert_eq!(res, 0.0);
    }

    #[test]
    fn test_resonance_max_variance_penalty() {
        // variance penalty is capped at 0.5 (when sqrt(σᵢ² + σⱼ²) >= 1.0)
        let dist_i = ConfidenceDistribution::new(1.0, 1.0);
        let dist_j = ConfidenceDistribution::new(1.0, 1.0);

        let res = distribution_resonance(&dist_i, &dist_j);
        // sqrt(1*1) * (1 - min(sqrt(2), 1.0) * 0.5) = 1.0 * (1 - 0.5) = 0.5
        assert!((res - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_resonance_raw() {
        let res = distribution_resonance_raw(0.8, 0.0, 0.8, 0.0);
        assert!((res - 0.8).abs() < 0.001);
    }

    // ---- System tests ----

    #[test]
    fn test_system_creation() {
        let system = DistributionResonanceSystem::with_defaults();
        assert!(system.config().enabled);

        // All 8 layers should have distributions
        for layer in Layer::all() {
            assert!(system.get_distribution(*layer).is_some());
        }
    }

    #[test]
    fn test_system_observe() {
        let mut system = DistributionResonanceSystem::with_defaults();

        system.observe(Layer::BasePhysics, 0.9);
        system.observe(Layer::BasePhysics, 0.88);
        system.observe(Layer::BasePhysics, 0.92);

        let dist = system.get_distribution(Layer::BasePhysics).unwrap();
        assert!(dist.sample_count >= 3);
        // Mean should have shifted toward 0.9
        assert!(dist.mean > 0.7);
    }

    #[test]
    fn test_system_resonance() {
        let mut system = DistributionResonanceSystem::with_defaults();

        // Observe consistent high confidence for both layers
        for _ in 0..5 {
            system.observe(Layer::BasePhysics, 0.9);
            system.observe(Layer::GaiaConsciousness, 0.85);
        }

        let res = system.resonance(Layer::BasePhysics, Layer::GaiaConsciousness);
        assert!(res > 0.0);
        assert!(res <= 1.0);
    }

    #[test]
    fn test_system_disabled() {
        let config = DistributionResonanceConfig {
            enabled: false,
            ..Default::default()
        };
        let system = DistributionResonanceSystem::new(config);

        let res = system.resonance(Layer::BasePhysics, Layer::GaiaConsciousness);
        // Should fallback to simple geometric mean
        assert!(res > 0.0);
    }

    #[test]
    fn test_pairwise_resonances() {
        let system = DistributionResonanceSystem::with_defaults();

        let layers = vec![
            Layer::BasePhysics,
            Layer::GaiaConsciousness,
            Layer::ExternalApis,
        ];
        let pairwise = system.pairwise_resonances(&layers);

        // Should have 3 pairs: (1,4), (1,7), (4,7)
        assert_eq!(pairwise.len(), 3);
    }

    #[test]
    fn test_system_reset() {
        let mut system = DistributionResonanceSystem::with_defaults();

        system.observe(Layer::BasePhysics, 0.9);
        system.observe(Layer::BasePhysics, 0.95);

        system.reset();

        let dist = system.get_distribution(Layer::BasePhysics).unwrap();
        assert!((dist.mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let dist = ConfidenceDistribution::new(0.8, 0.04);
        // std_dev = 0.2, cv = 0.2/0.8 = 0.25
        assert!((dist.coefficient_of_variation() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_std_dev() {
        let dist = ConfidenceDistribution::new(0.8, 0.04);
        assert!((dist.std_dev() - 0.2).abs() < 0.001);
    }
}
