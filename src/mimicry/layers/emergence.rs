//! Emergence Framework - Mathematical Formalization
//!
//! This module provides a mathematical framework for predicting and measuring
//! emergent properties in the multiplicative layer system. Emergence occurs
//! when the combined behavior of layers produces effects that cannot be
//! predicted from individual layer behaviors alone.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::layer::{Domain, Layer};
use super::stack::StackProcessResult;

/// Mathematical framework for emergence prediction.
#[derive(Debug, Clone)]
pub struct EmergenceFramework {
    /// Configuration for emergence detection.
    config: EmergenceConfig,
    /// Historical emergence data.
    history: EmergenceHistory,
    /// Learned emergence predictors.
    predictors: EmergencePredictors,
}

/// Configuration for emergence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceConfig {
    /// Minimum interaction strength for emergence consideration.
    pub min_interaction_strength: f32,
    /// Threshold for classifying emergence as significant.
    pub significance_threshold: f32,
    /// Number of historical samples to consider.
    pub history_window: usize,
    /// Whether to learn predictor weights.
    pub adaptive_learning: bool,
    /// Learning rate for predictor updates.
    pub learning_rate: f32,
}

impl Default for EmergenceConfig {
    fn default() -> Self {
        Self {
            min_interaction_strength: 0.1,
            significance_threshold: 0.15,
            history_window: 100,
            adaptive_learning: true,
            learning_rate: 0.01,
        }
    }
}

/// Historical emergence data.
#[derive(Debug, Clone, Default)]
struct EmergenceHistory {
    /// Past emergence measurements.
    measurements: Vec<EmergenceMeasurement>,
    /// Running statistics.
    stats: EmergenceStats,
}

/// A single emergence measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceMeasurement {
    /// Predicted emergence value.
    pub predicted: f32,
    /// Actual observed emergence value.
    pub actual: f32,
    /// Prediction error.
    pub error: f32,
    /// Layer configuration that produced this.
    pub active_layers: Vec<Layer>,
    /// Timestamp.
    pub timestamp: u64,
}

/// Running statistics for emergence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmergenceStats {
    /// Total measurements.
    pub total_measurements: u64,
    /// Average emergence value.
    pub average_emergence: f32,
    /// Maximum emergence observed.
    pub max_emergence: f32,
    /// Average prediction error.
    pub avg_prediction_error: f32,
    /// Count of significant emergence events.
    pub significant_events: u64,
}

/// Learned predictors for emergence.
#[derive(Debug, Clone)]
struct EmergencePredictors {
    /// Pairwise interaction weights.
    pairwise_weights: HashMap<(Layer, Layer), f32>,
    /// Higher-order interaction weights (3+ layers).
    higher_order_weights: HashMap<Vec<Layer>, f32>,
    /// Domain interaction weights (for future use).
    #[allow(dead_code)]
    domain_weights: HashMap<(Domain, Domain), f32>,
}

impl Default for EmergencePredictors {
    fn default() -> Self {
        Self {
            pairwise_weights: Self::initialize_pairwise_weights(),
            higher_order_weights: HashMap::new(),
            domain_weights: Self::initialize_domain_weights(),
        }
    }
}

impl EmergencePredictors {
    fn initialize_pairwise_weights() -> HashMap<(Layer, Layer), f32> {
        let mut weights = HashMap::new();

        // Initialize with theoretical interaction strengths
        // These are learned/refined over time
        for l1 in Layer::all() {
            for l2 in Layer::all() {
                if l1.number() < l2.number() {
                    let base_weight = match (l1, l2) {
                        // Strong synergies
                        (Layer::GaiaConsciousness, Layer::MultilingualProcessing) => 0.8,
                        (Layer::CrossDomain, Layer::GaiaConsciousness) => 0.75,
                        (Layer::BasePhysics, Layer::ExtendedPhysics) => 0.7,
                        (Layer::CollaborativeLearning, Layer::ExternalApis) => 0.65,
                        (Layer::BasePhysics, Layer::PreCognitiveVisualization) => 0.78,
                        (Layer::GaiaConsciousness, Layer::PreCognitiveVisualization) => 0.85,
                        (Layer::CollaborativeLearning, Layer::PreCognitiveVisualization) => 0.7,
                        (Layer::ExternalApis, Layer::PreCognitiveVisualization) => 0.65,
                        // Moderate synergies
                        _ if l1.can_bridge_to(*l2) => 0.5,
                        // Weak synergies
                        _ => 0.2,
                    };
                    weights.insert((*l1, *l2), base_weight);
                }
            }
        }

        weights
    }

    fn initialize_domain_weights() -> HashMap<(Domain, Domain), f32> {
        let mut weights = HashMap::new();

        // Domain interaction strengths
        weights.insert((Domain::Physics, Domain::Consciousness), 0.7);
        weights.insert((Domain::Language, Domain::Consciousness), 0.75);
        weights.insert((Domain::Physics, Domain::Emergent), 0.8);
        weights.insert((Domain::Consciousness, Domain::Social), 0.65);
        weights.insert((Domain::Social, Domain::External), 0.6);
        weights.insert((Domain::Physics, Domain::Visualization), 0.78);
        weights.insert((Domain::Consciousness, Domain::Visualization), 0.8);
        weights.insert((Domain::Social, Domain::Visualization), 0.7);
        weights.insert((Domain::External, Domain::Visualization), 0.65);

        weights
    }
}

/// Result of emergence analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceAnalysis {
    /// Total emergence value (beyond sum of parts).
    pub emergence_value: f32,
    /// Breakdown by layer pair interactions.
    pub pairwise_contributions: HashMap<String, f32>,
    /// Higher-order emergence (3+ layer interactions).
    pub higher_order_emergence: f32,
    /// Predicted vs actual comparison.
    pub prediction_accuracy: f32,
    /// Whether emergence is statistically significant.
    pub is_significant: bool,
    /// Dominant emergence mechanism.
    pub dominant_mechanism: EmergenceMechanism,
    /// Confidence in the analysis.
    pub confidence: f32,
}

/// Types of emergence mechanisms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceMechanism {
    /// Resonance between layers amplifies signal.
    Resonance,
    /// Synergy between complementary domains.
    Synergy,
    /// Collective behavior from multiple agents.
    Collective,
    /// Self-organization across layers.
    SelfOrganization,
    /// No dominant mechanism detected.
    None,
}

impl EmergenceFramework {
    /// Create a new emergence framework.
    pub fn new() -> Self {
        Self::with_config(EmergenceConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: EmergenceConfig) -> Self {
        Self {
            config,
            history: EmergenceHistory::default(),
            predictors: EmergencePredictors::default(),
        }
    }

    /// Predict emergence for a given layer configuration.
    pub fn predict(&self, active_layers: &[Layer]) -> f32 {
        let mut predicted = 0.0f32;

        // Pairwise contributions
        for (i, &l1) in active_layers.iter().enumerate() {
            for &l2 in active_layers.iter().skip(i + 1) {
                let (low, high) = if l1.number() < l2.number() {
                    (l1, l2)
                } else {
                    (l2, l1)
                };

                if let Some(&weight) = self.predictors.pairwise_weights.get(&(low, high)) {
                    predicted += weight;
                }
            }
        }

        // Higher-order contributions (if we have 3+ layers)
        if active_layers.len() >= 3 {
            let mut key: Vec<Layer> = active_layers.to_vec();
            key.sort_by_key(|l| l.number());

            if let Some(&weight) = self.predictors.higher_order_weights.get(&key) {
                predicted += weight;
            } else {
                // Estimate higher-order from pairwise
                predicted += 0.1 * (active_layers.len() as f32 - 2.0);
            }
        }

        // Normalize by number of possible interactions
        let n = active_layers.len() as f32;
        if n > 1.0 {
            predicted / (n * (n - 1.0) / 2.0)
        } else {
            0.0
        }
    }

    /// Analyze emergence from a processing result.
    pub fn analyze(&mut self, result: &StackProcessResult) -> EmergenceAnalysis {
        let active_layers: Vec<Layer> = result.layer_states.keys().copied().collect();

        // Predict emergence
        let predicted = self.predict(&active_layers);

        // Calculate actual emergence
        let actual = self.calculate_actual_emergence(result);

        // Calculate pairwise contributions
        let mut pairwise_contributions = HashMap::new();
        for (i, &l1) in active_layers.iter().enumerate() {
            for &l2 in active_layers.iter().skip(i + 1) {
                let (low, high) = if l1.number() < l2.number() {
                    (l1, l2)
                } else {
                    (l2, l1)
                };

                if let (Some(&c1), Some(&c2)) = (
                    result.layer_confidences.get(&low),
                    result.layer_confidences.get(&high),
                ) {
                    let interaction = (c1 * c2).sqrt() - (c1 + c2) / 2.0;
                    pairwise_contributions
                        .insert(format!("{}-{}", low.name(), high.name()), interaction);
                }
            }
        }

        // Calculate higher-order emergence
        let higher_order = if active_layers.len() >= 3 {
            let geometric_mean: f32 = result
                .layer_confidences
                .values()
                .product::<f32>()
                .powf(1.0 / result.layer_confidences.len() as f32);
            let arithmetic_mean: f32 = result.layer_confidences.values().sum::<f32>()
                / result.layer_confidences.len() as f32;
            geometric_mean - arithmetic_mean
        } else {
            0.0
        };

        // Determine dominant mechanism
        let mechanism = self.classify_mechanism(result, &pairwise_contributions);

        // Calculate prediction accuracy
        let error = (predicted - actual).abs();
        let accuracy = 1.0 - error.min(1.0);

        let is_significant = actual.abs() > self.config.significance_threshold;

        // Record measurement
        let measurement = EmergenceMeasurement {
            predicted,
            actual,
            error,
            active_layers: active_layers.clone(),
            timestamp: current_timestamp(),
        };
        self.record_measurement(measurement);

        // Update predictors if learning is enabled
        if self.config.adaptive_learning && is_significant {
            self.update_predictors(&active_layers, predicted, actual);
        }

        EmergenceAnalysis {
            emergence_value: actual,
            pairwise_contributions,
            higher_order_emergence: higher_order,
            prediction_accuracy: accuracy,
            is_significant,
            dominant_mechanism: mechanism,
            confidence: result.combined_confidence.min(1.0),
        }
    }

    /// Calculate actual emergence value from result.
    fn calculate_actual_emergence(&self, result: &StackProcessResult) -> f32 {
        if result.layer_confidences.is_empty() {
            return 0.0;
        }

        // Emergence = multiplicative - additive
        let values: Vec<f32> = result.layer_confidences.values().copied().collect();
        let n = values.len() as f32;

        let multiplicative = values.iter().product::<f32>().powf(1.0 / n);
        let additive = values.iter().sum::<f32>() / n;

        // Also factor in amplification
        let amplification_bonus = (result.total_amplification - 1.0).max(0.0) * 0.1;

        multiplicative - additive + amplification_bonus
    }

    /// Classify the dominant emergence mechanism.
    fn classify_mechanism(
        &self,
        result: &StackProcessResult,
        pairwise: &HashMap<String, f32>,
    ) -> EmergenceMechanism {
        // Check for resonance (high amplification)
        if result.total_amplification > 1.5 {
            return EmergenceMechanism::Resonance;
        }

        // Check for synergy (positive pairwise interactions)
        let positive_interactions = pairwise.values().filter(|&&v| v > 0.1).count();
        if positive_interactions >= 3 {
            return EmergenceMechanism::Synergy;
        }

        // Check for collective (many layers contributing similarly)
        let confidences: Vec<f32> = result.layer_confidences.values().copied().collect();
        if confidences.len() >= 4 {
            let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
            let variance = confidences.iter().map(|c| (c - mean).powi(2)).sum::<f32>()
                / confidences.len() as f32;
            if variance < 0.05 {
                return EmergenceMechanism::Collective;
            }
        }

        // Check for self-organization (convergence achieved)
        if result.converged && result.iterations > 2 {
            return EmergenceMechanism::SelfOrganization;
        }

        EmergenceMechanism::None
    }

    /// Record a measurement in history.
    fn record_measurement(&mut self, measurement: EmergenceMeasurement) {
        // Update stats
        self.history.stats.total_measurements += 1;
        let n = self.history.stats.total_measurements as f32;

        self.history.stats.average_emergence =
            (self.history.stats.average_emergence * (n - 1.0) + measurement.actual) / n;

        if measurement.actual > self.history.stats.max_emergence {
            self.history.stats.max_emergence = measurement.actual;
        }

        self.history.stats.avg_prediction_error =
            (self.history.stats.avg_prediction_error * (n - 1.0) + measurement.error) / n;

        if measurement.actual.abs() > self.config.significance_threshold {
            self.history.stats.significant_events += 1;
        }

        // Add to history (with window limit)
        self.history.measurements.push(measurement);
        while self.history.measurements.len() > self.config.history_window {
            self.history.measurements.remove(0);
        }
    }

    /// Update predictors based on observed emergence.
    fn update_predictors(&mut self, layers: &[Layer], predicted: f32, actual: f32) {
        let error = actual - predicted;
        let adjustment = error * self.config.learning_rate;

        // Update pairwise weights
        for (i, &l1) in layers.iter().enumerate() {
            for &l2 in layers.iter().skip(i + 1) {
                let (low, high) = if l1.number() < l2.number() {
                    (l1, l2)
                } else {
                    (l2, l1)
                };

                let weight = self
                    .predictors
                    .pairwise_weights
                    .entry((low, high))
                    .or_insert(0.5);
                *weight = (*weight + adjustment).clamp(0.0, 1.0);
            }
        }

        // Update higher-order weight if applicable
        if layers.len() >= 3 {
            let mut key: Vec<Layer> = layers.to_vec();
            key.sort_by_key(|l| l.number());

            let weight = self
                .predictors
                .higher_order_weights
                .entry(key)
                .or_insert(0.0);
            *weight = (*weight + adjustment).clamp(-1.0, 1.0);
        }
    }

    /// Get emergence statistics.
    pub fn stats(&self) -> &EmergenceStats {
        &self.history.stats
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        format!(
            "=== EMERGENCE FRAMEWORK ===\n\
             Total Measurements: {}\n\
             Average Emergence: {:.4}\n\
             Max Emergence: {:.4}\n\
             Avg Prediction Error: {:.4}\n\
             Significant Events: {} ({:.1}%)",
            self.history.stats.total_measurements,
            self.history.stats.average_emergence,
            self.history.stats.max_emergence,
            self.history.stats.avg_prediction_error,
            self.history.stats.significant_events,
            if self.history.stats.total_measurements > 0 {
                self.history.stats.significant_events as f32
                    / self.history.stats.total_measurements as f32
                    * 100.0
            } else {
                0.0
            }
        )
    }

    /// Reset the framework.
    pub fn reset(&mut self) {
        self.history = EmergenceHistory::default();
        self.predictors = EmergencePredictors::default();
    }
}

impl Default for EmergenceFramework {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in seconds.
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_creation() {
        let framework = EmergenceFramework::new();
        assert_eq!(framework.stats().total_measurements, 0);
    }

    #[test]
    fn test_emergence_prediction() {
        let framework = EmergenceFramework::new();

        let layers = vec![Layer::BasePhysics, Layer::ExtendedPhysics];
        let prediction = framework.predict(&layers);

        assert!(prediction > 0.0);
        assert!(prediction <= 1.0);
    }

    #[test]
    fn test_emergence_analysis() {
        let mut framework = EmergenceFramework::new();

        let mut result = StackProcessResult::empty();
        result.layer_confidences.insert(Layer::BasePhysics, 0.8);
        result
            .layer_confidences
            .insert(Layer::GaiaConsciousness, 0.9);
        result.total_amplification = 1.3;
        result.converged = true;
        result.combined_confidence = 0.85; // Set combined confidence for test

        let analysis = framework.analyze(&result);

        // The confidence should be set from combined_confidence
        assert!(analysis.confidence > 0.0);
        assert_eq!(framework.stats().total_measurements, 1);
    }

    #[test]
    fn test_mechanism_classification() {
        let framework = EmergenceFramework::new();

        let mut result = StackProcessResult::empty();
        result.layer_confidences.insert(Layer::BasePhysics, 0.8);
        result.total_amplification = 1.6;

        let mechanism = framework.classify_mechanism(&result, &HashMap::new());
        assert_eq!(mechanism, EmergenceMechanism::Resonance);
    }
}
