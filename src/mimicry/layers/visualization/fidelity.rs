//! Fidelity Tracker for Pre-Cognitive Visualization.
//!
//! Tracks the accuracy of visualization predictions vs actual outcomes.
//! Learns from deltas to improve future simulation and projection quality.
//! This is the feedback loop that makes the visualization system adaptive.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mimicry::layers::layer::Layer;

/// A single fidelity measurement comparing predicted vs actual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityMeasurement {
    /// The task or projection ID this measurement is for.
    pub reference_id: String,
    /// Predicted overall confidence.
    pub predicted_confidence: f32,
    /// Actual achieved confidence.
    pub actual_confidence: f32,
    /// Per-layer deltas (layer number -> delta).
    pub layer_deltas: HashMap<u8, f32>,
    /// Fidelity score: 1.0 = perfect prediction, 0.0 = worst.
    pub fidelity_score: f32,
    /// Timestamp of measurement.
    pub timestamp: u64,
}

impl FidelityMeasurement {
    /// Create a new fidelity measurement.
    pub fn new(reference_id: impl Into<String>, predicted: f32, actual: f32) -> Self {
        let delta = (predicted - actual).abs();
        // Fidelity score: inverse of normalized delta
        let fidelity = (1.0 - delta.min(1.0)).max(0.0);

        Self {
            reference_id: reference_id.into(),
            predicted_confidence: predicted,
            actual_confidence: actual,
            layer_deltas: HashMap::new(),
            fidelity_score: fidelity,
            timestamp: Self::current_time_millis(),
        }
    }

    /// Add a per-layer delta.
    pub fn with_layer_delta(mut self, layer: Layer, delta: f32) -> Self {
        self.layer_deltas.insert(layer.number(), delta);
        self
    }

    /// Check if the prediction was optimistic (predicted > actual).
    pub fn is_optimistic(&self) -> bool {
        self.predicted_confidence > self.actual_confidence
    }

    /// Get the absolute error.
    pub fn absolute_error(&self) -> f32 {
        (self.predicted_confidence - self.actual_confidence).abs()
    }

    /// Get current time in milliseconds.
    fn current_time_millis() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Configuration for the fidelity tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityConfig {
    /// Maximum number of measurements to retain.
    pub max_history: usize,
    /// Window size for computing rolling averages.
    pub rolling_window: usize,
    /// Learning rate for bias correction.
    pub bias_learning_rate: f32,
    /// Threshold below which fidelity triggers recalibration.
    pub recalibration_threshold: f32,
    /// Exponential moving average decay factor.
    pub ema_decay: f32,
}

impl Default for FidelityConfig {
    fn default() -> Self {
        Self {
            max_history: 500,
            rolling_window: 20,
            bias_learning_rate: 0.02,
            recalibration_threshold: 0.5,
            ema_decay: 0.95,
        }
    }
}

/// Statistics about fidelity tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityStats {
    /// Total measurements recorded.
    pub total_measurements: u64,
    /// Current rolling average fidelity.
    pub rolling_fidelity: f32,
    /// Exponential moving average of fidelity.
    pub ema_fidelity: f32,
    /// Current bias (positive = optimistic, negative = pessimistic).
    pub current_bias: f32,
    /// Number of recalibrations triggered.
    pub recalibrations: u64,
    /// Per-layer average deltas.
    pub layer_avg_deltas: HashMap<u8, f32>,
    /// Best fidelity score observed.
    pub best_fidelity: f32,
    /// Worst fidelity score observed.
    pub worst_fidelity: f32,
}

/// The Fidelity Tracker component of the Visualization Engine.
///
/// Compares visualization predictions (from the simulator and projector)
/// against actual outcomes after the forward pass completes. Tracks
/// accuracy over time and provides bias correction signals.
pub struct FidelityTracker {
    /// Configuration.
    config: FidelityConfig,
    /// Measurement history.
    measurements: Vec<FidelityMeasurement>,
    /// Exponential moving average of fidelity.
    ema_fidelity: f32,
    /// Current estimated bias.
    bias: f32,
    /// Total measurements recorded.
    total_measurements: u64,
    /// Recalibration counter.
    recalibrations: u64,
    /// Per-layer cumulative deltas and counts for averaging.
    layer_delta_sums: HashMap<u8, (f32, u64)>,
    /// Best fidelity observed.
    best_fidelity: f32,
    /// Worst fidelity observed.
    worst_fidelity: f32,
}

impl FidelityTracker {
    /// Create a new fidelity tracker.
    pub fn new(config: FidelityConfig) -> Self {
        Self {
            config,
            measurements: Vec::new(),
            ema_fidelity: 0.5,
            bias: 0.0,
            total_measurements: 0,
            recalibrations: 0,
            layer_delta_sums: HashMap::new(),
            best_fidelity: 0.0,
            worst_fidelity: 1.0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FidelityConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &FidelityConfig {
        &self.config
    }

    /// Get the current bias estimate.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Get the current EMA fidelity.
    pub fn ema_fidelity(&self) -> f32 {
        self.ema_fidelity
    }

    /// Record a fidelity measurement.
    ///
    /// This updates the rolling average, EMA, bias estimate, and
    /// triggers recalibration if fidelity drops too low.
    pub fn record(&mut self, measurement: FidelityMeasurement) -> bool {
        let fidelity = measurement.fidelity_score;
        let signed_error = measurement.predicted_confidence - measurement.actual_confidence;

        // Update EMA
        self.ema_fidelity =
            self.config.ema_decay * self.ema_fidelity + (1.0 - self.config.ema_decay) * fidelity;

        // Update bias
        self.bias = self.bias * (1.0 - self.config.bias_learning_rate)
            + signed_error * self.config.bias_learning_rate;

        // Update per-layer deltas
        for (&layer_num, &delta) in &measurement.layer_deltas {
            let entry = self.layer_delta_sums.entry(layer_num).or_insert((0.0, 0));
            entry.0 += delta;
            entry.1 += 1;
        }

        // Track extremes
        if fidelity > self.best_fidelity {
            self.best_fidelity = fidelity;
        }
        if fidelity < self.worst_fidelity {
            self.worst_fidelity = fidelity;
        }

        // Store measurement
        self.measurements.push(measurement);
        self.total_measurements += 1;

        // Trim history
        if self.measurements.len() > self.config.max_history {
            self.measurements.remove(0);
        }

        // Check for recalibration
        let needs_recalibration = self.ema_fidelity < self.config.recalibration_threshold;
        if needs_recalibration {
            self.recalibrations += 1;
        }
        needs_recalibration
    }

    /// Record a simple predicted vs actual measurement.
    pub fn record_simple(
        &mut self,
        reference_id: impl Into<String>,
        predicted: f32,
        actual: f32,
    ) -> bool {
        let measurement = FidelityMeasurement::new(reference_id, predicted, actual);
        self.record(measurement)
    }

    /// Get a bias-corrected confidence prediction.
    ///
    /// Given a raw predicted confidence, applies the learned bias
    /// correction to produce a more accurate estimate.
    pub fn correct_prediction(&self, raw_prediction: f32) -> f32 {
        (raw_prediction - self.bias).clamp(0.0, 1.5)
    }

    /// Get the rolling average fidelity over the recent window.
    pub fn rolling_fidelity(&self) -> f32 {
        let window = self.config.rolling_window.min(self.measurements.len());
        if window == 0 {
            return 0.5;
        }

        let start = self.measurements.len() - window;
        let sum: f32 = self.measurements[start..]
            .iter()
            .map(|m| m.fidelity_score)
            .sum();
        sum / window as f32
    }

    /// Get comprehensive fidelity statistics.
    pub fn stats(&self) -> FidelityStats {
        let layer_avg_deltas: HashMap<u8, f32> = self
            .layer_delta_sums
            .iter()
            .map(|(&layer, &(sum, count))| {
                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                (layer, avg)
            })
            .collect();

        FidelityStats {
            total_measurements: self.total_measurements,
            rolling_fidelity: self.rolling_fidelity(),
            ema_fidelity: self.ema_fidelity,
            current_bias: self.bias,
            recalibrations: self.recalibrations,
            layer_avg_deltas,
            best_fidelity: self.best_fidelity,
            worst_fidelity: self.worst_fidelity,
        }
    }

    /// Get the measurement history.
    pub fn measurements(&self) -> &[FidelityMeasurement] {
        &self.measurements
    }

    /// Check if the system needs recalibration based on current EMA.
    pub fn needs_recalibration(&self) -> bool {
        self.ema_fidelity < self.config.recalibration_threshold
    }

    /// Reset the tracker state (keeps config).
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.ema_fidelity = 0.5;
        self.bias = 0.0;
        self.total_measurements = 0;
        self.recalibrations = 0;
        self.layer_delta_sums.clear();
        self.best_fidelity = 0.0;
        self.worst_fidelity = 1.0;
    }
}

impl Default for FidelityTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fidelity_measurement_creation() {
        let m = FidelityMeasurement::new("test", 0.8, 0.75);
        assert_eq!(m.reference_id, "test");
        assert_eq!(m.predicted_confidence, 0.8);
        assert_eq!(m.actual_confidence, 0.75);
        // |0.8 - 0.75| = 0.05, fidelity = 1.0 - 0.05 = 0.95
        assert!((m.fidelity_score - 0.95).abs() < 0.001);
        assert!(m.is_optimistic());
    }

    #[test]
    fn test_fidelity_perfect_prediction() {
        let m = FidelityMeasurement::new("perfect", 0.8, 0.8);
        assert!((m.fidelity_score - 1.0).abs() < 0.001);
        assert_eq!(m.absolute_error(), 0.0);
    }

    #[test]
    fn test_fidelity_worst_prediction() {
        let m = FidelityMeasurement::new("worst", 1.0, 0.0);
        assert!((m.fidelity_score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_tracker_creation() {
        let tracker = FidelityTracker::with_defaults();
        assert_eq!(tracker.ema_fidelity(), 0.5);
        assert_eq!(tracker.bias(), 0.0);
        assert!(tracker.measurements().is_empty());
    }

    #[test]
    fn test_record_measurement() {
        let mut tracker = FidelityTracker::with_defaults();

        let needs_recal = tracker.record_simple("t1", 0.8, 0.75);
        assert!(!needs_recal); // Single good measurement shouldn't trigger recal

        assert_eq!(tracker.stats().total_measurements, 1);
        assert!(tracker.ema_fidelity() > 0.5); // Should improve from default
    }

    #[test]
    fn test_bias_tracking() {
        let mut tracker = FidelityTracker::with_defaults();

        // Consistently optimistic predictions
        for i in 0..20 {
            tracker.record_simple(format!("t{}", i), 0.9, 0.7);
        }

        // Bias should be positive (optimistic)
        assert!(tracker.bias() > 0.0);
    }

    #[test]
    fn test_bias_correction() {
        let mut tracker = FidelityTracker::with_defaults();

        // Build up positive bias
        for i in 0..10 {
            tracker.record_simple(format!("t{}", i), 0.9, 0.7);
        }

        // Corrected prediction should be lower than raw
        let corrected = tracker.correct_prediction(0.9);
        assert!(corrected < 0.9);
    }

    #[test]
    fn test_rolling_fidelity() {
        let mut tracker = FidelityTracker::with_defaults();

        // Record good predictions
        for i in 0..5 {
            tracker.record_simple(format!("t{}", i), 0.8, 0.78);
        }

        let rolling = tracker.rolling_fidelity();
        assert!(rolling > 0.9); // Close predictions should yield high fidelity
    }

    #[test]
    fn test_recalibration_trigger() {
        let config = FidelityConfig {
            recalibration_threshold: 0.8,
            ema_decay: 0.5, // Fast adaptation for test
            ..Default::default()
        };
        let mut tracker = FidelityTracker::new(config);

        // Record terrible predictions
        for i in 0..10 {
            tracker.record_simple(format!("t{}", i), 0.9, 0.1);
        }

        assert!(tracker.needs_recalibration());
        assert!(tracker.stats().recalibrations > 0);
    }

    #[test]
    fn test_layer_deltas() {
        let mut tracker = FidelityTracker::with_defaults();

        let m = FidelityMeasurement::new("test", 0.8, 0.75)
            .with_layer_delta(Layer::BasePhysics, -0.05)
            .with_layer_delta(Layer::GaiaConsciousness, 0.02);

        tracker.record(m);

        let stats = tracker.stats();
        assert!(stats.layer_avg_deltas.contains_key(&1)); // L1
        assert!(stats.layer_avg_deltas.contains_key(&4)); // L4
    }

    #[test]
    fn test_stats() {
        let mut tracker = FidelityTracker::with_defaults();
        tracker.record_simple("t1", 0.8, 0.75);
        tracker.record_simple("t2", 0.9, 0.85);

        let stats = tracker.stats();
        assert_eq!(stats.total_measurements, 2);
        assert!(stats.ema_fidelity > 0.5);
        assert!(stats.best_fidelity > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut tracker = FidelityTracker::with_defaults();
        tracker.record_simple("t1", 0.8, 0.75);
        tracker.reset();

        assert!(tracker.measurements().is_empty());
        assert_eq!(tracker.ema_fidelity(), 0.5);
        assert_eq!(tracker.bias(), 0.0);
    }
}
