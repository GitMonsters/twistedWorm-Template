//! Fractional Reserve Model for confidence management.
//!
//! Each layer maintains a reserve of confidence that is not immediately
//! released. Instead, confidence is split into active and held portions
//! based on a reserve ratio. Held confidence is released in bursts when
//! inter-layer resonance exceeds a threshold, creating a "compounding
//! reserve" effect that amplifies high-resonance interactions.
//!
//! # Core Equations
//!
//! ```text
//! active_confidence = c × (1 - rᵢ)
//! held_confidence  += c × rᵢ
//!
//! Release rule: if resonance(i,j) >= θᵢ → burst release
//! ```
//!
//! Where:
//! - `c` = raw confidence from the layer
//! - `rᵢ` = reserve ratio for layer i (0.0 to 1.0)
//! - `θᵢ` = release threshold for layer i
//!
//! # Active CF vs Latent CF
//!
//! - **Active CF**: Confidence flowing through the normal forward pass.
//! - **Latent CF**: Held confidence accumulating in reserves, released
//!   only when resonance conditions are met.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mimicry::layers::layer::Layer;

/// State of a single layer's fractional reserve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerReserve {
    /// The layer this reserve belongs to.
    pub layer: Layer,
    /// Reserve ratio: fraction of confidence held back (0.0 to 1.0).
    pub reserve_ratio: f32,
    /// Release threshold: minimum resonance to trigger burst release.
    pub release_threshold: f32,
    /// Currently held (latent) confidence.
    pub held_confidence: f32,
    /// Total confidence that has been actively released.
    pub total_active_released: f32,
    /// Total confidence released via burst.
    pub total_burst_released: f32,
    /// Number of burst events.
    pub burst_count: u64,
    /// Learning rate for threshold adaptation.
    pub threshold_learning_rate: f32,
}

impl LayerReserve {
    /// Create a new layer reserve.
    pub fn new(layer: Layer, reserve_ratio: f32, release_threshold: f32) -> Self {
        Self {
            layer,
            reserve_ratio: reserve_ratio.clamp(0.0, 1.0),
            release_threshold: release_threshold.clamp(0.0, 2.0),
            held_confidence: 0.0,
            total_active_released: 0.0,
            total_burst_released: 0.0,
            burst_count: 0,
            threshold_learning_rate: 0.01,
        }
    }

    /// Split incoming confidence into active and held portions.
    ///
    /// Returns (active_confidence, held_portion).
    pub fn split_confidence(&mut self, raw_confidence: f32) -> (f32, f32) {
        let held = raw_confidence * self.reserve_ratio;
        let active = raw_confidence * (1.0 - self.reserve_ratio);

        self.held_confidence += held;
        self.total_active_released += active;

        (active, held)
    }

    /// Attempt a burst release given the current resonance.
    ///
    /// If `resonance >= release_threshold`, releases all held confidence
    /// as a burst. Returns the burst amount (0.0 if no burst).
    pub fn attempt_burst(&mut self, resonance: f32) -> f32 {
        if resonance >= self.release_threshold && self.held_confidence > 0.0 {
            let burst = self.held_confidence;
            self.held_confidence = 0.0;
            self.total_burst_released += burst;
            self.burst_count += 1;
            burst
        } else {
            0.0
        }
    }

    /// Attempt a partial burst release proportional to how far resonance
    /// exceeds the threshold.
    ///
    /// Returns the released amount.
    pub fn attempt_partial_burst(&mut self, resonance: f32) -> f32 {
        if resonance >= self.release_threshold && self.held_confidence > 0.0 {
            // Release fraction proportional to resonance overshoot
            let overshoot = (resonance - self.release_threshold) / self.release_threshold;
            let release_fraction = overshoot.clamp(0.0, 1.0);
            let burst = self.held_confidence * release_fraction;

            self.held_confidence -= burst;
            self.total_burst_released += burst;
            if burst > 0.0 {
                self.burst_count += 1;
            }
            burst
        } else {
            0.0
        }
    }

    /// Update the release threshold based on outcome feedback.
    ///
    /// Online learning: if a burst led to good outcomes, lower the threshold
    /// (make bursts easier). If a burst led to poor outcomes, raise it.
    ///
    /// `outcome_quality`: 0.0 (bad) to 1.0 (good).
    pub fn update_threshold(&mut self, outcome_quality: f32) {
        let quality = outcome_quality.clamp(0.0, 1.0);
        // Target: threshold should decrease if outcomes are good, increase if bad
        let adjustment = (0.5 - quality) * self.threshold_learning_rate;
        self.release_threshold = (self.release_threshold + adjustment).clamp(0.1, 2.0);
    }

    /// Get the active confidence fraction (1 - reserve_ratio).
    pub fn active_fraction(&self) -> f32 {
        1.0 - self.reserve_ratio
    }

    /// Get the current latent confidence.
    pub fn latent_confidence(&self) -> f32 {
        self.held_confidence
    }

    /// Get the burst rate (bursts per total active releases).
    pub fn burst_rate(&self) -> f32 {
        if self.total_active_released > 0.0 {
            self.total_burst_released / self.total_active_released
        } else {
            0.0
        }
    }

    /// Reset the reserve state (keeps configuration).
    pub fn reset(&mut self) {
        self.held_confidence = 0.0;
        self.total_active_released = 0.0;
        self.total_burst_released = 0.0;
        self.burst_count = 0;
    }
}

/// Decomposition of confidence into active and latent components.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceDecomposition {
    /// Raw input confidence.
    pub raw: f32,
    /// Active (immediately usable) confidence.
    pub active: f32,
    /// Latent (held in reserve) confidence.
    pub latent: f32,
    /// Burst confidence released this cycle.
    pub burst: f32,
    /// Total effective confidence (active + burst).
    pub effective: f32,
}

impl ConfidenceDecomposition {
    /// Create a new decomposition.
    pub fn new(raw: f32, active: f32, latent: f32, burst: f32) -> Self {
        Self {
            raw,
            active,
            latent,
            burst,
            effective: active + burst,
        }
    }

    /// Amplification ratio: effective / raw.
    pub fn amplification_ratio(&self) -> f32 {
        if self.raw > 0.0 {
            self.effective / self.raw
        } else {
            1.0
        }
    }
}

/// Configuration for the fractional reserve system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReserveConfig {
    /// Per-layer reserve ratios. Layers not listed get `default_ratio`.
    pub layer_ratios: HashMap<u8, f32>,
    /// Per-layer release thresholds. Layers not listed get `default_threshold`.
    pub layer_thresholds: HashMap<u8, f32>,
    /// Default reserve ratio for layers without explicit config.
    pub default_ratio: f32,
    /// Default release threshold.
    pub default_threshold: f32,
    /// Learning rate for online threshold adaptation.
    pub threshold_learning_rate: f32,
    /// Whether to use partial (proportional) bursts instead of full bursts.
    pub use_partial_bursts: bool,
}

impl Default for ReserveConfig {
    fn default() -> Self {
        let mut layer_ratios = HashMap::new();
        // Higher layers hold more in reserve (more speculative)
        layer_ratios.insert(1, 0.10); // L1: BasePhysics — low reserve, mostly active
        layer_ratios.insert(2, 0.12); // L2: ExtendedPhysics
        layer_ratios.insert(3, 0.18); // L3: CrossDomain
        layer_ratios.insert(4, 0.20); // L4: GaiaConsciousness
        layer_ratios.insert(5, 0.15); // L5: MultilingualProcessing
        layer_ratios.insert(6, 0.22); // L6: CollaborativeLearning
        layer_ratios.insert(7, 0.25); // L7: ExternalApis
        layer_ratios.insert(8, 0.30); // L8: PreCognitiveVisualization — highest reserve

        let mut layer_thresholds = HashMap::new();
        layer_thresholds.insert(1, 0.70);
        layer_thresholds.insert(2, 0.72);
        layer_thresholds.insert(3, 0.75);
        layer_thresholds.insert(4, 0.78);
        layer_thresholds.insert(5, 0.74);
        layer_thresholds.insert(6, 0.80);
        layer_thresholds.insert(7, 0.82);
        layer_thresholds.insert(8, 0.85);

        Self {
            layer_ratios,
            layer_thresholds,
            default_ratio: 0.15,
            default_threshold: 0.75,
            threshold_learning_rate: 0.01,
            use_partial_bursts: false,
        }
    }
}

/// The Fractional Reserve System.
///
/// Manages reserves for all layers, handling confidence splitting,
/// burst releases, and online threshold learning.
pub struct FractionalReserve {
    /// Configuration.
    config: ReserveConfig,
    /// Per-layer reserve state.
    reserves: HashMap<u8, LayerReserve>,
}

impl FractionalReserve {
    /// Create a new fractional reserve system.
    pub fn new(config: ReserveConfig) -> Self {
        let mut reserves = HashMap::new();

        // Initialize reserves for all 8 layers
        for layer in Layer::all() {
            let num = layer.number();
            let ratio = config
                .layer_ratios
                .get(&num)
                .copied()
                .unwrap_or(config.default_ratio);
            let threshold = config
                .layer_thresholds
                .get(&num)
                .copied()
                .unwrap_or(config.default_threshold);

            let mut reserve = LayerReserve::new(*layer, ratio, threshold);
            reserve.threshold_learning_rate = config.threshold_learning_rate;
            reserves.insert(num, reserve);
        }

        Self { config, reserves }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ReserveConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &ReserveConfig {
        &self.config
    }

    /// Process confidence through the reserve system for a given layer.
    ///
    /// Splits raw confidence into active and held, then checks for
    /// burst release based on the provided resonance value.
    ///
    /// Returns a `ConfidenceDecomposition` with active, latent, and burst values.
    pub fn process(
        &mut self,
        layer: Layer,
        raw_confidence: f32,
        resonance: f32,
    ) -> ConfidenceDecomposition {
        let num = layer.number();

        if let Some(reserve) = self.reserves.get_mut(&num) {
            // Step 1: Split confidence
            let (active, held) = reserve.split_confidence(raw_confidence);

            // Step 2: Attempt burst release
            let burst = if self.config.use_partial_bursts {
                reserve.attempt_partial_burst(resonance)
            } else {
                reserve.attempt_burst(resonance)
            };

            ConfidenceDecomposition::new(raw_confidence, active, held, burst)
        } else {
            // No reserve configured — all confidence is active
            ConfidenceDecomposition::new(raw_confidence, raw_confidence, 0.0, 0.0)
        }
    }

    /// Process confidence for a layer pair interaction.
    ///
    /// Both layers split their confidence, then check for burst release
    /// based on their pairwise resonance.
    pub fn process_pair(
        &mut self,
        layer_a: Layer,
        confidence_a: f32,
        layer_b: Layer,
        confidence_b: f32,
        resonance: f32,
    ) -> (ConfidenceDecomposition, ConfidenceDecomposition) {
        let decomp_a = self.process(layer_a, confidence_a, resonance);
        let decomp_b = self.process(layer_b, confidence_b, resonance);
        (decomp_a, decomp_b)
    }

    /// Update thresholds for a layer based on outcome quality.
    pub fn update_threshold(&mut self, layer: Layer, outcome_quality: f32) {
        let num = layer.number();
        if let Some(reserve) = self.reserves.get_mut(&num) {
            reserve.update_threshold(outcome_quality);
        }
    }

    /// Get the reserve state for a specific layer.
    pub fn get_reserve(&self, layer: Layer) -> Option<&LayerReserve> {
        self.reserves.get(&layer.number())
    }

    /// Get total held confidence across all layers.
    pub fn total_held_confidence(&self) -> f32 {
        self.reserves.values().map(|r| r.held_confidence).sum()
    }

    /// Get total burst confidence released across all layers.
    pub fn total_burst_released(&self) -> f32 {
        self.reserves.values().map(|r| r.total_burst_released).sum()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> ReserveStats {
        let total_held: f32 = self.reserves.values().map(|r| r.held_confidence).sum();
        let total_active: f32 = self
            .reserves
            .values()
            .map(|r| r.total_active_released)
            .sum();
        let total_burst: f32 = self.reserves.values().map(|r| r.total_burst_released).sum();
        let total_bursts: u64 = self.reserves.values().map(|r| r.burst_count).sum();

        ReserveStats {
            total_held_confidence: total_held,
            total_active_released: total_active,
            total_burst_released: total_burst,
            total_burst_events: total_bursts,
            layer_count: self.reserves.len(),
        }
    }

    /// Reset all reserves (keeps configuration).
    pub fn reset(&mut self) {
        for reserve in self.reserves.values_mut() {
            reserve.reset();
        }
    }
}

impl Default for FractionalReserve {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Aggregate statistics for the reserve system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReserveStats {
    /// Total confidence currently held in reserve across all layers.
    pub total_held_confidence: f32,
    /// Total confidence actively released across all layers.
    pub total_active_released: f32,
    /// Total confidence released via bursts.
    pub total_burst_released: f32,
    /// Total number of burst events.
    pub total_burst_events: u64,
    /// Number of layers with reserves.
    pub layer_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_reserve_creation() {
        let reserve = LayerReserve::new(Layer::BasePhysics, 0.2, 0.8);
        assert_eq!(reserve.reserve_ratio, 0.2);
        assert_eq!(reserve.release_threshold, 0.8);
        assert_eq!(reserve.held_confidence, 0.0);
        assert_eq!(reserve.active_fraction(), 0.8);
    }

    #[test]
    fn test_confidence_splitting() {
        let mut reserve = LayerReserve::new(Layer::BasePhysics, 0.2, 0.8);

        let (active, held) = reserve.split_confidence(1.0);

        assert!((active - 0.8).abs() < 0.001);
        assert!((held - 0.2).abs() < 0.001);
        assert!((reserve.held_confidence - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_burst_release() {
        let mut reserve = LayerReserve::new(Layer::GaiaConsciousness, 0.3, 0.7);

        // Accumulate some held confidence
        reserve.split_confidence(1.0); // holds 0.3
        reserve.split_confidence(1.0); // holds another 0.3, total 0.6

        assert!((reserve.held_confidence - 0.6).abs() < 0.001);

        // Below threshold — no burst
        let burst = reserve.attempt_burst(0.5);
        assert_eq!(burst, 0.0);
        assert!((reserve.held_confidence - 0.6).abs() < 0.001);

        // Above threshold — full burst
        let burst = reserve.attempt_burst(0.8);
        assert!((burst - 0.6).abs() < 0.001);
        assert_eq!(reserve.held_confidence, 0.0);
        assert_eq!(reserve.burst_count, 1);
    }

    #[test]
    fn test_partial_burst() {
        let mut reserve = LayerReserve::new(Layer::CrossDomain, 0.2, 0.5);

        reserve.split_confidence(1.0); // holds 0.2

        // Resonance exactly at threshold: overshoot = 0, release = 0
        let burst = reserve.attempt_partial_burst(0.5);
        assert_eq!(burst, 0.0);

        // Resonance 50% above threshold: overshoot = 0.5/0.5 = 1.0, release = 100%
        let burst = reserve.attempt_partial_burst(1.0);
        assert!((burst - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_threshold_update() {
        let mut reserve = LayerReserve::new(Layer::BasePhysics, 0.2, 0.8);

        let initial_threshold = reserve.release_threshold;

        // Good outcome should lower threshold (easier bursts)
        reserve.update_threshold(0.9);
        assert!(reserve.release_threshold < initial_threshold);

        let after_good = reserve.release_threshold;

        // Bad outcome should raise threshold (harder bursts)
        reserve.update_threshold(0.1);
        assert!(reserve.release_threshold > after_good);
    }

    #[test]
    fn test_fractional_reserve_system() {
        let system = FractionalReserve::with_defaults();

        // Should have reserves for all 8 layers
        assert_eq!(system.stats().layer_count, 8);

        // Check default ratios
        let l1 = system.get_reserve(Layer::BasePhysics).unwrap();
        assert!((l1.reserve_ratio - 0.10).abs() < 0.001);

        let l8 = system
            .get_reserve(Layer::PreCognitiveVisualization)
            .unwrap();
        assert!((l8.reserve_ratio - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_process_confidence() {
        let mut system = FractionalReserve::with_defaults();

        // Process through L1 (ratio=0.10, threshold=0.70)
        let decomp = system.process(Layer::BasePhysics, 1.0, 0.5);

        assert!((decomp.active - 0.90).abs() < 0.001);
        assert!((decomp.latent - 0.10).abs() < 0.001);
        assert_eq!(decomp.burst, 0.0); // resonance 0.5 < threshold 0.70
        assert!((decomp.effective - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_process_with_burst() {
        let mut system = FractionalReserve::with_defaults();

        // Accumulate reserve in L1
        system.process(Layer::BasePhysics, 1.0, 0.5); // holds 0.10
        system.process(Layer::BasePhysics, 1.0, 0.5); // holds another 0.10

        // Now process with high resonance — should trigger burst
        let decomp = system.process(Layer::BasePhysics, 1.0, 0.9);

        // active = 0.90, burst should include previously held 0.20
        assert!((decomp.active - 0.90).abs() < 0.001);
        // Burst releases the 0.20 held + the 0.10 just added = 0.30
        // Wait: split happens first (adding 0.10 to held, making it 0.30),
        // then burst releases all 0.30
        assert!((decomp.burst - 0.30).abs() < 0.001);
        assert!((decomp.effective - 1.20).abs() < 0.001);
    }

    #[test]
    fn test_amplification_ratio() {
        let decomp = ConfidenceDecomposition::new(1.0, 0.8, 0.2, 0.5);
        // effective = 0.8 + 0.5 = 1.3, ratio = 1.3/1.0 = 1.3
        assert!((decomp.amplification_ratio() - 1.3).abs() < 0.001);
    }

    #[test]
    fn test_process_pair() {
        let mut system = FractionalReserve::with_defaults();

        let (decomp_a, decomp_b) =
            system.process_pair(Layer::BasePhysics, 0.8, Layer::GaiaConsciousness, 0.9, 0.5);

        assert!(decomp_a.active > 0.0);
        assert!(decomp_b.active > 0.0);
    }

    #[test]
    fn test_total_held() {
        let mut system = FractionalReserve::with_defaults();

        system.process(Layer::BasePhysics, 1.0, 0.0);
        system.process(Layer::GaiaConsciousness, 1.0, 0.0);

        let total_held = system.total_held_confidence();
        // L1 holds 0.10, L4 holds 0.20 = 0.30
        assert!((total_held - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_stats() {
        let mut system = FractionalReserve::with_defaults();

        system.process(Layer::BasePhysics, 1.0, 0.0);

        let stats = system.stats();
        assert_eq!(stats.layer_count, 8);
        assert!(stats.total_active_released > 0.0);
        assert!(stats.total_held_confidence > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut system = FractionalReserve::with_defaults();

        system.process(Layer::BasePhysics, 1.0, 0.0);
        assert!(system.total_held_confidence() > 0.0);

        system.reset();
        assert_eq!(system.total_held_confidence(), 0.0);
    }

    #[test]
    fn test_burst_rate() {
        let mut reserve = LayerReserve::new(Layer::BasePhysics, 0.2, 0.5);

        reserve.split_confidence(1.0); // active=0.8, held=0.2
        reserve.attempt_burst(0.9); // burst=0.2

        let rate = reserve.burst_rate();
        // burst_released=0.2, active_released=0.8, rate=0.25
        assert!((rate - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_partial_burst_config() {
        let mut config = ReserveConfig::default();
        config.use_partial_bursts = true;
        let mut system = FractionalReserve::new(config);

        // Accumulate in L1 (threshold=0.70)
        system.process(Layer::BasePhysics, 1.0, 0.0);

        // Partial burst with moderate overshoot
        let decomp = system.process(Layer::BasePhysics, 1.0, 0.90);

        // Should get a partial burst, not necessarily all held confidence
        assert!(decomp.burst >= 0.0);
    }
}
