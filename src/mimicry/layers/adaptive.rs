//! Phase 5 Adaptive Systems
//!
//! Three new capabilities that make the multiplicative integration system
//! self-tuning and prevent trivial saturation:
//!
//! - **AdaptiveConfidenceCap**: Dynamically adjusts confidence ceiling based on
//!   scenario difficulty, preventing the compounding model from trivially
//!   saturating to max_confidence on every input.
//!
//! - **DynamicBridgeWeighting**: Learns per-bridge weights from outcomes,
//!   implementing an attention-like mechanism where useful bridges are
//!   strengthened and underperforming bridges are dampened.
//!
//! - **OnlineLearningSystem**: Continuously updates layer and bridge parameters
//!   from a feedback signal, enabling the system to improve over time.

use std::collections::HashMap;

use super::layer::Layer;

// =============================================================================
// Adaptive Confidence Cap
// =============================================================================

/// Dynamically adjusts the confidence ceiling based on input characteristics
/// and historical performance.
///
/// Instead of a fixed `max_confidence = 2.0`, the cap adapts:
/// - **Low-confidence inputs** get a lower cap (harder to saturate)
/// - **High-difficulty tasks** (measured by layer disagreement) lower the cap
/// - **Consistent saturation** triggers cap reduction (anti-saturation)
/// - The cap slowly recovers over time toward the configured baseline
///
/// Formula:
/// ```text
/// effective_cap = base_cap × difficulty_factor × saturation_penalty
/// difficulty_factor = 1.0 - (1.0 - input_confidence) × difficulty_sensitivity
/// saturation_penalty = 1.0 - (saturation_rate × anti_saturation_strength)
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveConfidenceCap {
    config: AdaptiveCapConfig,
    /// Rolling window of recent cap-hit events (true = hit cap, false = didn't).
    saturation_history: Vec<bool>,
    /// Running saturation rate (0.0 to 1.0).
    saturation_rate: f32,
    /// Current effective cap override (None = use formula).
    current_cap: Option<f32>,
    /// Total samples processed.
    total_samples: u64,
}

/// Configuration for the adaptive confidence cap.
#[derive(Debug, Clone)]
pub struct AdaptiveCapConfig {
    /// Base maximum confidence (the "ceiling" that gets adjusted).
    pub base_cap: f32,
    /// Minimum allowed cap (never go below this).
    pub min_cap: f32,
    /// How sensitive the cap is to input difficulty (0.0 = ignore, 1.0 = full).
    pub difficulty_sensitivity: f32,
    /// How strongly saturation history reduces the cap (0.0 to 1.0).
    pub anti_saturation_strength: f32,
    /// Window size for saturation history.
    pub history_window: usize,
    /// Recovery rate: how fast the cap recovers when not saturating (per sample).
    pub recovery_rate: f32,
    /// Whether to enable the adaptive cap.
    pub enabled: bool,
}

impl Default for AdaptiveCapConfig {
    fn default() -> Self {
        Self {
            base_cap: 2.0,
            min_cap: 0.5,
            difficulty_sensitivity: 0.3,
            anti_saturation_strength: 0.5,
            history_window: 20,
            recovery_rate: 0.02,
            enabled: false, // Off by default for backward compat
        }
    }
}

impl AdaptiveConfidenceCap {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(AdaptiveCapConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: AdaptiveCapConfig) -> Self {
        Self {
            config,
            saturation_history: Vec::new(),
            saturation_rate: 0.0,
            current_cap: None,
            total_samples: 0,
        }
    }

    /// Calculate the effective confidence cap for a given input.
    ///
    /// `input_confidence` is the raw confidence of the initial input.
    /// `layer_variance` is the variance across layer confidences (0 = agreement, high = disagreement).
    pub fn effective_cap(&self, input_confidence: f32, layer_variance: f32) -> f32 {
        if !self.config.enabled {
            return self.config.base_cap;
        }

        // Factor 1: Input difficulty — lower confidence = harder = lower cap
        let difficulty_factor =
            1.0 - (1.0 - input_confidence.clamp(0.0, 1.0)) * self.config.difficulty_sensitivity;

        // Factor 2: Layer disagreement — high variance = uncertain = lower cap
        let disagreement_factor = 1.0 / (1.0 + layer_variance * 2.0);

        // Factor 3: Anti-saturation — frequent cap-hitting lowers the cap
        let saturation_penalty = 1.0 - self.saturation_rate * self.config.anti_saturation_strength;

        let raw_cap =
            self.config.base_cap * difficulty_factor * disagreement_factor * saturation_penalty;

        raw_cap.clamp(self.config.min_cap, self.config.base_cap)
    }

    /// Record whether the last processing result hit the cap.
    ///
    /// Call this after each `process_bidirectional` with:
    /// - `combined_confidence`: the output confidence
    /// - `cap_used`: the cap that was in effect
    pub fn record_result(&mut self, combined_confidence: f32, cap_used: f32) {
        self.total_samples += 1;

        // Did it saturate? (within 1% of cap)
        let saturated = (combined_confidence / cap_used) > 0.99;

        self.saturation_history.push(saturated);
        if self.saturation_history.len() > self.config.history_window {
            self.saturation_history.remove(0);
        }

        // Update saturation rate
        if !self.saturation_history.is_empty() {
            let hits = self.saturation_history.iter().filter(|&&s| s).count();
            self.saturation_rate = hits as f32 / self.saturation_history.len() as f32;
        }

        // Recovery: if not saturating, slowly reduce the penalty
        if !saturated {
            self.saturation_rate = (self.saturation_rate - self.config.recovery_rate).max(0.0);
        }
    }

    /// Get the current saturation rate (0.0 to 1.0).
    pub fn saturation_rate(&self) -> f32 {
        self.saturation_rate
    }

    /// Get total samples processed.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Get the base cap (uncorrected).
    pub fn base_cap(&self) -> f32 {
        self.config.base_cap
    }

    /// Check if the adaptive cap is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Reset history and state.
    pub fn reset(&mut self) {
        self.saturation_history.clear();
        self.saturation_rate = 0.0;
        self.current_cap = None;
        self.total_samples = 0;
    }
}

impl Default for AdaptiveConfidenceCap {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Dynamic Bridge Weighting
// =============================================================================

/// Learns per-bridge weights from outcomes, implementing an attention-like
/// mechanism for the bridge network.
///
/// Each bridge gets a weight in [0.1, 2.0] that modulates its amplification
/// contribution. Weights are updated via exponential moving average of
/// the bridge's "usefulness" — measured by how much it improves combined
/// confidence when active.
///
/// Formula:
/// ```text
/// usefulness = (confidence_with_bridge - confidence_without) / confidence_without
/// weight_new = weight_old × (1 - lr) + (1.0 + usefulness) × lr
/// ```
#[derive(Debug, Clone)]
pub struct DynamicBridgeWeighting {
    config: DynamicWeightConfig,
    /// Per-bridge weights, keyed by (source_layer_num, target_layer_num).
    weights: HashMap<(u8, u8), BridgeWeight>,
    /// Total updates performed.
    total_updates: u64,
}

/// Configuration for dynamic bridge weighting.
#[derive(Debug, Clone)]
pub struct DynamicWeightConfig {
    /// Learning rate for weight updates (0.0 to 1.0).
    pub learning_rate: f32,
    /// Minimum allowed weight.
    pub min_weight: f32,
    /// Maximum allowed weight.
    pub max_weight: f32,
    /// Initial weight for new bridges.
    pub initial_weight: f32,
    /// Decay rate: how fast weights regress to initial (per update).
    pub decay_rate: f32,
    /// Whether to enable dynamic weighting.
    pub enabled: bool,
}

impl Default for DynamicWeightConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            min_weight: 0.1,
            max_weight: 2.0,
            initial_weight: 1.0,
            decay_rate: 0.001,
            enabled: false,
        }
    }
}

/// Weight state for a single bridge.
#[derive(Debug, Clone)]
pub struct BridgeWeight {
    /// Current weight value.
    pub weight: f32,
    /// Running average usefulness.
    pub avg_usefulness: f32,
    /// Number of updates.
    pub updates: u64,
    /// Peak weight achieved.
    pub peak_weight: f32,
}

impl BridgeWeight {
    fn new(initial: f32) -> Self {
        Self {
            weight: initial,
            avg_usefulness: 0.0,
            updates: 0,
            peak_weight: initial,
        }
    }
}

impl DynamicBridgeWeighting {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(DynamicWeightConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: DynamicWeightConfig) -> Self {
        Self {
            config,
            weights: HashMap::new(),
            total_updates: 0,
        }
    }

    /// Get the weight for a bridge between two layers.
    pub fn weight_for(&self, source: Layer, target: Layer) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }
        let key = Self::key(source, target);
        self.weights
            .get(&key)
            .map(|bw| bw.weight)
            .unwrap_or(self.config.initial_weight)
    }

    /// Update the weight for a bridge based on its observed usefulness.
    ///
    /// `usefulness` should be positive if the bridge helped, negative if it hurt.
    /// Typical range: [-1.0, 1.0].
    pub fn update(&mut self, source: Layer, target: Layer, usefulness: f32) {
        if !self.config.enabled {
            return;
        }

        let key = Self::key(source, target);
        let bw = self
            .weights
            .entry(key)
            .or_insert_with(|| BridgeWeight::new(self.config.initial_weight));

        bw.updates += 1;
        self.total_updates += 1;

        // EMA update of usefulness
        let lr = self.config.learning_rate;
        bw.avg_usefulness = bw.avg_usefulness * (1.0 - lr) + usefulness * lr;

        // Weight update: shift toward (1.0 + avg_usefulness)
        let target_weight =
            (1.0 + bw.avg_usefulness).clamp(self.config.min_weight, self.config.max_weight);
        bw.weight = bw.weight * (1.0 - lr) + target_weight * lr;

        // Apply decay toward initial
        bw.weight = bw.weight * (1.0 - self.config.decay_rate)
            + self.config.initial_weight * self.config.decay_rate;

        // Clamp
        bw.weight = bw
            .weight
            .clamp(self.config.min_weight, self.config.max_weight);

        // Track peak
        if bw.weight > bw.peak_weight {
            bw.peak_weight = bw.weight;
        }
    }

    /// Update all bridge weights based on a processing result.
    ///
    /// `per_bridge_deltas` maps (source_num, target_num) to the confidence
    /// delta attributed to that bridge.
    pub fn update_from_result(&mut self, per_bridge_deltas: &HashMap<(u8, u8), f32>) {
        for (&(src, tgt), &delta) in per_bridge_deltas {
            if let (Some(source), Some(target)) = (num_to_layer(src), num_to_layer(tgt)) {
                self.update(source, target, delta);
            }
        }
    }

    /// Get all current weights.
    pub fn all_weights(&self) -> &HashMap<(u8, u8), BridgeWeight> {
        &self.weights
    }

    /// Get total updates performed.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Reset all weights to initial.
    pub fn reset(&mut self) {
        self.weights.clear();
        self.total_updates = 0;
    }

    /// Canonical key: always (min, max) layer numbers.
    fn key(source: Layer, target: Layer) -> (u8, u8) {
        let (a, b) = (source.number(), target.number());
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }
}

impl Default for DynamicBridgeWeighting {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Online Learning System
// =============================================================================

/// Continuous online learning that updates layer and system parameters
/// based on feedback signals.
///
/// Tracks per-layer "effectiveness" scores and uses them to:
/// - Adjust per-layer amplification factors
/// - Recommend layer enable/disable decisions
/// - Track learning curves and detect plateaus
///
/// Each layer's effectiveness is updated via:
/// ```text
/// effectiveness_new = effectiveness_old × (1 - lr) + observed_contribution × lr
/// ```
#[derive(Debug, Clone)]
pub struct OnlineLearningSystem {
    config: OnlineLearningConfig,
    /// Per-layer effectiveness scores.
    layer_effectiveness: HashMap<Layer, LayerEffectiveness>,
    /// Global learning state.
    global_state: GlobalLearningState,
}

/// Configuration for online learning.
#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    /// Learning rate for effectiveness updates.
    pub learning_rate: f32,
    /// Minimum effectiveness before recommending disable.
    pub disable_threshold: f32,
    /// How many samples before learning kicks in.
    pub warmup_samples: u64,
    /// Plateau detection: if improvement < this for N samples, declare plateau.
    pub plateau_threshold: f32,
    /// Plateau detection: number of samples with low improvement.
    pub plateau_window: usize,
    /// Whether to enable online learning.
    pub enabled: bool,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.02,
            disable_threshold: 0.05,
            warmup_samples: 10,
            plateau_threshold: 0.001,
            plateau_window: 20,
            enabled: false,
        }
    }
}

/// Effectiveness tracking for a single layer.
#[derive(Debug, Clone)]
pub struct LayerEffectiveness {
    /// Running effectiveness score (0.0 to 1.0+).
    pub effectiveness: f32,
    /// Number of observations.
    pub observations: u64,
    /// Running average confidence contribution.
    pub avg_confidence_contribution: f32,
    /// Peak confidence contribution.
    pub peak_contribution: f32,
    /// Recent improvement deltas (for plateau detection).
    pub recent_improvements: Vec<f32>,
    /// Whether this layer is recommended to be active.
    pub recommended_active: bool,
}

impl LayerEffectiveness {
    fn new() -> Self {
        Self {
            effectiveness: 0.5, // Start neutral
            observations: 0,
            avg_confidence_contribution: 0.0,
            peak_contribution: 0.0,
            recent_improvements: Vec::new(),
            recommended_active: true,
        }
    }
}

/// Global learning state.
#[derive(Debug, Clone)]
pub struct GlobalLearningState {
    /// Total learning updates.
    pub total_updates: u64,
    /// Whether the system has completed warmup.
    pub warmup_complete: bool,
    /// Global average confidence (running).
    pub avg_confidence: f32,
    /// Best confidence achieved.
    pub best_confidence: f32,
    /// Whether a plateau has been detected.
    pub plateau_detected: bool,
    /// Recent global improvements.
    recent_global_improvements: Vec<f32>,
}

impl Default for GlobalLearningState {
    fn default() -> Self {
        Self {
            total_updates: 0,
            warmup_complete: false,
            avg_confidence: 0.0,
            best_confidence: 0.0,
            plateau_detected: false,
            recent_global_improvements: Vec::new(),
        }
    }
}

impl OnlineLearningSystem {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(OnlineLearningConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: OnlineLearningConfig) -> Self {
        Self {
            config,
            layer_effectiveness: HashMap::new(),
            global_state: GlobalLearningState::default(),
        }
    }

    /// Record a layer's contribution to the overall result.
    ///
    /// `layer_confidence` is this layer's output confidence.
    /// `combined_confidence` is the overall system confidence.
    pub fn observe_layer(&mut self, layer: Layer, layer_confidence: f32, combined_confidence: f32) {
        if !self.config.enabled {
            return;
        }

        let eff = self
            .layer_effectiveness
            .entry(layer)
            .or_insert_with(LayerEffectiveness::new);

        eff.observations += 1;

        // Contribution: how much does this layer's confidence align with combined?
        let contribution = if combined_confidence > 0.0 {
            (layer_confidence / combined_confidence).min(2.0)
        } else {
            0.0
        };

        // EMA update
        let lr = self.config.learning_rate;
        eff.effectiveness = eff.effectiveness * (1.0 - lr) + contribution * lr;
        eff.avg_confidence_contribution =
            eff.avg_confidence_contribution * (1.0 - lr) + layer_confidence * lr;

        if layer_confidence > eff.peak_contribution {
            eff.peak_contribution = layer_confidence;
        }

        // Track improvement for plateau detection
        let improvement = (contribution - eff.effectiveness).abs();
        eff.recent_improvements.push(improvement);
        if eff.recent_improvements.len() > self.config.plateau_window {
            eff.recent_improvements.remove(0);
        }

        // Recommendation: disable if effectiveness below threshold after warmup
        eff.recommended_active = eff.effectiveness >= self.config.disable_threshold
            || eff.observations < self.config.warmup_samples;
    }

    /// Update global state with overall result.
    pub fn observe_global(&mut self, combined_confidence: f32) {
        if !self.config.enabled {
            return;
        }

        self.global_state.total_updates += 1;

        // Check warmup
        if self.global_state.total_updates >= self.config.warmup_samples {
            self.global_state.warmup_complete = true;
        }

        // Update running average
        let lr = self.config.learning_rate;
        let improvement = combined_confidence - self.global_state.avg_confidence;
        self.global_state.avg_confidence =
            self.global_state.avg_confidence * (1.0 - lr) + combined_confidence * lr;

        if combined_confidence > self.global_state.best_confidence {
            self.global_state.best_confidence = combined_confidence;
        }

        // Plateau detection
        self.global_state
            .recent_global_improvements
            .push(improvement.abs());
        if self.global_state.recent_global_improvements.len() > self.config.plateau_window {
            self.global_state.recent_global_improvements.remove(0);
        }

        if self.global_state.recent_global_improvements.len() >= self.config.plateau_window {
            let avg_improvement: f32 = self
                .global_state
                .recent_global_improvements
                .iter()
                .sum::<f32>()
                / self.global_state.recent_global_improvements.len() as f32;
            self.global_state.plateau_detected = avg_improvement < self.config.plateau_threshold;
        }
    }

    /// Get the recommended amplification factor for a layer.
    ///
    /// Layers with high effectiveness get boosted; low effectiveness get dampened.
    pub fn recommended_amplification(&self, layer: Layer) -> f32 {
        if !self.config.enabled || !self.global_state.warmup_complete {
            return 1.0;
        }

        self.layer_effectiveness
            .get(&layer)
            .map(|eff| {
                // Map effectiveness [0, 1+] to amplification [0.5, 1.5]
                (0.5 + eff.effectiveness).clamp(0.5, 1.5)
            })
            .unwrap_or(1.0)
    }

    /// Check if a layer is recommended to be active.
    pub fn is_layer_recommended(&self, layer: Layer) -> bool {
        if !self.config.enabled || !self.global_state.warmup_complete {
            return true;
        }
        self.layer_effectiveness
            .get(&layer)
            .map(|eff| eff.recommended_active)
            .unwrap_or(true)
    }

    /// Get the effectiveness score for a layer.
    pub fn layer_effectiveness(&self, layer: Layer) -> Option<&LayerEffectiveness> {
        self.layer_effectiveness.get(&layer)
    }

    /// Get all layer effectiveness scores.
    pub fn all_effectiveness(&self) -> &HashMap<Layer, LayerEffectiveness> {
        &self.layer_effectiveness
    }

    /// Get global learning state.
    pub fn global_state(&self) -> &GlobalLearningState {
        &self.global_state
    }

    /// Check if plateau is detected.
    pub fn is_plateau(&self) -> bool {
        self.global_state.plateau_detected
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Reset all learning state.
    pub fn reset(&mut self) {
        self.layer_effectiveness.clear();
        self.global_state = GlobalLearningState::default();
    }
}

impl Default for OnlineLearningSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helper
// =============================================================================

/// Convert a layer number (1-8) to a Layer enum variant.
fn num_to_layer(n: u8) -> Option<Layer> {
    match n {
        1 => Some(Layer::BasePhysics),
        2 => Some(Layer::ExtendedPhysics),
        3 => Some(Layer::CrossDomain),
        4 => Some(Layer::GaiaConsciousness),
        5 => Some(Layer::MultilingualProcessing),
        6 => Some(Layer::CollaborativeLearning),
        7 => Some(Layer::ExternalApis),
        8 => Some(Layer::PreCognitiveVisualization),
        _ => None,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AdaptiveConfidenceCap tests ----

    #[test]
    fn test_adaptive_cap_disabled_returns_base() {
        let cap = AdaptiveConfidenceCap::new();
        assert!(!cap.is_enabled());
        assert_eq!(cap.effective_cap(0.5, 0.1), 2.0);
    }

    #[test]
    fn test_adaptive_cap_enabled_reduces_for_low_input() {
        let cap = AdaptiveConfidenceCap::with_config(AdaptiveCapConfig {
            enabled: true,
            base_cap: 2.0,
            min_cap: 0.5,
            difficulty_sensitivity: 0.5,
            ..Default::default()
        });

        let high_input_cap = cap.effective_cap(0.9, 0.0);
        let low_input_cap = cap.effective_cap(0.2, 0.0);

        assert!(
            low_input_cap < high_input_cap,
            "Low input ({}) should get lower cap than high input ({})",
            low_input_cap,
            high_input_cap
        );
    }

    #[test]
    fn test_adaptive_cap_reduces_for_high_variance() {
        let cap = AdaptiveConfidenceCap::with_config(AdaptiveCapConfig {
            enabled: true,
            base_cap: 2.0,
            min_cap: 0.5,
            ..Default::default()
        });

        let low_var_cap = cap.effective_cap(0.8, 0.01);
        let high_var_cap = cap.effective_cap(0.8, 0.5);

        assert!(
            high_var_cap < low_var_cap,
            "High variance ({}) should get lower cap than low variance ({})",
            high_var_cap,
            low_var_cap
        );
    }

    #[test]
    fn test_adaptive_cap_anti_saturation() {
        let mut cap = AdaptiveConfidenceCap::with_config(AdaptiveCapConfig {
            enabled: true,
            base_cap: 2.0,
            min_cap: 0.5,
            anti_saturation_strength: 0.5,
            history_window: 5,
            ..Default::default()
        });

        // Simulate 5 saturating results
        for _ in 0..5 {
            cap.record_result(1.99, 2.0);
        }

        assert!(cap.saturation_rate() > 0.9);

        // Cap should now be reduced
        let effective = cap.effective_cap(0.8, 0.0);
        assert!(
            effective < 2.0,
            "After saturation, cap ({}) should be < base (2.0)",
            effective
        );
    }

    #[test]
    fn test_adaptive_cap_never_below_min() {
        let cap = AdaptiveConfidenceCap::with_config(AdaptiveCapConfig {
            enabled: true,
            base_cap: 2.0,
            min_cap: 0.5,
            difficulty_sensitivity: 1.0,
            anti_saturation_strength: 1.0,
            ..Default::default()
        });

        let effective = cap.effective_cap(0.01, 10.0);
        assert!(
            effective >= 0.5,
            "Cap ({}) should never go below min_cap (0.5)",
            effective
        );
    }

    #[test]
    fn test_adaptive_cap_reset() {
        let mut cap = AdaptiveConfidenceCap::with_config(AdaptiveCapConfig {
            enabled: true,
            ..Default::default()
        });

        cap.record_result(1.99, 2.0);
        cap.record_result(1.99, 2.0);
        assert!(cap.total_samples() > 0);

        cap.reset();
        assert_eq!(cap.total_samples(), 0);
        assert_eq!(cap.saturation_rate(), 0.0);
    }

    // ---- DynamicBridgeWeighting tests ----

    #[test]
    fn test_bridge_weighting_disabled_returns_one() {
        let bw = DynamicBridgeWeighting::new();
        assert!(!bw.is_enabled());
        assert_eq!(
            bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics),
            1.0
        );
    }

    #[test]
    fn test_bridge_weighting_positive_update() {
        let mut bw = DynamicBridgeWeighting::with_config(DynamicWeightConfig {
            enabled: true,
            learning_rate: 0.1,
            ..Default::default()
        });

        let initial = bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics);

        // Positive usefulness should increase weight
        for _ in 0..10 {
            bw.update(Layer::BasePhysics, Layer::ExtendedPhysics, 0.5);
        }

        let after = bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics);
        assert!(
            after > initial,
            "Positive updates should increase weight: {} -> {}",
            initial,
            after
        );
    }

    #[test]
    fn test_bridge_weighting_negative_update() {
        let mut bw = DynamicBridgeWeighting::with_config(DynamicWeightConfig {
            enabled: true,
            learning_rate: 0.1,
            ..Default::default()
        });

        // Initialize with some positive first
        bw.update(Layer::BasePhysics, Layer::ExtendedPhysics, 0.0);
        let baseline = bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics);

        // Negative usefulness should decrease weight
        for _ in 0..20 {
            bw.update(Layer::BasePhysics, Layer::ExtendedPhysics, -0.5);
        }

        let after = bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics);
        assert!(
            after < baseline,
            "Negative updates should decrease weight: {} -> {}",
            baseline,
            after
        );
    }

    #[test]
    fn test_bridge_weighting_clamped() {
        let mut bw = DynamicBridgeWeighting::with_config(DynamicWeightConfig {
            enabled: true,
            learning_rate: 0.5,
            min_weight: 0.1,
            max_weight: 2.0,
            ..Default::default()
        });

        // Extreme positive
        for _ in 0..100 {
            bw.update(Layer::BasePhysics, Layer::ExtendedPhysics, 5.0);
        }
        let w = bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics);
        assert!(w <= 2.0, "Weight ({}) should be <= max (2.0)", w);

        // Extreme negative
        for _ in 0..100 {
            bw.update(Layer::CrossDomain, Layer::GaiaConsciousness, -5.0);
        }
        let w = bw.weight_for(Layer::CrossDomain, Layer::GaiaConsciousness);
        assert!(w >= 0.1, "Weight ({}) should be >= min (0.1)", w);
    }

    #[test]
    fn test_bridge_weighting_symmetric_key() {
        let mut bw = DynamicBridgeWeighting::with_config(DynamicWeightConfig {
            enabled: true,
            ..Default::default()
        });

        bw.update(Layer::ExtendedPhysics, Layer::BasePhysics, 0.3);

        // Should be accessible from either direction
        let w1 = bw.weight_for(Layer::BasePhysics, Layer::ExtendedPhysics);
        let w2 = bw.weight_for(Layer::ExtendedPhysics, Layer::BasePhysics);
        assert_eq!(w1, w2, "Symmetric lookup should return same weight");
    }

    #[test]
    fn test_bridge_weighting_reset() {
        let mut bw = DynamicBridgeWeighting::with_config(DynamicWeightConfig {
            enabled: true,
            ..Default::default()
        });

        bw.update(Layer::BasePhysics, Layer::ExtendedPhysics, 0.5);
        assert!(bw.total_updates() > 0);

        bw.reset();
        assert_eq!(bw.total_updates(), 0);
        assert!(bw.all_weights().is_empty());
    }

    // ---- OnlineLearningSystem tests ----

    #[test]
    fn test_online_learning_disabled() {
        let ol = OnlineLearningSystem::new();
        assert!(!ol.is_enabled());
        assert_eq!(ol.recommended_amplification(Layer::BasePhysics), 1.0);
        assert!(ol.is_layer_recommended(Layer::BasePhysics));
    }

    #[test]
    fn test_online_learning_observation() {
        let mut ol = OnlineLearningSystem::with_config(OnlineLearningConfig {
            enabled: true,
            warmup_samples: 2,
            learning_rate: 0.1,
            ..Default::default()
        });

        ol.observe_layer(Layer::BasePhysics, 0.8, 1.0);
        ol.observe_layer(Layer::BasePhysics, 0.9, 1.0);
        ol.observe_global(1.0);

        let eff = ol.layer_effectiveness(Layer::BasePhysics);
        assert!(eff.is_some());
        assert_eq!(eff.unwrap().observations, 2);
    }

    #[test]
    fn test_online_learning_warmup() {
        let mut ol = OnlineLearningSystem::with_config(OnlineLearningConfig {
            enabled: true,
            warmup_samples: 5,
            ..Default::default()
        });

        // Before warmup, always recommend active
        assert!(ol.is_layer_recommended(Layer::BasePhysics));

        // Even after low-effectiveness observation, still active during warmup
        ol.observe_layer(Layer::BasePhysics, 0.01, 1.0);
        assert!(ol.is_layer_recommended(Layer::BasePhysics));

        // Complete warmup
        for _ in 0..5 {
            ol.observe_global(1.0);
        }
        assert!(ol.global_state().warmup_complete);
    }

    #[test]
    fn test_online_learning_plateau_detection() {
        let mut ol = OnlineLearningSystem::with_config(OnlineLearningConfig {
            enabled: true,
            warmup_samples: 1,
            plateau_threshold: 0.001,
            plateau_window: 5,
            learning_rate: 0.5, // Fast learning so EMA converges quickly
            ..Default::default()
        });

        // Feed identical results to trigger plateau
        for _ in 0..20 {
            ol.observe_global(1.0);
        }

        assert!(
            ol.is_plateau(),
            "Should detect plateau after identical results"
        );
    }

    #[test]
    fn test_online_learning_recommended_amplification() {
        let mut ol = OnlineLearningSystem::with_config(OnlineLearningConfig {
            enabled: true,
            warmup_samples: 2,
            learning_rate: 0.5, // Fast learning for test
            ..Default::default()
        });

        // Warmup
        for _ in 0..3 {
            ol.observe_global(1.0);
        }

        // High-effectiveness layer
        for _ in 0..10 {
            ol.observe_layer(Layer::BasePhysics, 1.5, 1.0);
        }
        let high_amp = ol.recommended_amplification(Layer::BasePhysics);

        // Low-effectiveness layer
        for _ in 0..10 {
            ol.observe_layer(Layer::ExternalApis, 0.1, 1.0);
        }
        let low_amp = ol.recommended_amplification(Layer::ExternalApis);

        assert!(
            high_amp > low_amp,
            "High-effectiveness layer ({}) should get higher amp than low ({})",
            high_amp,
            low_amp
        );
    }

    #[test]
    fn test_online_learning_reset() {
        let mut ol = OnlineLearningSystem::with_config(OnlineLearningConfig {
            enabled: true,
            warmup_samples: 1,
            ..Default::default()
        });

        ol.observe_layer(Layer::BasePhysics, 0.8, 1.0);
        ol.observe_global(0.8);
        assert!(ol.global_state().total_updates > 0);

        ol.reset();
        assert_eq!(ol.global_state().total_updates, 0);
        assert!(ol.all_effectiveness().is_empty());
    }

    // ---- Helper tests ----

    #[test]
    fn test_num_to_layer() {
        assert_eq!(num_to_layer(1), Some(Layer::BasePhysics));
        assert_eq!(num_to_layer(8), Some(Layer::PreCognitiveVisualization));
        assert_eq!(num_to_layer(0), None);
        assert_eq!(num_to_layer(9), None);
    }
}
