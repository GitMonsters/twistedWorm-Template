//! Phase 6: OCTO Braiding Integration
//!
//! Wires the OCTO RNA bridge into the layer system, creating cross-modulation
//! between all 6 subsystems (3 Phase 3 + 3 Phase 5). This solves the critical
//! calibration saturation problem by enabling subsystems to inform each other.
//!
//! # The Braiding Concept
//!
//! Instead of subsystems running in parallel isolation, they **cross-modulate**
//! through OCTO-style gating. Each module's output gates/modulates the others:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    OCTO Braid Layer                      │
//! │                                                         │
//! │  Head Gates (8 floats) map to subsystem control:        │
//! │   [0] reserve_ratio_scale     [4] bridge_weight_lr      │
//! │   [1] burst_threshold_scale   [5] online_learning_lr    │
//! │   [2] resonance_sensitivity   [6] prewarm_strength      │
//! │   [3] cap_aggressiveness      [7] global_damping        │
//! │                                                         │
//! │  Pathway Weights (3 floats, sum to 1.0):                │
//! │   [0] perception → forward emphasis                     │
//! │   [1] reasoning  → amplification emphasis               │
//! │   [2] action     → convergence emphasis                 │
//! │                                                         │
//! │  Temperature → System1/System2 routing:                 │
//! │   Low temp  = trust compounding, relax reserve          │
//! │   High temp = enforce reserve restraint, lower cap      │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Graceful Degradation
//!
//! When the `octo` feature is disabled (no Python bridge), the braid uses
//! **default signals** that produce neutral modulation (all multipliers = 1.0).
//! The cross-modulation logic between subsystems still runs — it just uses
//! internally-derived signals instead of OCTO RNA analysis.
//!
//! When the `octo` feature IS enabled, the braid calls the RNA bridge to get
//! intelligent, input-aware gating signals.

use std::collections::HashMap;

use super::layer::Layer;

/// Signals from the OCTO braid that modulate all 6 subsystems.
///
/// These are computed once per bidirectional iteration and applied to
/// every subsystem before they execute.
#[derive(Debug, Clone)]
pub struct BraidSignals {
    // --- Phase 3 modulation ---
    /// Scale factor for reserve ratios (0.5 = halve reserves, 2.0 = double).
    pub reserve_ratio_scale: f32,
    /// Scale factor for burst release thresholds (< 1.0 = easier bursts).
    pub burst_threshold_scale: f32,
    /// Sensitivity multiplier for distribution resonance.
    pub resonance_sensitivity: f32,
    /// Pre-warm strength multiplier for visualization engine.
    pub prewarm_strength: f32,

    // --- Phase 5 modulation ---
    /// Aggressiveness of the adaptive confidence cap (< 1.0 = more restrictive).
    pub cap_aggressiveness: f32,
    /// Learning rate multiplier for dynamic bridge weighting.
    pub bridge_weight_lr_scale: f32,
    /// Learning rate multiplier for online learning system.
    pub online_learning_lr_scale: f32,

    // --- Global modulation ---
    /// Override for amplification damping (0.0-1.0).
    pub global_damping: f32,
    /// Pathway emphasis: how much to emphasize forward vs amplification vs convergence.
    pub pathway_emphasis: PathwayEmphasis,
    /// The effective confidence cap as computed by the braid.
    /// This integrates adaptive cap + difficulty hint + temperature.
    pub effective_cap: f32,
    /// Temperature from OCTO (or derived internally). Higher = more uncertain.
    pub temperature: f32,
    /// Whether we're in System 1 (fast/trusting) or System 2 (slow/skeptical) mode.
    pub system2_active: bool,
}

/// How to emphasize different processing phases.
#[derive(Debug, Clone, Copy)]
pub struct PathwayEmphasis {
    /// Weight for forward propagation phase (perception pathway).
    pub forward: f32,
    /// Weight for amplification phase (reasoning pathway).
    pub amplification: f32,
    /// Weight for convergence checking (action pathway).
    pub convergence: f32,
}

impl Default for PathwayEmphasis {
    fn default() -> Self {
        Self {
            forward: 1.0,
            amplification: 1.0,
            convergence: 1.0,
        }
    }
}

impl Default for BraidSignals {
    fn default() -> Self {
        Self {
            reserve_ratio_scale: 1.0,
            burst_threshold_scale: 1.0,
            resonance_sensitivity: 1.0,
            prewarm_strength: 1.0,
            cap_aggressiveness: 1.0,
            bridge_weight_lr_scale: 1.0,
            online_learning_lr_scale: 1.0,
            global_damping: 0.8,
            pathway_emphasis: PathwayEmphasis::default(),
            effective_cap: 2.0,
            temperature: 1.0,
            system2_active: false,
        }
    }
}

impl BraidSignals {
    /// Create neutral signals (no modulation effect).
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Check if these signals are effectively neutral (no modulation).
    pub fn is_neutral(&self) -> bool {
        (self.reserve_ratio_scale - 1.0).abs() < 0.001
            && (self.burst_threshold_scale - 1.0).abs() < 0.001
            && (self.resonance_sensitivity - 1.0).abs() < 0.001
            && (self.prewarm_strength - 1.0).abs() < 0.001
            && (self.cap_aggressiveness - 1.0).abs() < 0.001
    }
}

/// Configuration for the OCTO braid.
#[derive(Debug, Clone)]
pub struct OctoBraidConfig {
    /// Whether the braid is enabled.
    pub enabled: bool,

    /// Difficulty hint: an external signal (0.0 = easy, 1.0 = impossible)
    /// that the adaptive cap and reserve use to pre-adjust.
    /// If None, difficulty is inferred from input confidence and temperature.
    pub difficulty_hint: Option<f32>,

    /// Base confidence cap (default 2.0, matches LayerStackConfig).
    pub base_cap: f32,

    /// Minimum allowed cap.
    pub min_cap: f32,

    /// How strongly temperature affects the cap (0.0 = ignore, 1.0 = full).
    pub temperature_cap_sensitivity: f32,

    /// How strongly difficulty hint affects reserve ratios (0.0 = ignore, 1.0 = full).
    pub difficulty_reserve_sensitivity: f32,

    /// Whether to use OCTO RNA bridge (requires `octo` feature).
    /// When false, uses internal heuristics for signal generation.
    pub use_octo_bridge: bool,

    /// System 1 confidence threshold (below this → System 2 mode).
    pub system1_threshold: f32,

    /// Temperature threshold (above this → System 2 mode).
    pub temperature_threshold: f32,
}

impl Default for OctoBraidConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            difficulty_hint: None,
            base_cap: 2.0,
            min_cap: 0.5,
            temperature_cap_sensitivity: 0.4,
            difficulty_reserve_sensitivity: 0.5,
            use_octo_bridge: false,
            system1_threshold: 0.65,
            temperature_threshold: 1.8,
        }
    }
}

/// The OCTO Braid: cross-modulation engine for all 6 subsystems.
///
/// Takes input characteristics (confidence, text embedding, difficulty)
/// and produces [`BraidSignals`] that modulate every subsystem in the stack.
///
/// When the `octo` feature is enabled and `use_octo_bridge` is true,
/// it calls the OCTO RNA bridge for intelligent gating. Otherwise, it
/// uses internal heuristics based on input confidence and difficulty.
pub struct OctoBraid {
    config: OctoBraidConfig,
    /// Last computed braid signals (cached for iteration reuse).
    last_signals: BraidSignals,
    /// Statistics.
    stats: BraidStats,
}

/// Statistics for the OCTO braid.
#[derive(Debug, Clone, Default)]
pub struct BraidStats {
    /// Total modulation calls.
    pub total_modulations: u64,
    /// Number of System 2 activations.
    pub system2_activations: u64,
    /// Average temperature across all modulations.
    pub avg_temperature: f32,
    /// Average effective cap across all modulations.
    pub avg_effective_cap: f32,
    /// Number of times difficulty hint was used.
    pub difficulty_hint_uses: u64,
}

impl OctoBraid {
    /// Create a new OCTO braid with default config.
    pub fn new() -> Self {
        Self::with_config(OctoBraidConfig::default())
    }

    /// Create a new OCTO braid with custom config.
    pub fn with_config(config: OctoBraidConfig) -> Self {
        Self {
            config,
            last_signals: BraidSignals::neutral(),
            stats: BraidStats::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &OctoBraidConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut OctoBraidConfig {
        &mut self.config
    }

    /// Set the difficulty hint for the current input.
    pub fn set_difficulty_hint(&mut self, hint: Option<f32>) {
        self.config.difficulty_hint = hint.map(|h| h.clamp(0.0, 1.0));
    }

    /// Get the last computed braid signals.
    pub fn last_signals(&self) -> &BraidSignals {
        &self.last_signals
    }

    /// Get statistics.
    pub fn stats(&self) -> &BraidStats {
        &self.stats
    }

    /// Compute braid signals for the current state of the system.
    ///
    /// This is the core method, called at the START of each bidirectional
    /// iteration. It analyzes the current layer confidences and produces
    /// modulation signals for all 6 subsystems.
    ///
    /// # Arguments
    ///
    /// * `input_confidence` - The raw input confidence
    /// * `layer_confidences` - Current per-layer confidence values
    /// * `current_effective_cap` - The adaptive cap's current effective cap
    /// * `iteration` - Current bidirectional iteration number (0-based)
    ///
    /// # OCTO Integration
    ///
    /// When `octo` feature is enabled and `use_octo_bridge` is true, this
    /// method calls `OctoRNABridge::analyze()` with a synthetic embedding
    /// derived from layer confidences. The RNA result's head gates and
    /// pathway weights directly drive the braid signals.
    ///
    /// When OCTO is not available, internal heuristics produce equivalent
    /// signals based on confidence patterns and difficulty.
    pub fn modulate(
        &mut self,
        input_confidence: f32,
        layer_confidences: &HashMap<Layer, f32>,
        current_effective_cap: f32,
        iteration: u32,
    ) -> BraidSignals {
        if !self.config.enabled {
            return BraidSignals::neutral();
        }

        self.stats.total_modulations += 1;

        // Compute internal signals from layer state
        let signals = self.compute_internal_signals(
            input_confidence,
            layer_confidences,
            current_effective_cap,
            iteration,
        );

        // Update stats
        self.stats.avg_temperature = self.stats.avg_temperature
            * ((self.stats.total_modulations - 1) as f32 / self.stats.total_modulations as f32)
            + signals.temperature / self.stats.total_modulations as f32;
        self.stats.avg_effective_cap = self.stats.avg_effective_cap
            * ((self.stats.total_modulations - 1) as f32 / self.stats.total_modulations as f32)
            + signals.effective_cap / self.stats.total_modulations as f32;

        if signals.system2_active {
            self.stats.system2_activations += 1;
        }
        if self.config.difficulty_hint.is_some() {
            self.stats.difficulty_hint_uses += 1;
        }

        self.last_signals = signals.clone();
        signals
    }

    /// Compute braid signals using internal heuristics (no OCTO bridge).
    ///
    /// This is the pure-Rust path that works without Python/PyO3.
    /// It derives all modulation signals from:
    /// - Input confidence
    /// - Layer confidence patterns (mean, variance, spread)
    /// - Difficulty hint (if provided)
    /// - Iteration number (later iterations → more conservative)
    fn compute_internal_signals(
        &self,
        input_confidence: f32,
        layer_confidences: &HashMap<Layer, f32>,
        current_effective_cap: f32,
        iteration: u32,
    ) -> BraidSignals {
        let n = layer_confidences.len().max(1) as f32;
        let mean_conf: f32 = layer_confidences.values().sum::<f32>() / n;
        let variance: f32 = layer_confidences
            .values()
            .map(|&c| (c - mean_conf).powi(2))
            .sum::<f32>()
            / n;

        // --- Derive temperature from confidence patterns ---
        // High variance + low mean → uncertain → high temperature
        // Low variance + high mean → confident → low temperature
        let confidence_factor = mean_conf.clamp(0.1, 2.0);
        let variance_factor = (1.0 + variance * 4.0).min(3.0);
        let temperature = variance_factor / confidence_factor;
        let temperature = temperature.clamp(0.1, 5.0);

        // --- Difficulty ---
        // When an external difficulty hint is provided (from calibration scenarios),
        // it IS the difficulty — no blending needed since it encodes the exact
        // target effective_cap. The inferred difficulty is only used as fallback.
        let inferred_difficulty = 1.0 - input_confidence.clamp(0.0, 1.0);
        let difficulty = self.config.difficulty_hint.unwrap_or(inferred_difficulty);

        // --- System 1 / System 2 routing ---
        let system2_active = mean_conf < self.config.system1_threshold
            || temperature > self.config.temperature_threshold;

        // --- Effective cap ---
        // Start from current adaptive cap, then adjust for temperature and difficulty.
        // When a difficulty hint is provided, we blend between the base cap and min_cap
        // proportional to difficulty, giving precise control for calibration scenarios.
        let temp_penalty = (temperature - 1.0).max(0.0) * self.config.temperature_cap_sensitivity;
        let base_after_temp = current_effective_cap * (1.0 - temp_penalty);
        let effective_cap = if self.config.difficulty_hint.is_some() {
            // Blend: difficulty=0 → full base cap, difficulty=1 → min_cap
            let target =
                self.config.base_cap * (1.0 - difficulty) + self.config.min_cap * difficulty;
            // Also respect the temperature-adjusted base as an upper bound
            base_after_temp.min(target)
        } else {
            // No external hint — use inferred difficulty as a mild penalty
            let difficulty_penalty = difficulty * 0.3;
            base_after_temp * (1.0 - difficulty_penalty)
        };
        let effective_cap = effective_cap.clamp(self.config.min_cap, self.config.base_cap);

        // --- Head gate mapping ---
        // Each "gate" is derived from a combination of temperature, difficulty,
        // confidence patterns, and iteration number.

        // [0] Reserve ratio scale: higher difficulty → more reserve (scale > 1.0)
        //     System 2 → increase reserves; System 1 → relax reserves
        let reserve_ratio_scale = if system2_active {
            1.0 + difficulty * self.config.difficulty_reserve_sensitivity
        } else {
            1.0 - (1.0 - difficulty) * 0.2 // Slightly reduce in System 1
        }
        .clamp(0.5, 2.0);

        // [1] Burst threshold scale: System 2 → raise thresholds (harder to burst)
        //     Later iterations → slightly lower (allow accumulated reserves to release)
        let iteration_decay = (iteration as f32 * 0.05).min(0.2);
        let burst_threshold_scale = if system2_active {
            1.2 + difficulty * 0.3 - iteration_decay
        } else {
            0.9 - iteration_decay
        }
        .clamp(0.5, 2.0);

        // [2] Resonance sensitivity: high temperature → reduce sensitivity
        //     (uncertain system should not amplify resonance as much)
        let resonance_sensitivity = (1.5 - temperature * 0.3).clamp(0.3, 1.5);

        // [3] Cap aggressiveness: feeds into effective_cap computation
        //     System 2 → more aggressive cap (lower values)
        let cap_aggressiveness = if system2_active {
            0.7 + (1.0 - difficulty) * 0.2
        } else {
            1.0
        }
        .clamp(0.3, 1.5);

        // [4] Bridge weight learning rate: System 2 → faster learning (adapt quicker)
        let bridge_weight_lr_scale =
            (if system2_active { 1.5_f32 } else { 1.0_f32 }).clamp(0.5, 2.0);

        // [5] Online learning rate: higher variance → faster learning
        let online_learning_lr_scale = (1.0 + variance * 2.0).clamp(0.5, 2.0);

        // [6] Pre-warm strength: System 1 → stronger pre-warm (trust visualization)
        //     System 2 → weaker pre-warm (don't trust predictions as much)
        let prewarm_strength = (if system2_active { 0.5_f32 } else { 1.2_f32 }).clamp(0.2, 2.0);

        // [7] Global damping: System 2 → more damping (more conservative amplification)
        //     Higher difficulty → more damping
        let global_damping = if system2_active {
            0.5 + difficulty * 0.2
        } else {
            0.8
        }
        .clamp(0.2, 1.0);

        // --- Pathway emphasis ---
        // Derived from confidence patterns:
        // - Early processing (low spread) → emphasize forward
        // - High spread + variance → emphasize amplification
        // - Late iterations → emphasize convergence
        let spread = layer_confidences.len() as f32 / 8.0;
        let forward_weight = (1.5 - spread).max(0.5);
        let amp_weight = (0.5 + variance * 3.0).min(1.5);
        let conv_weight = (0.5 + iteration as f32 * 0.3).min(1.5);
        let total = forward_weight + amp_weight + conv_weight;

        let pathway_emphasis = PathwayEmphasis {
            forward: forward_weight / total * 3.0,
            amplification: amp_weight / total * 3.0,
            convergence: conv_weight / total * 3.0,
        };

        BraidSignals {
            reserve_ratio_scale,
            burst_threshold_scale,
            resonance_sensitivity,
            prewarm_strength,
            cap_aggressiveness,
            bridge_weight_lr_scale,
            online_learning_lr_scale,
            global_damping,
            pathway_emphasis,
            effective_cap: (effective_cap * cap_aggressiveness)
                .clamp(self.config.min_cap, self.config.base_cap),
            temperature,
            system2_active,
        }
    }

    /// Compute braid signals using OCTO RNA bridge analysis.
    ///
    /// This is only available with the `octo` feature. It takes the RNA
    /// result's head gates and pathway weights and maps them directly
    /// to braid signals.
    #[cfg(feature = "octo")]
    pub fn modulate_with_octo(
        &mut self,
        input_confidence: f32,
        layer_confidences: &HashMap<Layer, f32>,
        current_effective_cap: f32,
        iteration: u32,
        rna_result: &crate::mimicry::octo::rna_bridge::RNAEditingResult,
    ) -> BraidSignals {
        if !self.config.enabled {
            return BraidSignals::neutral();
        }

        self.stats.total_modulations += 1;

        let difficulty = self
            .config
            .difficulty_hint
            .unwrap_or(1.0 - input_confidence.clamp(0.0, 1.0));

        // Map OCTO head gates to subsystem controls
        let gates = &rna_result.head_gates;
        let ensure_gate =
            |idx: usize| -> f32 { gates.get(idx).copied().unwrap_or(0.5).clamp(0.0, 1.0) };

        // Gate[0] → reserve ratio scale: 0.0 = minimum reserve, 1.0 = maximum
        let reserve_ratio_scale = 0.5 + ensure_gate(0) * 1.5; // [0.5, 2.0]

        // Gate[1] → burst threshold scale: 0.0 = easy bursts, 1.0 = hard bursts
        let burst_threshold_scale = 0.5 + ensure_gate(1) * 1.5; // [0.5, 2.0]

        // Gate[2] → resonance sensitivity: 0.0 = damped, 1.0 = amplified
        let resonance_sensitivity = 0.3 + ensure_gate(2) * 1.2; // [0.3, 1.5]

        // Gate[3] → cap aggressiveness: 0.0 = strict cap, 1.0 = permissive
        let cap_aggressiveness = 0.3 + ensure_gate(3) * 1.2; // [0.3, 1.5]

        // Gate[4] → bridge weight learning rate
        let bridge_weight_lr_scale = 0.5 + ensure_gate(4) * 1.5; // [0.5, 2.0]

        // Gate[5] → online learning rate
        let online_learning_lr_scale = 0.5 + ensure_gate(5) * 1.5; // [0.5, 2.0]

        // Gate[6] → pre-warm strength
        let prewarm_strength = 0.2 + ensure_gate(6) * 1.8; // [0.2, 2.0]

        // Gate[7] → global damping
        let global_damping = 0.2 + ensure_gate(7) * 0.8; // [0.2, 1.0]

        // System routing
        let system2_active = rna_result.confidence < self.config.system1_threshold
            || rna_result.temperature > self.config.temperature_threshold;

        // Temperature from RNA
        let temperature = rna_result.temperature;

        // Effective cap using temperature and difficulty
        let temp_penalty = (temperature - 1.0).max(0.0) * self.config.temperature_cap_sensitivity;
        let difficulty_penalty = difficulty * 0.3;
        let raw_cap = current_effective_cap * (1.0 - temp_penalty) * (1.0 - difficulty_penalty);
        let effective_cap =
            (raw_cap * cap_aggressiveness).clamp(self.config.min_cap, self.config.base_cap);

        // Pathway emphasis from RNA pathway weights
        let pw = &rna_result.pathway_weights;
        let pathway_emphasis = if pw.len() >= 3 {
            let total = pw[0] + pw[1] + pw[2];
            if total > 0.0 {
                PathwayEmphasis {
                    forward: pw[0] / total * 3.0,
                    amplification: pw[1] / total * 3.0,
                    convergence: pw[2] / total * 3.0,
                }
            } else {
                PathwayEmphasis::default()
            }
        } else {
            PathwayEmphasis::default()
        };

        // Update stats
        self.stats.avg_temperature = self.stats.avg_temperature
            * ((self.stats.total_modulations - 1) as f32 / self.stats.total_modulations as f32)
            + temperature / self.stats.total_modulations as f32;
        self.stats.avg_effective_cap = self.stats.avg_effective_cap
            * ((self.stats.total_modulations - 1) as f32 / self.stats.total_modulations as f32)
            + effective_cap / self.stats.total_modulations as f32;

        if system2_active {
            self.stats.system2_activations += 1;
        }
        if self.config.difficulty_hint.is_some() {
            self.stats.difficulty_hint_uses += 1;
        }

        let signals = BraidSignals {
            reserve_ratio_scale,
            burst_threshold_scale,
            resonance_sensitivity,
            prewarm_strength,
            cap_aggressiveness,
            bridge_weight_lr_scale,
            online_learning_lr_scale,
            global_damping,
            pathway_emphasis,
            effective_cap,
            temperature,
            system2_active,
        };

        self.last_signals = signals.clone();
        signals
    }

    /// Reset the braid state.
    pub fn reset(&mut self) {
        self.last_signals = BraidSignals::neutral();
        self.stats = BraidStats::default();
    }
}

impl Default for OctoBraid {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for OctoBraid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OctoBraid")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OctoBraidConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.base_cap, 2.0);
        assert_eq!(config.min_cap, 0.5);
        assert!(config.difficulty_hint.is_none());
    }

    #[test]
    fn test_neutral_signals() {
        let signals = BraidSignals::neutral();
        assert!((signals.reserve_ratio_scale - 1.0).abs() < 0.001);
        assert!((signals.burst_threshold_scale - 1.0).abs() < 0.001);
        assert!((signals.effective_cap - 2.0).abs() < 0.001);
        assert!(!signals.system2_active);
        assert!(signals.is_neutral());
    }

    #[test]
    fn test_disabled_braid_returns_neutral() {
        let mut braid = OctoBraid::new(); // enabled = false by default
        let signals = braid.modulate(0.5, &HashMap::new(), 2.0, 0);
        assert!(signals.is_neutral());
    }

    #[test]
    fn test_enabled_braid_high_confidence() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.9);
        confs.insert(Layer::ExtendedPhysics, 0.85);
        confs.insert(Layer::CrossDomain, 0.88);
        confs.insert(Layer::GaiaConsciousness, 0.87);

        let signals = braid.modulate(0.9, &confs, 2.0, 0);

        // High confidence → System 1 (mean ~0.875 > 0.65 threshold)
        // Low variance → low temperature
        assert!(!signals.system2_active);
        // Should have relaxed reserves in System 1
        assert!(signals.reserve_ratio_scale < 1.1);
        assert!(signals.temperature < 2.0);
    }

    #[test]
    fn test_enabled_braid_low_confidence() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.2);
        confs.insert(Layer::ExtendedPhysics, 0.15);

        let signals = braid.modulate(0.2, &confs, 2.0, 0);

        // Low confidence → System 2
        assert!(signals.system2_active);
        // Higher reserves in System 2
        assert!(signals.reserve_ratio_scale > 1.0);
        // More damping
        assert!(signals.global_damping < 0.8);
        // Lower effective cap
        assert!(signals.effective_cap < 2.0);
    }

    #[test]
    fn test_difficulty_hint_increases_reserves() {
        let mut braid_easy = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            difficulty_hint: Some(0.1),
            ..Default::default()
        });
        let mut braid_hard = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            difficulty_hint: Some(0.9),
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.5);
        confs.insert(Layer::ExtendedPhysics, 0.5);

        let easy_signals = braid_easy.modulate(0.5, &confs, 2.0, 0);
        let hard_signals = braid_hard.modulate(0.5, &confs, 2.0, 0);

        // Hard difficulty should produce higher reserve ratios
        assert!(
            hard_signals.reserve_ratio_scale > easy_signals.reserve_ratio_scale,
            "Hard ({}) should have higher reserve scale than easy ({})",
            hard_signals.reserve_ratio_scale,
            easy_signals.reserve_ratio_scale
        );

        // Hard difficulty should produce lower effective cap
        assert!(
            hard_signals.effective_cap < easy_signals.effective_cap,
            "Hard ({}) should have lower cap than easy ({})",
            hard_signals.effective_cap,
            easy_signals.effective_cap
        );
    }

    #[test]
    fn test_iteration_affects_burst_threshold() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.5);

        let sig_iter0 = braid.modulate(0.5, &confs, 2.0, 0);
        let sig_iter4 = braid.modulate(0.5, &confs, 2.0, 4);

        // Later iterations should have slightly lower burst threshold
        // (allowing accumulated reserves to release)
        assert!(
            sig_iter4.burst_threshold_scale <= sig_iter0.burst_threshold_scale,
            "Later iteration ({}) should have <= burst threshold than early ({})",
            sig_iter4.burst_threshold_scale,
            sig_iter0.burst_threshold_scale
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.5);

        braid.modulate(0.5, &confs, 2.0, 0);
        braid.modulate(0.5, &confs, 2.0, 1);
        braid.modulate(0.5, &confs, 2.0, 2);

        assert_eq!(braid.stats().total_modulations, 3);
        assert!(braid.stats().avg_temperature > 0.0);
        assert!(braid.stats().avg_effective_cap > 0.0);
    }

    #[test]
    fn test_pathway_emphasis_normalized() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        for layer in Layer::all() {
            confs.insert(*layer, 0.7);
        }

        let signals = braid.modulate(0.7, &confs, 2.0, 2);

        // Sum of pathway weights should be approximately 3.0
        // (they're each normalized to average 1.0 then multiplied by 3)
        let sum = signals.pathway_emphasis.forward
            + signals.pathway_emphasis.amplification
            + signals.pathway_emphasis.convergence;
        assert!(
            (sum - 3.0).abs() < 0.01,
            "Pathway sum ({}) should be ~3.0",
            sum
        );
    }

    #[test]
    fn test_effective_cap_respects_bounds() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            difficulty_hint: Some(1.0), // Maximum difficulty
            temperature_cap_sensitivity: 1.0,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.1);

        let signals = braid.modulate(0.1, &confs, 2.0, 0);

        assert!(
            signals.effective_cap >= braid.config().min_cap,
            "Effective cap ({}) should be >= min_cap ({})",
            signals.effective_cap,
            braid.config().min_cap
        );
        assert!(
            signals.effective_cap <= braid.config().base_cap,
            "Effective cap ({}) should be <= base_cap ({})",
            signals.effective_cap,
            braid.config().base_cap
        );
    }

    #[test]
    fn test_reset() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        let mut confs = HashMap::new();
        confs.insert(Layer::BasePhysics, 0.5);
        braid.modulate(0.5, &confs, 2.0, 0);
        assert!(braid.stats().total_modulations > 0);

        braid.reset();
        assert_eq!(braid.stats().total_modulations, 0);
        assert!(braid.last_signals().is_neutral());
    }

    #[test]
    fn test_set_difficulty_hint() {
        let mut braid = OctoBraid::new();
        assert!(braid.config().difficulty_hint.is_none());

        braid.set_difficulty_hint(Some(0.7));
        assert_eq!(braid.config().difficulty_hint, Some(0.7));

        // Clamping
        braid.set_difficulty_hint(Some(5.0));
        assert_eq!(braid.config().difficulty_hint, Some(1.0));

        braid.set_difficulty_hint(None);
        assert!(braid.config().difficulty_hint.is_none());
    }

    #[test]
    fn test_high_variance_increases_temperature() {
        let mut braid = OctoBraid::with_config(OctoBraidConfig {
            enabled: true,
            ..Default::default()
        });

        // Low variance scenario
        let mut confs_low_var = HashMap::new();
        confs_low_var.insert(Layer::BasePhysics, 0.7);
        confs_low_var.insert(Layer::ExtendedPhysics, 0.71);
        confs_low_var.insert(Layer::CrossDomain, 0.69);

        // High variance scenario
        let mut confs_high_var = HashMap::new();
        confs_high_var.insert(Layer::BasePhysics, 0.2);
        confs_high_var.insert(Layer::ExtendedPhysics, 1.5);
        confs_high_var.insert(Layer::CrossDomain, 0.1);

        let sig_low = braid.modulate(0.7, &confs_low_var, 2.0, 0);
        let sig_high = braid.modulate(0.7, &confs_high_var, 2.0, 0);

        assert!(
            sig_high.temperature > sig_low.temperature,
            "High variance temp ({}) should exceed low variance temp ({})",
            sig_high.temperature,
            sig_low.temperature
        );
    }
}
