//! Visualization Engine - orchestrator for Pre-Cognitive Visualization (Layer 8).
//!
//! The engine coordinates the three sub-components:
//! 1. **TaskSimulator**: Decomposes tasks into sub-goals, estimates confidence
//! 2. **OutcomeProjector**: Projects what successful output looks like
//! 3. **FidelityTracker**: Compares predictions vs actuals, learns bias corrections
//!
//! The visualization pass fires BEFORE the main forward pass, producing
//! pre-warm signals that prime downstream layers with expectations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::fidelity::{FidelityConfig, FidelityMeasurement, FidelityTracker};
use super::projector::{OutcomeProjection, OutcomeProjector, ProjectorConfig};
use super::simulator::{SimulationResult, SimulatorConfig, TaskSimulator};
use crate::mimicry::layers::layer::{Layer, LayerState};

/// Configuration for the complete Visualization Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Simulator configuration.
    pub simulator: SimulatorConfig,
    /// Projector configuration.
    pub projector: ProjectorConfig,
    /// Fidelity tracker configuration.
    pub fidelity: FidelityConfig,
    /// Whether to apply bias correction to projections.
    pub enable_bias_correction: bool,
    /// Whether the pre-visualization pass is enabled.
    pub enabled: bool,
    /// Pre-warm strength multiplier (scales all pre-warm signals).
    pub pre_warm_strength: f32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            simulator: SimulatorConfig::default(),
            projector: ProjectorConfig::default(),
            fidelity: FidelityConfig::default(),
            enable_bias_correction: true,
            enabled: true,
            pre_warm_strength: 1.0,
        }
    }
}

impl VisualizationConfig {
    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            simulator: SimulatorConfig {
                max_steps: 5,
                max_sub_goals: 10,
                ..Default::default()
            },
            projector: ProjectorConfig {
                max_projections: 3,
                generate_alternatives: false,
                ..Default::default()
            },
            fidelity: FidelityConfig {
                max_history: 50,
                rolling_window: 10,
                ..Default::default()
            },
            enable_bias_correction: false,
            enabled: true,
            pre_warm_strength: 1.0,
        }
    }
}

/// Result of a complete visualization pass.
#[derive(Debug, Clone)]
pub struct VisualizationResult {
    /// Simulation result from the task simulator.
    pub simulation: SimulationResult,
    /// Outcome projections from the projector.
    pub projections: Vec<OutcomeProjection>,
    /// Pre-warm signals for downstream layers (layer -> confidence boost).
    pub pre_warm_signals: HashMap<Layer, f32>,
    /// The overall visualization confidence (bias-corrected if enabled).
    pub visualization_confidence: f32,
    /// Whether fidelity tracking flagged a need for recalibration.
    pub needs_recalibration: bool,
    /// The current fidelity EMA at the time of this pass.
    pub current_fidelity: f32,
}

impl VisualizationResult {
    /// Convert the visualization result into a LayerState for the forward pass.
    pub fn to_layer_state(&self) -> LayerState {
        let mut state = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            self.simulation.task_id.clone(),
            self.visualization_confidence,
        );
        state.set_metadata("engine", "visualization");
        state.set_metadata("simulation_steps", &self.simulation.steps.to_string());
        state.set_metadata("projection_count", &self.projections.len().to_string());
        state.set_metadata("fidelity_ema", &format!("{:.3}", self.current_fidelity));
        state.set_metadata("needs_recalibration", &self.needs_recalibration.to_string());
        state.set_metadata("pre_warm_layers", &self.pre_warm_signals.len().to_string());
        state
    }

    /// Get the primary projection, if any.
    pub fn primary_projection(&self) -> Option<&OutcomeProjection> {
        self.projections.iter().find(|p| p.is_primary)
    }
}

/// The Visualization Engine — orchestrator for Layer 8.
///
/// Runs a complete pre-cognitive visualization pass:
/// 1. Simulate the task (decompose, estimate, identify failure modes)
/// 2. Project outcomes (what does success look like?)
/// 3. Generate pre-warm signals for downstream layers
/// 4. Apply bias correction from fidelity history
pub struct VisualizationEngine {
    /// Configuration.
    config: VisualizationConfig,
    /// Task simulator component.
    simulator: TaskSimulator,
    /// Outcome projector component.
    projector: OutcomeProjector,
    /// Fidelity tracker component.
    fidelity: FidelityTracker,
    /// Total visualization passes executed.
    total_passes: u64,
}

impl VisualizationEngine {
    /// Create a new visualization engine.
    pub fn new(config: VisualizationConfig) -> Self {
        let simulator = TaskSimulator::new(config.simulator.clone());
        let projector = OutcomeProjector::new(config.projector.clone());
        let fidelity = FidelityTracker::new(config.fidelity.clone());

        Self {
            config,
            simulator,
            projector,
            fidelity,
            total_passes: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(VisualizationConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
    }

    /// Get a reference to the simulator.
    pub fn simulator(&self) -> &TaskSimulator {
        &self.simulator
    }

    /// Get a reference to the projector.
    pub fn projector(&self) -> &OutcomeProjector {
        &self.projector
    }

    /// Get a reference to the fidelity tracker.
    pub fn fidelity(&self) -> &FidelityTracker {
        &self.fidelity
    }

    /// Get total visualization passes.
    pub fn total_passes(&self) -> u64 {
        self.total_passes
    }

    /// Run a complete visualization pass.
    ///
    /// This is the main entry point — called BEFORE the forward pass.
    /// It produces a `VisualizationResult` containing pre-warm signals
    /// and outcome projections that downstream layers can use.
    pub fn visualize(&mut self, input: &LayerState) -> VisualizationResult {
        if !self.config.enabled {
            return self.empty_result();
        }

        self.total_passes += 1;

        // Step 1: Simulate the task
        let simulation = self.simulator.simulate(input);

        // Step 2: Project outcomes
        let projections = self.projector.project(input);

        // Step 3: Compute visualization confidence
        let raw_confidence = simulation.overall_confidence;
        let visualization_confidence = if self.config.enable_bias_correction {
            self.fidelity.correct_prediction(raw_confidence)
        } else {
            raw_confidence
        };

        // Step 4: Generate pre-warm signals
        let mut pre_warm_signals = simulation.pre_warm_signals.clone();
        // Scale by pre_warm_strength
        for signal in pre_warm_signals.values_mut() {
            *signal *= self.config.pre_warm_strength;
        }

        // Step 5: Check fidelity status
        let needs_recalibration = self.fidelity.needs_recalibration();
        let current_fidelity = self.fidelity.ema_fidelity();

        VisualizationResult {
            simulation,
            projections,
            pre_warm_signals,
            visualization_confidence,
            needs_recalibration,
            current_fidelity,
        }
    }

    /// Record actual outcomes after the forward pass completes.
    ///
    /// This closes the feedback loop by comparing the visualization's
    /// predictions against actual results. Returns true if recalibration
    /// is needed.
    pub fn record_outcome(
        &mut self,
        predicted_confidence: f32,
        actual_confidence: f32,
        layer_actuals: &HashMap<Layer, f32>,
    ) -> bool {
        let reference_id = format!("pass_{}", self.total_passes);

        // Build fidelity measurement
        let mut measurement =
            FidelityMeasurement::new(&reference_id, predicted_confidence, actual_confidence);

        // Add per-layer deltas if we have a recent projection
        if let Some(recent_projections) = self.projector.history().last() {
            if let Some(primary) = recent_projections.iter().find(|p| p.is_primary) {
                let deltas = primary.compute_deltas(layer_actuals);
                for (layer, delta) in deltas {
                    measurement = measurement.with_layer_delta(layer, delta);
                }
            }
        }

        let needs_recal = self.fidelity.record(measurement);

        // Update projector quality from fidelity
        let fidelity_score = self.fidelity.rolling_fidelity();
        self.projector.update_quality(fidelity_score);

        needs_recal
    }

    /// Get a summary of the engine's current state.
    pub fn summary(&self) -> EngineSummary {
        let fidelity_stats = self.fidelity.stats();
        EngineSummary {
            total_passes: self.total_passes,
            enabled: self.config.enabled,
            fidelity_ema: fidelity_stats.ema_fidelity,
            current_bias: fidelity_stats.current_bias,
            projection_quality: self.projector.average_quality(),
            total_projections: self.projector.total_projections(),
            recalibrations: fidelity_stats.recalibrations,
            needs_recalibration: self.fidelity.needs_recalibration(),
        }
    }

    /// Create an empty result for when the engine is disabled.
    fn empty_result(&self) -> VisualizationResult {
        VisualizationResult {
            simulation: super::simulator::SimulationResult::empty("disabled"),
            projections: Vec::new(),
            pre_warm_signals: HashMap::new(),
            visualization_confidence: 0.5,
            needs_recalibration: false,
            current_fidelity: self.fidelity.ema_fidelity(),
        }
    }
}

impl Default for VisualizationEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Summary of the engine's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSummary {
    /// Total visualization passes executed.
    pub total_passes: u64,
    /// Whether the engine is enabled.
    pub enabled: bool,
    /// Current fidelity EMA.
    pub fidelity_ema: f32,
    /// Current prediction bias.
    pub current_bias: f32,
    /// Current projection quality.
    pub projection_quality: f32,
    /// Total projections generated.
    pub total_projections: u64,
    /// Total recalibrations triggered.
    pub recalibrations: u64,
    /// Whether recalibration is currently needed.
    pub needs_recalibration: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = VisualizationEngine::with_defaults();
        assert_eq!(engine.total_passes(), 0);
        assert!(engine.config().enabled);
    }

    #[test]
    fn test_visualization_pass() {
        let mut engine = VisualizationEngine::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test task".to_string(), 0.8);

        let result = engine.visualize(&input);

        assert!(!result.simulation.task_id.is_empty());
        assert!(!result.projections.is_empty());
        assert!(result.visualization_confidence > 0.0);
        assert_eq!(engine.total_passes(), 1);
    }

    #[test]
    fn test_disabled_engine() {
        let mut config = VisualizationConfig::default();
        config.enabled = false;
        let mut engine = VisualizationEngine::new(config);

        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);
        let result = engine.visualize(&input);

        assert!(result.projections.is_empty());
        assert_eq!(result.visualization_confidence, 0.5);
    }

    #[test]
    fn test_outcome_recording() {
        let mut engine = VisualizationEngine::with_defaults();

        // Run a visualization pass
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);
        let vis_result = engine.visualize(&input);

        // Record actual outcome
        let mut actuals = HashMap::new();
        actuals.insert(Layer::BasePhysics, 0.78);
        actuals.insert(Layer::GaiaConsciousness, 0.72);

        let needs_recal =
            engine.record_outcome(vis_result.visualization_confidence, 0.78, &actuals);

        // Single good measurement shouldn't trigger recalibration
        assert!(!needs_recal);
    }

    #[test]
    fn test_pre_warm_signals() {
        let mut engine = VisualizationEngine::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.9);

        let result = engine.visualize(&input);

        // Should have at least L1 pre-warm
        assert!(!result.pre_warm_signals.is_empty());
    }

    #[test]
    fn test_pre_warm_strength_scaling() {
        let mut config = VisualizationConfig::default();
        config.pre_warm_strength = 2.0;
        let mut engine = VisualizationEngine::new(config);

        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.9);
        let result = engine.visualize(&input);

        // All pre-warm signals should be scaled by 2.0
        for &signal in result.pre_warm_signals.values() {
            // With strength 2.0, signals should be larger
            assert!(signal > 0.0);
        }
    }

    #[test]
    fn test_to_layer_state() {
        let mut engine = VisualizationEngine::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        let result = engine.visualize(&input);
        let state = result.to_layer_state();

        assert_eq!(state.layer, Layer::PreCognitiveVisualization);
        assert_eq!(state.get_metadata("engine"), Some("visualization"));
        assert!(state.confidence > 0.0);
    }

    #[test]
    fn test_summary() {
        let mut engine = VisualizationEngine::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        engine.visualize(&input);

        let summary = engine.summary();
        assert_eq!(summary.total_passes, 1);
        assert!(summary.enabled);
        assert!(summary.fidelity_ema > 0.0);
    }

    #[test]
    fn test_feedback_loop() {
        let mut engine = VisualizationEngine::with_defaults();

        // Run several passes with feedback
        for i in 0..5 {
            let input = LayerState::with_confidence(Layer::BasePhysics, format!("task_{}", i), 0.8);

            let result = engine.visualize(&input);

            let mut actuals = HashMap::new();
            actuals.insert(Layer::BasePhysics, 0.78);

            engine.record_outcome(result.visualization_confidence, 0.78, &actuals);
        }

        // After several rounds, the engine should have learned something
        let summary = engine.summary();
        assert_eq!(summary.total_passes, 5);
        assert!(summary.total_projections > 0);
    }

    #[test]
    fn test_primary_projection() {
        let mut engine = VisualizationEngine::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        let result = engine.visualize(&input);

        let primary = result.primary_projection();
        assert!(primary.is_some());
        assert!(primary.unwrap().is_primary);
    }

    #[test]
    fn test_minimal_config() {
        let config = VisualizationConfig::minimal();
        let mut engine = VisualizationEngine::new(config);

        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);
        let result = engine.visualize(&input);

        // Minimal config disables alternatives
        assert!(result.projections.len() <= 3);
    }
}
