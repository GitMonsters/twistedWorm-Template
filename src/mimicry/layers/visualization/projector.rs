//! Outcome Projector for Pre-Cognitive Visualization.
//!
//! Projects what a successful output looks like before the main forward pass.
//! Produces confidence-scored outcome sketches that downstream layers can
//! compare against during execution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mimicry::layers::layer::{Layer, LayerState};

/// A projected outcome with confidence scoring.
#[derive(Debug, Clone)]
pub struct OutcomeProjection {
    /// Unique identifier for this projection.
    pub id: String,
    /// Confidence that this outcome will be achieved.
    pub confidence: f32,
    /// Expected layer outputs (layer -> expected confidence).
    pub expected_layer_outputs: HashMap<Layer, f32>,
    /// Quality score of the projection itself (meta-confidence).
    pub projection_quality: f32,
    /// Tags describing the expected outcome characteristics.
    pub outcome_tags: Vec<String>,
    /// Whether this is the primary (most likely) projection.
    pub is_primary: bool,
}

impl OutcomeProjection {
    /// Create a new outcome projection.
    pub fn new(id: impl Into<String>, confidence: f32) -> Self {
        Self {
            id: id.into(),
            confidence: confidence.clamp(0.0, 1.0),
            expected_layer_outputs: HashMap::new(),
            projection_quality: 0.5,
            outcome_tags: Vec::new(),
            is_primary: false,
        }
    }

    /// Set an expected layer output.
    pub fn expect_layer(mut self, layer: Layer, confidence: f32) -> Self {
        self.expected_layer_outputs
            .insert(layer, confidence.clamp(0.0, 2.0));
        self
    }

    /// Set projection quality.
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.projection_quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Add an outcome tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.outcome_tags.push(tag.into());
        self
    }

    /// Mark as primary projection.
    pub fn as_primary(mut self) -> Self {
        self.is_primary = true;
        self
    }

    /// Compute the delta between projected and actual layer confidence.
    /// Returns (layer, delta) pairs where delta = actual - expected.
    pub fn compute_deltas(&self, actual: &HashMap<Layer, f32>) -> Vec<(Layer, f32)> {
        self.expected_layer_outputs
            .iter()
            .filter_map(|(layer, expected)| {
                actual
                    .get(layer)
                    .map(|actual_conf| (*layer, actual_conf - expected))
            })
            .collect()
    }
}

/// Configuration for the outcome projector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectorConfig {
    /// Maximum number of outcome projections to generate.
    pub max_projections: usize,
    /// Minimum confidence for a projection to be retained.
    pub min_projection_confidence: f32,
    /// Learning rate for updating projection quality from fidelity feedback.
    pub quality_learning_rate: f32,
    /// Whether to generate alternative (non-primary) projections.
    pub generate_alternatives: bool,
    /// Decay factor for old projection quality scores.
    pub quality_decay: f32,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            max_projections: 5,
            min_projection_confidence: 0.2,
            quality_learning_rate: 0.05,
            generate_alternatives: true,
            quality_decay: 0.99,
        }
    }
}

/// The Outcome Projector component of the Visualization Engine.
///
/// Given an input state (and optionally a simulation result), projects
/// what the final output should look like. Each projection includes
/// expected per-layer confidences so the system can track fidelity
/// during execution.
pub struct OutcomeProjector {
    /// Configuration.
    config: ProjectorConfig,
    /// Running average of projection quality.
    average_quality: f32,
    /// Total projections generated.
    total_projections: u64,
    /// History of projection sets (bounded).
    history: Vec<Vec<OutcomeProjection>>,
}

impl OutcomeProjector {
    /// Create a new outcome projector.
    pub fn new(config: ProjectorConfig) -> Self {
        Self {
            config,
            average_quality: 0.5,
            total_projections: 0,
            history: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ProjectorConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &ProjectorConfig {
        &self.config
    }

    /// Get the running average projection quality.
    pub fn average_quality(&self) -> f32 {
        self.average_quality
    }

    /// Get total projections generated.
    pub fn total_projections(&self) -> u64 {
        self.total_projections
    }

    /// Project outcomes for a given input state.
    ///
    /// Returns a set of outcome projections ranked by confidence.
    /// The first projection is marked as primary.
    pub fn project(&mut self, input: &LayerState) -> Vec<OutcomeProjection> {
        let mut projections = Vec::new();

        // Primary projection: best-case scenario based on input confidence
        let primary = self.build_primary_projection(input);
        projections.push(primary);

        // Alternative projections
        if self.config.generate_alternatives {
            let alternatives = self.build_alternative_projections(input);
            projections.extend(alternatives);
        }

        // Filter by minimum confidence
        projections.retain(|p| p.confidence >= self.config.min_projection_confidence);

        // Truncate to max
        projections.truncate(self.config.max_projections);

        // Update stats
        self.total_projections += projections.len() as u64;

        // Store in history (bounded)
        self.history.push(projections.clone());
        if self.history.len() > 50 {
            self.history.remove(0);
        }

        projections
    }

    /// Project outcomes and return as a LayerState for downstream consumption.
    pub fn project_as_state(&mut self, input: &LayerState) -> LayerState {
        let projections = self.project(input);

        let primary_confidence = projections
            .iter()
            .find(|p| p.is_primary)
            .map(|p| p.confidence)
            .unwrap_or(0.5);

        let mut state = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            projections.len() as u32,
            primary_confidence,
        );
        state.set_metadata("projection_type", "outcome");
        state.set_metadata("projection_count", &projections.len().to_string());
        state.set_metadata("average_quality", &format!("{:.3}", self.average_quality));
        state.add_upstream(input.id.clone());
        state
    }

    /// Update projection quality based on fidelity feedback.
    ///
    /// `fidelity_score` is between 0.0 (projection was completely wrong)
    /// and 1.0 (projection was perfectly accurate).
    pub fn update_quality(&mut self, fidelity_score: f32) {
        let score = fidelity_score.clamp(0.0, 1.0);
        self.average_quality = self.average_quality * (1.0 - self.config.quality_learning_rate)
            + score * self.config.quality_learning_rate;
    }

    /// Build the primary projection.
    fn build_primary_projection(&self, input: &LayerState) -> OutcomeProjection {
        let base_confidence = input.confidence;

        // Expected outputs scale with input confidence and projection quality
        let quality_factor = self.average_quality;

        OutcomeProjection::new(
            format!("proj_primary_{}", self.total_projections),
            base_confidence * quality_factor * 1.1,
        )
        .expect_layer(Layer::BasePhysics, base_confidence * 0.95)
        .expect_layer(Layer::ExtendedPhysics, base_confidence * 0.90)
        .expect_layer(Layer::CrossDomain, base_confidence * 0.85)
        .expect_layer(Layer::GaiaConsciousness, base_confidence * 0.80)
        .expect_layer(Layer::CollaborativeLearning, base_confidence * 0.75)
        .expect_layer(Layer::ExternalApis, base_confidence * 0.70)
        .with_quality(quality_factor)
        .with_tag("primary")
        .with_tag("optimistic")
        .as_primary()
    }

    /// Build alternative projections.
    fn build_alternative_projections(&self, input: &LayerState) -> Vec<OutcomeProjection> {
        let base_confidence = input.confidence;
        let mut alternatives = Vec::new();

        // Conservative projection: lower confidence, higher quality expectation
        let conservative = OutcomeProjection::new(
            format!("proj_conservative_{}", self.total_projections),
            base_confidence * 0.7,
        )
        .expect_layer(Layer::BasePhysics, base_confidence * 0.85)
        .expect_layer(Layer::GaiaConsciousness, base_confidence * 0.60)
        .with_quality(self.average_quality * 1.1)
        .with_tag("conservative");
        alternatives.push(conservative);

        // Failure-aware projection: accounts for potential degradation
        if base_confidence < 0.7 {
            let degraded = OutcomeProjection::new(
                format!("proj_degraded_{}", self.total_projections),
                base_confidence * 0.5,
            )
            .expect_layer(Layer::BasePhysics, base_confidence * 0.70)
            .expect_layer(Layer::GaiaConsciousness, base_confidence * 0.40)
            .with_quality(self.average_quality * 0.8)
            .with_tag("degraded")
            .with_tag("failure_aware");
            alternatives.push(degraded);
        }

        alternatives
    }

    /// Get projection history.
    pub fn history(&self) -> &[Vec<OutcomeProjection>] {
        &self.history
    }
}

impl Default for OutcomeProjector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_projection_creation() {
        let proj = OutcomeProjection::new("test", 0.8)
            .expect_layer(Layer::BasePhysics, 0.9)
            .with_quality(0.7)
            .with_tag("test_tag")
            .as_primary();

        assert_eq!(proj.id, "test");
        assert_eq!(proj.confidence, 0.8);
        assert!(proj.is_primary);
        assert_eq!(proj.projection_quality, 0.7);
        assert!(proj
            .expected_layer_outputs
            .contains_key(&Layer::BasePhysics));
        assert_eq!(proj.outcome_tags, vec!["test_tag"]);
    }

    #[test]
    fn test_projector_creation() {
        let proj = OutcomeProjector::with_defaults();
        assert_eq!(proj.total_projections(), 0);
        assert_eq!(proj.average_quality(), 0.5);
    }

    #[test]
    fn test_basic_projection() {
        let mut proj = OutcomeProjector::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        let results = proj.project(&input);

        assert!(!results.is_empty());
        assert!(results[0].is_primary);
        assert!(results[0].confidence > 0.0);
    }

    #[test]
    fn test_projection_with_alternatives() {
        let mut proj = OutcomeProjector::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.5);

        let results = proj.project(&input);

        // Should have primary + conservative + degraded (since confidence < 0.7)
        assert!(results.len() >= 2);
        let has_primary = results.iter().any(|p| p.is_primary);
        assert!(has_primary);
    }

    #[test]
    fn test_quality_update() {
        let mut proj = OutcomeProjector::with_defaults();
        let initial_quality = proj.average_quality();

        // Good fidelity should improve quality
        proj.update_quality(0.9);
        assert!(proj.average_quality() > initial_quality);

        // Bad fidelity should decrease quality
        let after_good = proj.average_quality();
        proj.update_quality(0.1);
        assert!(proj.average_quality() < after_good);
    }

    #[test]
    fn test_projection_as_state() {
        let mut proj = OutcomeProjector::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        let state = proj.project_as_state(&input);

        assert_eq!(state.layer, Layer::PreCognitiveVisualization);
        assert!(state.confidence > 0.0);
        assert_eq!(state.get_metadata("projection_type"), Some("outcome"));
    }

    #[test]
    fn test_compute_deltas() {
        let proj = OutcomeProjection::new("test", 0.8)
            .expect_layer(Layer::BasePhysics, 0.9)
            .expect_layer(Layer::GaiaConsciousness, 0.7);

        let mut actual = HashMap::new();
        actual.insert(Layer::BasePhysics, 0.85);
        actual.insert(Layer::GaiaConsciousness, 0.75);

        let deltas = proj.compute_deltas(&actual);

        assert_eq!(deltas.len(), 2);
        // BasePhysics: 0.85 - 0.9 = -0.05
        let bp_delta = deltas
            .iter()
            .find(|(l, _)| *l == Layer::BasePhysics)
            .unwrap()
            .1;
        assert!((bp_delta - (-0.05)).abs() < 0.001);
    }

    #[test]
    fn test_projection_history() {
        let mut proj = OutcomeProjector::with_defaults();

        for _ in 0..3 {
            let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);
            proj.project(&input);
        }

        assert_eq!(proj.history().len(), 3);
    }

    #[test]
    fn test_confidence_clamping() {
        let proj = OutcomeProjection::new("test", 1.5);
        assert_eq!(proj.confidence, 1.0); // Clamped to 1.0

        let proj2 = OutcomeProjection::new("test", -0.5);
        assert_eq!(proj2.confidence, 0.0); // Clamped to 0.0
    }
}
