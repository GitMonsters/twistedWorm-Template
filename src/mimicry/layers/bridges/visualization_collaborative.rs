//! Visualization-Collaborative Bridge (Layer 6 ↔ Layer 8).
//!
//! This bridge connects the Collaborative Learning layer (multi-agent
//! amplification) with the Pre-Cognitive Visualization layer.
//! Collective intelligence insights flow upward to inform visualization,
//! while simulation directives flow downward to regulate
//! collaborative dynamics.

use crate::mimicry::layers::bridge::{
    AmplificationResult, BidirectionalBridge, BridgeError, BridgeResult,
};
use crate::mimicry::layers::layer::{Layer, LayerState};

/// Bridge between Collaborative Learning (L6) and Pre-Cognitive Visualization (L8).
///
/// This bridge enables the visualization layer to monitor and balance
/// multi-agent collaborative dynamics. In the UGI model, this
/// corresponds to the left hemisphere (analytical) receiving collective
/// signals and applying adaptive timestepping to maintain stability.
pub struct VisualizationCollaborativeBridge {
    /// Current resonance factor.
    resonance: f32,
    /// Amplification factor for this bridge.
    amplification_factor: f32,
    /// Learning rate for reinforcement.
    learning_rate: f32,
    /// Diffusion coefficient — controls how collective patterns
    /// spread to visualization and back (inspired by UGI field diffusion D).
    diffusion_coefficient: f32,
    /// Total successful transfers.
    successful_transfers: u64,
}

impl VisualizationCollaborativeBridge {
    /// Create a new visualization-collaborative bridge.
    pub fn new() -> Self {
        Self {
            resonance: 1.0,
            amplification_factor: 1.14,
            learning_rate: 0.01,
            diffusion_coefficient: 0.1,
            successful_transfers: 0,
        }
    }

    /// Create with custom resonance.
    pub fn with_resonance(mut self, resonance: f32) -> Self {
        self.resonance = resonance;
        self
    }

    /// Transform collaborative learning output into pre-cognitive visualization input.
    fn transform_forward(&self, collaborative_state: &LayerState) -> LayerState {
        let mut new_state = LayerState::new(
            Layer::PreCognitiveVisualization,
            collaborative_state.data_arc(),
        );
        new_state.confidence =
            collaborative_state.confidence * self.resonance * (1.0 + self.diffusion_coefficient);
        new_state.add_upstream(collaborative_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_collaborative");
        new_state.set_metadata("direction", "forward");
        new_state.set_metadata(
            "diffusion_coefficient",
            &format!("{:.3}", self.diffusion_coefficient),
        );
        new_state
    }

    /// Transform visualization directive into collaborative guidance.
    fn transform_backward(&self, visualization_state: &LayerState) -> LayerState {
        let mut new_state =
            LayerState::new(Layer::CollaborativeLearning, visualization_state.data_arc());
        new_state.confidence = visualization_state.confidence * self.resonance * 1.04;
        new_state.add_upstream(visualization_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_collaborative");
        new_state.set_metadata("direction", "backward");
        new_state.set_metadata("balance_directive", "true");
        new_state
    }
}

impl Default for VisualizationCollaborativeBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl BidirectionalBridge for VisualizationCollaborativeBridge {
    fn name(&self) -> &str {
        "VisualizationCollaborativeBridge"
    }

    fn source_layer(&self) -> Layer {
        Layer::CollaborativeLearning
    }

    fn target_layer(&self) -> Layer {
        Layer::PreCognitiveVisualization
    }

    fn forward(&self, input: &LayerState) -> BridgeResult<LayerState> {
        if input.layer != Layer::CollaborativeLearning {
            return Err(BridgeError::InvalidInput(format!(
                "Expected CollaborativeLearning layer, got {:?}",
                input.layer
            )));
        }
        Ok(self.transform_forward(input))
    }

    fn backward(&self, feedback: &LayerState) -> BridgeResult<LayerState> {
        if feedback.layer != Layer::PreCognitiveVisualization {
            return Err(BridgeError::InvalidInput(format!(
                "Expected PreCognitiveVisualization layer, got {:?}",
                feedback.layer
            )));
        }
        Ok(self.transform_backward(feedback))
    }

    fn amplify(
        &self,
        up: &LayerState,
        down: &LayerState,
        max_iterations: u32,
    ) -> BridgeResult<AmplificationResult> {
        let mut up_state = up.clone();
        let mut down_state = down.clone();
        let mut iterations = 0;
        let convergence_threshold = 0.001;
        let mut previous_combined = 0.0f32;

        for i in 0..max_iterations {
            let forward_influence =
                up_state.confidence * self.resonance * self.diffusion_coefficient * 2.0;
            down_state.confidence = (down_state.confidence + forward_influence).min(2.0);

            let backward_influence = down_state.confidence * self.resonance * 0.28;
            up_state.confidence = (up_state.confidence + backward_influence).min(2.0);

            let combined = up_state.confidence * down_state.confidence * self.amplification_factor;

            if (combined - previous_combined).abs() < convergence_threshold {
                iterations = i + 1;
                break;
            }

            previous_combined = combined;
            iterations = i + 1;

            up_state.increment_amplification();
            down_state.increment_amplification();
        }

        let combined_confidence =
            up_state.confidence * down_state.confidence * self.amplification_factor;

        Ok(AmplificationResult {
            up_state,
            down_state,
            combined_confidence,
            amplification_factor: self.amplification_factor,
            iterations,
            converged: iterations < max_iterations,
            resonance: self.resonance,
        })
    }

    fn resonance(&self) -> f32 {
        self.resonance
    }

    fn reinforce(&mut self, result: &AmplificationResult) {
        if result.converged && result.combined_confidence > 1.0 {
            self.resonance = (self.resonance + self.learning_rate).min(2.0);
            self.successful_transfers += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = VisualizationCollaborativeBridge::new();
        assert_eq!(bridge.source_layer(), Layer::CollaborativeLearning);
        assert_eq!(bridge.target_layer(), Layer::PreCognitiveVisualization);
        assert_eq!(bridge.resonance(), 1.0);
    }

    #[test]
    fn test_forward_propagation() {
        let bridge = VisualizationCollaborativeBridge::new();
        let input = LayerState::with_confidence(
            Layer::CollaborativeLearning,
            "collective pattern".to_string(),
            0.8,
        );

        let output = bridge.forward(&input).unwrap();
        assert_eq!(output.layer, Layer::PreCognitiveVisualization);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn test_backward_propagation() {
        let bridge = VisualizationCollaborativeBridge::new();
        let feedback = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            "simulation directive".to_string(),
            0.9,
        );

        let output = bridge.backward(&feedback).unwrap();
        assert_eq!(output.layer, Layer::CollaborativeLearning);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn test_invalid_layer() {
        let bridge = VisualizationCollaborativeBridge::new();
        let wrong_layer = LayerState::new(Layer::BasePhysics, "test".to_string());
        assert!(bridge.forward(&wrong_layer).is_err());
    }

    #[test]
    fn test_amplification() {
        let bridge = VisualizationCollaborativeBridge::new();
        let up = LayerState::with_confidence(Layer::CollaborativeLearning, (), 0.8);
        let down = LayerState::with_confidence(Layer::PreCognitiveVisualization, (), 0.9);

        let result = bridge.amplify(&up, &down, 10).unwrap();
        assert!(result.combined_confidence > 0.72);
        assert!(result.iterations > 0);
    }
}
