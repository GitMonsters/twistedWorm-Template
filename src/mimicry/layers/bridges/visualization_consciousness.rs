//! Visualization-Consciousness Bridge (Layer 4 ↔ Layer 8).
//!
//! This bridge connects GAIA Consciousness (intuition, analogical reasoning)
//! with the Pre-Cognitive Visualization layer. Consciousness provides
//! deep pattern insights that inform visualization decisions, while visualization
//! directives can guide the focus of intuitive pattern recognition.

use crate::mimicry::layers::bridge::{
    AmplificationResult, BidirectionalBridge, BridgeError, BridgeResult,
};
use crate::mimicry::layers::layer::{Layer, LayerState};

/// Bridge between GAIA Consciousness (L4) and Pre-Cognitive Visualization (L8).
///
/// This is a critical long-range bridge that connects intuition with
/// visualization. In the UGI neural field model, this corresponds to the
/// right hemisphere (creative/nonlinear) feeding its pattern insights
/// to the visualization layer for stability assessment.
pub struct VisualizationConsciousnessBridge {
    /// Current resonance factor.
    resonance: f32,
    /// Amplification factor for this bridge.
    amplification_factor: f32,
    /// Learning rate for reinforcement.
    learning_rate: f32,
    /// Nonlinear drive (beta_R) — controls how strongly consciousness
    /// patterns influence visualization.
    nonlinear_drive: f32,
    /// Total successful transfers.
    successful_transfers: u64,
}

impl VisualizationConsciousnessBridge {
    /// Create a new visualization-consciousness bridge.
    pub fn new() -> Self {
        Self {
            resonance: 1.0,
            amplification_factor: 1.18,
            learning_rate: 0.01,
            nonlinear_drive: 4.5,
            successful_transfers: 0,
        }
    }

    /// Create with custom resonance.
    pub fn with_resonance(mut self, resonance: f32) -> Self {
        self.resonance = resonance;
        self
    }

    /// Sigmoid activation (firing-rate model from UGI neural fields).
    fn activation(x: f32) -> f32 {
        1.0 / (1.0 + (-4.0 * x).exp())
    }

    /// Transform consciousness insight into visualization input.
    fn transform_forward(&self, consciousness_state: &LayerState) -> LayerState {
        let mut new_state = LayerState::new(
            Layer::PreCognitiveVisualization,
            consciousness_state.data_arc(),
        );
        let activated = Self::activation(consciousness_state.confidence);
        let nonlinear_term = self.nonlinear_drive * activated * (1.0 - activated);
        new_state.confidence =
            (consciousness_state.confidence * self.resonance + nonlinear_term * 0.1).min(2.0);
        new_state.add_upstream(consciousness_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_consciousness");
        new_state.set_metadata("direction", "forward");
        new_state.set_metadata("nonlinear_drive", &format!("{:.3}", self.nonlinear_drive));
        new_state
    }

    /// Transform visualization directive into consciousness guidance.
    fn transform_backward(&self, visualization_state: &LayerState) -> LayerState {
        let mut new_state =
            LayerState::new(Layer::GaiaConsciousness, visualization_state.data_arc());
        new_state.confidence = visualization_state.confidence * self.resonance * 1.06;
        new_state.add_upstream(visualization_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_consciousness");
        new_state.set_metadata("direction", "backward");
        new_state.set_metadata("visualization_guided", "true");
        new_state
    }
}

impl Default for VisualizationConsciousnessBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl BidirectionalBridge for VisualizationConsciousnessBridge {
    fn name(&self) -> &str {
        "VisualizationConsciousnessBridge"
    }

    fn source_layer(&self) -> Layer {
        Layer::GaiaConsciousness
    }

    fn target_layer(&self) -> Layer {
        Layer::PreCognitiveVisualization
    }

    fn forward(&self, input: &LayerState) -> BridgeResult<LayerState> {
        if input.layer != Layer::GaiaConsciousness {
            return Err(BridgeError::InvalidInput(format!(
                "Expected GaiaConsciousness layer, got {:?}",
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
            let activated = Self::activation(up_state.confidence);
            let forward_influence = activated * self.resonance * 0.3;
            down_state.confidence = (down_state.confidence + forward_influence).min(2.0);

            let backward_influence = down_state.confidence * self.resonance * 0.25;
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
        let bridge = VisualizationConsciousnessBridge::new();
        assert_eq!(bridge.source_layer(), Layer::GaiaConsciousness);
        assert_eq!(bridge.target_layer(), Layer::PreCognitiveVisualization);
        assert_eq!(bridge.resonance(), 1.0);
    }

    #[test]
    fn test_forward_propagation() {
        let bridge = VisualizationConsciousnessBridge::new();
        let input = LayerState::with_confidence(
            Layer::GaiaConsciousness,
            "intuition pattern".to_string(),
            0.8,
        );

        let output = bridge.forward(&input).unwrap();
        assert_eq!(output.layer, Layer::PreCognitiveVisualization);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn test_backward_propagation() {
        let bridge = VisualizationConsciousnessBridge::new();
        let feedback = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            "visualization directive".to_string(),
            0.9,
        );

        let output = bridge.backward(&feedback).unwrap();
        assert_eq!(output.layer, Layer::GaiaConsciousness);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn test_invalid_layer() {
        let bridge = VisualizationConsciousnessBridge::new();
        let wrong_layer = LayerState::new(Layer::BasePhysics, "test".to_string());
        assert!(bridge.forward(&wrong_layer).is_err());
    }

    #[test]
    fn test_amplification() {
        let bridge = VisualizationConsciousnessBridge::new();
        let up = LayerState::with_confidence(Layer::GaiaConsciousness, (), 0.8);
        let down = LayerState::with_confidence(Layer::PreCognitiveVisualization, (), 0.9);

        let result = bridge.amplify(&up, &down, 10).unwrap();
        assert!(result.combined_confidence > 0.72);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_nonlinear_activation() {
        let mid = VisualizationConsciousnessBridge::activation(0.0);
        assert!((mid - 0.5).abs() < 0.01);

        let high = VisualizationConsciousnessBridge::activation(2.0);
        assert!(high > 0.99);
    }
}
