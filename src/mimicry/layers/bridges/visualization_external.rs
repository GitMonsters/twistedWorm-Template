//! Visualization-External Bridge (Layer 7 ↔ Layer 8).
//!
//! This bridge connects the external APIs layer with the Pre-Cognitive
//! Visualization layer. External feedback flows upward to
//! inform visualization decisions, while simulation directives flow
//! downward to regulate external interactions.

use crate::mimicry::layers::bridge::{
    AmplificationResult, BidirectionalBridge, BridgeError, BridgeResult,
};
use crate::mimicry::layers::layer::{Layer, LayerState};

/// Bridge between External APIs (L7) and Pre-Cognitive Visualization (L8).
///
/// This bridge enables the visualization layer to receive real-time external
/// signals and feed stability/coherence directives back to the external
/// interface. It implements adaptive coupling inspired by the corpus
/// callosum coupling (kappa) from the UGI neural field model.
pub struct VisualizationExternalBridge {
    /// Current resonance factor.
    resonance: f32,
    /// Amplification factor for this bridge.
    amplification_factor: f32,
    /// Learning rate for reinforcement.
    learning_rate: f32,
    /// Coupling strength (kappa) — governs how strongly external signals
    /// influence visualization and vice versa.
    coupling_kappa: f32,
    /// Total successful transfers.
    successful_transfers: u64,
}

impl VisualizationExternalBridge {
    /// Create a new visualization-external bridge.
    pub fn new() -> Self {
        Self {
            resonance: 1.0,
            amplification_factor: 1.12,
            learning_rate: 0.01,
            coupling_kappa: 0.4,
            successful_transfers: 0,
        }
    }

    /// Create with custom resonance.
    pub fn with_resonance(mut self, resonance: f32) -> Self {
        self.resonance = resonance;
        self
    }

    /// Transform external API output into pre-cognitive visualization input.
    fn transform_forward(&self, external_state: &LayerState) -> LayerState {
        let mut new_state =
            LayerState::new(Layer::PreCognitiveVisualization, external_state.data_arc());
        // External signals are modulated by coupling strength
        new_state.confidence = external_state.confidence * self.resonance * self.coupling_kappa;
        new_state.add_upstream(external_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_external");
        new_state.set_metadata("direction", "forward");
        new_state.set_metadata("coupling_kappa", &format!("{:.3}", self.coupling_kappa));
        new_state
    }

    /// Transform visualization simulation directives into external API guidance.
    fn transform_backward(&self, visualization_state: &LayerState) -> LayerState {
        let mut new_state = LayerState::new(Layer::ExternalApis, visualization_state.data_arc());
        // Simulation directives carry authority — slight confidence boost
        new_state.confidence = visualization_state.confidence * self.resonance * 1.08;
        new_state.add_upstream(visualization_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_external");
        new_state.set_metadata("direction", "backward");
        new_state.set_metadata("simulation_directive", "true");
        new_state
    }
}

impl Default for VisualizationExternalBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl BidirectionalBridge for VisualizationExternalBridge {
    fn name(&self) -> &str {
        "VisualizationExternalBridge"
    }

    fn source_layer(&self) -> Layer {
        Layer::ExternalApis
    }

    fn target_layer(&self) -> Layer {
        Layer::PreCognitiveVisualization
    }

    fn forward(&self, input: &LayerState) -> BridgeResult<LayerState> {
        if input.layer != Layer::ExternalApis {
            return Err(BridgeError::InvalidInput(format!(
                "Expected ExternalApis layer, got {:?}",
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
            // Forward: external feedback influences visualization
            let forward_influence =
                up_state.confidence * self.resonance * self.coupling_kappa * 0.25;
            down_state.confidence = (down_state.confidence + forward_influence).min(2.0);

            // Backward: visualization stabilizes external
            let backward_influence = down_state.confidence * self.resonance * 0.3;
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
    fn test_visualization_bridge_creation() {
        let bridge = VisualizationExternalBridge::new();
        assert_eq!(bridge.source_layer(), Layer::ExternalApis);
        assert_eq!(bridge.target_layer(), Layer::PreCognitiveVisualization);
        assert_eq!(bridge.resonance(), 1.0);
    }

    #[test]
    fn test_visualization_forward_propagation() {
        let bridge = VisualizationExternalBridge::new();
        let input =
            LayerState::with_confidence(Layer::ExternalApis, "external data".to_string(), 0.8);

        let output = bridge.forward(&input).unwrap();
        assert_eq!(output.layer, Layer::PreCognitiveVisualization);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn test_visualization_backward_propagation() {
        let bridge = VisualizationExternalBridge::new();
        let feedback = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            "simulation directive".to_string(),
            0.9,
        );

        let output = bridge.backward(&feedback).unwrap();
        assert_eq!(output.layer, Layer::ExternalApis);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn test_visualization_invalid_layer() {
        let bridge = VisualizationExternalBridge::new();
        let wrong_layer = LayerState::new(Layer::BasePhysics, "test".to_string());
        assert!(bridge.forward(&wrong_layer).is_err());
    }

    #[test]
    fn test_visualization_amplification() {
        let bridge = VisualizationExternalBridge::new();
        let up = LayerState::with_confidence(Layer::ExternalApis, (), 0.8);
        let down = LayerState::with_confidence(Layer::PreCognitiveVisualization, (), 0.9);

        let result = bridge.amplify(&up, &down, 10).unwrap();
        assert!(result.combined_confidence > 0.72);
        assert!(result.iterations > 0);
    }
}
