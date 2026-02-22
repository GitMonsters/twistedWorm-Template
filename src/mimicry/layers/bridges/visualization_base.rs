//! Visualization-Base Bridge (Layer 8 ↔ Layer 1).
//!
//! Pre-cognitive visualization pre-warms base computation with expectations
//! from simulated task outcomes. Base physics grounds visualization with
//! fundamental processing constraints.

use crate::mimicry::layers::bridge::{
    AmplificationResult, BidirectionalBridge, BridgeError, BridgeResult,
};
use crate::mimicry::layers::layer::{Layer, LayerState};

/// Bridge between Base Physics (L1) and Pre-Cognitive Visualization (L8).
///
/// This bridge enables visualization expectations to pre-warm base-level
/// computation, while base physics grounds visualization with fundamental
/// processing constraints. The lower default resonance (0.78) reflects
/// the grounding constraint imposed by base physics on speculative
/// visualization outputs.
pub struct VisualizationBaseBridge {
    /// Current resonance factor.
    resonance: f32,
    /// Amplification factor for this bridge.
    amplification_factor: f32,
    /// Learning rate for reinforcement.
    learning_rate: f32,
    /// Total successful transfers.
    successful_transfers: u64,
}

impl VisualizationBaseBridge {
    /// Create a new visualization-base bridge.
    pub fn new() -> Self {
        Self {
            resonance: 0.78,
            amplification_factor: 1.10,
            learning_rate: 0.01,
            successful_transfers: 0,
        }
    }

    /// Create with custom resonance.
    pub fn with_resonance(mut self, resonance: f32) -> Self {
        self.resonance = resonance;
        self
    }

    /// Create with custom amplification factor.
    pub fn with_amplification_factor(mut self, factor: f32) -> Self {
        self.amplification_factor = factor;
        self
    }

    /// Transform base physics output into visualization input.
    /// Grounded by base constraints.
    fn transform_forward(&self, base_state: &LayerState) -> LayerState {
        let mut new_state =
            LayerState::new(Layer::PreCognitiveVisualization, base_state.data_arc());
        new_state.confidence = base_state.confidence * self.resonance;
        new_state.add_upstream(base_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_base");
        new_state.set_metadata("direction", "forward");
        new_state
    }

    /// Transform visualization expectations into base physics priming.
    /// Pre-warms base processing.
    fn transform_backward(&self, visualization_state: &LayerState) -> LayerState {
        let mut new_state = LayerState::new(Layer::BasePhysics, visualization_state.data_arc());
        // Backward pre-warming gets a slight confidence boost from expectation priming
        new_state.confidence = visualization_state.confidence * self.resonance * 1.05;
        new_state.add_upstream(visualization_state.id.clone());
        new_state.set_metadata("source_bridge", "visualization_base");
        new_state.set_metadata("direction", "backward");
        new_state.set_metadata("pre_warm", "true");
        new_state
    }
}

impl Default for VisualizationBaseBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl BidirectionalBridge for VisualizationBaseBridge {
    fn name(&self) -> &str {
        "VisualizationBaseBridge"
    }

    fn source_layer(&self) -> Layer {
        Layer::BasePhysics
    }

    fn target_layer(&self) -> Layer {
        Layer::PreCognitiveVisualization
    }

    fn forward(&self, input: &LayerState) -> BridgeResult<LayerState> {
        if input.layer != Layer::BasePhysics {
            return Err(BridgeError::InvalidInput(format!(
                "Expected BasePhysics layer, got {:?}",
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
            // Forward pass: base physics grounds visualization
            let forward_influence = up_state.confidence * self.resonance * 0.3;
            down_state.confidence = (down_state.confidence + forward_influence).min(2.0);

            // Backward pass: visualization pre-warms base processing
            let backward_influence = down_state.confidence * self.resonance * 0.3;
            up_state.confidence = (up_state.confidence + backward_influence).min(2.0);

            // Apply amplification
            let combined = up_state.confidence * down_state.confidence * self.amplification_factor;

            // Check convergence
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
            // Increase resonance for successful amplification
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
        let bridge = VisualizationBaseBridge::new();
        assert_eq!(bridge.source_layer(), Layer::BasePhysics);
        assert_eq!(bridge.target_layer(), Layer::PreCognitiveVisualization);
        assert_eq!(bridge.resonance(), 0.78);
    }

    #[test]
    fn test_forward_propagation() {
        let bridge = VisualizationBaseBridge::new();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        let output = bridge.forward(&input).unwrap();
        assert_eq!(output.layer, Layer::PreCognitiveVisualization);
        // 0.8 * 0.78 = 0.624
        assert!((output.confidence - 0.624).abs() < 0.001);
    }

    #[test]
    fn test_backward_propagation() {
        let bridge = VisualizationBaseBridge::new();
        let feedback = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            "expectation".to_string(),
            0.9,
        );

        let output = bridge.backward(&feedback).unwrap();
        assert_eq!(output.layer, Layer::BasePhysics);
        // 0.9 * 0.78 * 1.05 = 0.7371
        assert!((output.confidence - 0.7371).abs() < 0.001);
        assert_eq!(output.get_metadata("pre_warm"), Some("true"));
    }

    #[test]
    fn test_invalid_layer() {
        let bridge = VisualizationBaseBridge::new();
        let wrong_layer = LayerState::new(Layer::GaiaConsciousness, "test".to_string());

        assert!(bridge.forward(&wrong_layer).is_err());
    }

    #[test]
    fn test_amplification() {
        let bridge = VisualizationBaseBridge::new();
        let up = LayerState::with_confidence(Layer::BasePhysics, (), 0.8);
        let down = LayerState::with_confidence(Layer::PreCognitiveVisualization, (), 0.9);

        let result = bridge.amplify(&up, &down, 10).unwrap();
        assert!(result.combined_confidence > 0.72); // Should be amplified
        assert!(result.iterations > 0);
    }
}
