//! Layer definitions for the 8-Layer Multiplicative Integration System.
//!
//! This module defines the core layer types and their states, enabling
//! bidirectional information flow with multiplicative confidence amplification.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// The 8 layers of the multiplicative integration system.
///
/// Information flows bidirectionally between layers, with each layer
/// both contributing to and receiving refinements from connected layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layer {
    /// Layer 1: Base physics processing - core 8-phase pipeline (phases 1-4)
    /// Handles: Perception, Memory, Compression, Reasoning
    BasePhysics,

    /// Layer 2: Extended physics domains - 8-phase pipeline (phases 5-8)
    /// Handles: Planning, Tool Use, Execution, Learning
    ExtendedPhysics,

    /// Layer 3: Cross-domain relationships
    /// Handles: Emergence detection, composition analysis
    CrossDomain,

    /// Layer 4: GAIA Consciousness - Intuition engine
    /// Handles: Pattern recognition beyond explicit rules, analogical reasoning
    GaiaConsciousness,

    /// Layer 5: Multilingual processing
    /// Handles: Translation, perspective shifting, linguistic analysis
    MultilingualProcessing,

    /// Layer 6: Collaborative learning
    /// Handles: Multi-agent amplification, social learning
    CollaborativeLearning,

    /// Layer 7: Real-time external APIs
    /// Handles: External validation, feedback loops
    ExternalApis,

    /// Layer 8: Pre-Cognitive Visualization
    /// Handles: Task-space simulation before action, outcome projection,
    /// fidelity tracking, and pre-warming of downstream layer confidences.
    /// This layer fires BEFORE the main forward pass, building a structural
    /// model of the task and projecting what a successful output looks like.
    PreCognitiveVisualization,
}

impl Layer {
    /// Returns all layers in order from base to visualization.
    pub fn all() -> &'static [Layer] {
        &[
            Layer::BasePhysics,
            Layer::ExtendedPhysics,
            Layer::CrossDomain,
            Layer::GaiaConsciousness,
            Layer::MultilingualProcessing,
            Layer::CollaborativeLearning,
            Layer::ExternalApis,
            Layer::PreCognitiveVisualization,
        ]
    }

    /// Returns the layer number (1-8).
    pub fn number(&self) -> u8 {
        match self {
            Layer::BasePhysics => 1,
            Layer::ExtendedPhysics => 2,
            Layer::CrossDomain => 3,
            Layer::GaiaConsciousness => 4,
            Layer::MultilingualProcessing => 5,
            Layer::CollaborativeLearning => 6,
            Layer::ExternalApis => 7,
            Layer::PreCognitiveVisualization => 8,
        }
    }

    /// Returns a human-readable name for the layer.
    pub fn name(&self) -> &'static str {
        match self {
            Layer::BasePhysics => "Base Physics",
            Layer::ExtendedPhysics => "Extended Physics",
            Layer::CrossDomain => "Cross-Domain",
            Layer::GaiaConsciousness => "GAIA Consciousness",
            Layer::MultilingualProcessing => "Multilingual Processing",
            Layer::CollaborativeLearning => "Collaborative Learning",
            Layer::ExternalApis => "External APIs",
            Layer::PreCognitiveVisualization => "Pre-Cognitive Visualization",
        }
    }

    /// Returns the primary function of this layer.
    pub fn function(&self) -> &'static str {
        match self {
            Layer::BasePhysics => "specialization",
            Layer::ExtendedPhysics => "specialization",
            Layer::CrossDomain => "emergence, composition",
            Layer::GaiaConsciousness => "analogical reasoning, intuition",
            Layer::MultilingualProcessing => "perspective, translation",
            Layer::CollaborativeLearning => "amplification",
            Layer::ExternalApis => "feedback",
            Layer::PreCognitiveVisualization => "simulation, outcome projection, fidelity tracking",
        }
    }

    /// Returns the layers that this layer can directly bridge to.
    pub fn connected_layers(&self) -> Vec<Layer> {
        match self {
            Layer::BasePhysics => vec![
                Layer::ExtendedPhysics,
                Layer::CrossDomain,
                Layer::GaiaConsciousness,
                Layer::MultilingualProcessing,
                Layer::PreCognitiveVisualization,
            ],
            Layer::ExtendedPhysics => {
                vec![Layer::BasePhysics, Layer::CrossDomain, Layer::ExternalApis]
            }
            Layer::CrossDomain => vec![
                Layer::BasePhysics,
                Layer::ExtendedPhysics,
                Layer::GaiaConsciousness,
                Layer::CollaborativeLearning,
            ],
            Layer::GaiaConsciousness => vec![
                Layer::BasePhysics,
                Layer::CrossDomain,
                Layer::MultilingualProcessing,
                Layer::ExternalApis,
                Layer::PreCognitiveVisualization,
            ],
            Layer::MultilingualProcessing => vec![
                Layer::BasePhysics,
                Layer::GaiaConsciousness,
                Layer::CollaborativeLearning,
            ],
            Layer::CollaborativeLearning => vec![
                Layer::CrossDomain,
                Layer::MultilingualProcessing,
                Layer::ExternalApis,
                Layer::PreCognitiveVisualization,
            ],
            Layer::ExternalApis => vec![
                Layer::ExtendedPhysics,
                Layer::GaiaConsciousness,
                Layer::CollaborativeLearning,
                Layer::PreCognitiveVisualization,
            ],
            Layer::PreCognitiveVisualization => vec![
                Layer::BasePhysics,
                Layer::GaiaConsciousness,
                Layer::CollaborativeLearning,
                Layer::ExternalApis,
            ],
        }
    }

    /// Check if this layer can bridge to another layer.
    pub fn can_bridge_to(&self, other: Layer) -> bool {
        self.connected_layers().contains(&other)
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer {}: {}", self.number(), self.name())
    }
}

/// Domain categories for cross-layer pattern transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Physics,
    Language,
    Consciousness,
    Social,
    External,
    Emergent,
    Visualization,
}

impl Domain {
    /// Returns the primary layer for this domain.
    pub fn primary_layer(&self) -> Layer {
        match self {
            Domain::Physics => Layer::BasePhysics,
            Domain::Language => Layer::MultilingualProcessing,
            Domain::Consciousness => Layer::GaiaConsciousness,
            Domain::Social => Layer::CollaborativeLearning,
            Domain::External => Layer::ExternalApis,
            Domain::Emergent => Layer::CrossDomain,
            Domain::Visualization => Layer::PreCognitiveVisualization,
        }
    }
}

/// State held by a layer, containing typed data and metadata.
#[derive(Clone)]
pub struct LayerState {
    /// The layer this state belongs to.
    pub layer: Layer,

    /// Unique identifier for this state instance.
    pub id: String,

    /// The actual data payload (type-erased for flexibility).
    data: Arc<dyn Any + Send + Sync>,

    /// Confidence level for this state (0.0 - potentially > 1.0 after amplification).
    pub confidence: f32,

    /// Metadata key-value pairs.
    pub metadata: HashMap<String, String>,

    /// IDs of upstream states that contributed to this state.
    pub upstream_refs: Vec<String>,

    /// IDs of downstream states that depend on this state.
    pub downstream_refs: Vec<String>,

    /// Timestamp when this state was created (millis since epoch).
    pub created_at: u64,

    /// Number of amplification iterations this state has undergone.
    pub amplification_iterations: u32,
}

impl LayerState {
    /// Create a new layer state with the given data.
    pub fn new<T: Any + Send + Sync + 'static>(layer: Layer, data: T) -> Self {
        Self {
            layer,
            id: Self::generate_id(),
            data: Arc::new(data),
            confidence: 1.0,
            metadata: HashMap::new(),
            upstream_refs: Vec::new(),
            downstream_refs: Vec::new(),
            created_at: Self::current_time_millis(),
            amplification_iterations: 0,
        }
    }

    /// Create a new layer state with explicit confidence.
    pub fn with_confidence<T: Any + Send + Sync + 'static>(
        layer: Layer,
        data: T,
        confidence: f32,
    ) -> Self {
        let mut state = Self::new(layer, data);
        state.confidence = confidence;
        state
    }

    /// Try to downcast the data to the expected type.
    pub fn data<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.data.downcast_ref::<T>()
    }

    /// Get a clone of the Arc-wrapped data.
    pub fn data_arc(&self) -> Arc<dyn Any + Send + Sync> {
        Arc::clone(&self.data)
    }

    /// Set metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Add an upstream reference.
    pub fn add_upstream(&mut self, id: impl Into<String>) {
        self.upstream_refs.push(id.into());
    }

    /// Add a downstream reference.
    pub fn add_downstream(&mut self, id: impl Into<String>) {
        self.downstream_refs.push(id.into());
    }

    /// Increment the amplification iteration counter.
    pub fn increment_amplification(&mut self) {
        self.amplification_iterations += 1;
    }

    /// Apply a confidence multiplier (for multiplicative amplification).
    pub fn amplify_confidence(&mut self, factor: f32) {
        self.confidence *= factor;
        self.increment_amplification();
    }

    /// Generate a unique ID.
    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("ls_{:x}", nanos)
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

impl fmt::Debug for LayerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerState")
            .field("layer", &self.layer)
            .field("id", &self.id)
            .field("confidence", &self.confidence)
            .field("upstream_refs", &self.upstream_refs.len())
            .field("downstream_refs", &self.downstream_refs.len())
            .field("amplification_iterations", &self.amplification_iterations)
            .finish()
    }
}

/// Configuration for a layer's behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// The layer this configuration applies to.
    pub layer: Layer,

    /// Whether this layer is enabled.
    pub enabled: bool,

    /// Maximum number of amplification iterations.
    pub max_amplification_iterations: u32,

    /// Minimum confidence threshold for output.
    pub min_confidence_threshold: f32,

    /// Base amplification factor for this layer.
    pub amplification_factor: f32,

    /// Whether to propagate backward (higher to lower layers).
    pub enable_backward_propagation: bool,

    /// Custom parameters for layer-specific behavior.
    pub custom_params: HashMap<String, String>,
}

impl LayerConfig {
    /// Create a new configuration for a layer with defaults.
    pub fn new(layer: Layer) -> Self {
        Self {
            layer,
            enabled: true,
            max_amplification_iterations: 10,
            min_confidence_threshold: 0.3,
            amplification_factor: 1.1,
            enable_backward_propagation: true,
            custom_params: HashMap::new(),
        }
    }

    /// Create a disabled configuration.
    pub fn disabled(layer: Layer) -> Self {
        let mut config = Self::new(layer);
        config.enabled = false;
        config
    }

    /// Set a custom parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_params.insert(key.into(), value.into());
        self
    }

    /// Set amplification factor.
    pub fn with_amplification_factor(mut self, factor: f32) -> Self {
        self.amplification_factor = factor;
        self
    }

    /// Set max iterations.
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_amplification_iterations = max;
        self
    }
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self::new(Layer::BasePhysics)
    }
}

/// Represents the direction of information flow between layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowDirection {
    /// Forward: from lower layer number to higher.
    Forward,
    /// Backward: from higher layer number to lower.
    Backward,
    /// Bidirectional: both directions simultaneously.
    Bidirectional,
}

/// A signal passed between layers during propagation.
#[derive(Debug, Clone)]
pub struct LayerSignal {
    /// Source layer.
    pub source: Layer,
    /// Target layer.
    pub target: Layer,
    /// Direction of flow.
    pub direction: FlowDirection,
    /// The payload state.
    pub state: LayerState,
    /// Resonance factor from the bridge.
    pub resonance: f32,
}

impl LayerSignal {
    /// Create a new signal.
    pub fn new(source: Layer, target: Layer, state: LayerState) -> Self {
        let direction = if source.number() < target.number() {
            FlowDirection::Forward
        } else if source.number() > target.number() {
            FlowDirection::Backward
        } else {
            FlowDirection::Bidirectional
        };

        Self {
            source,
            target,
            direction,
            state,
            resonance: 1.0,
        }
    }

    /// Set the resonance factor.
    pub fn with_resonance(mut self, resonance: f32) -> Self {
        self.resonance = resonance;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_ordering() {
        assert_eq!(Layer::BasePhysics.number(), 1);
        assert_eq!(Layer::ExternalApis.number(), 7);
        assert_eq!(Layer::PreCognitiveVisualization.number(), 8);
    }

    #[test]
    fn test_layer_connections() {
        // Base physics should connect to extended physics
        assert!(Layer::BasePhysics.can_bridge_to(Layer::ExtendedPhysics));
        // External APIs should not directly connect to base physics
        assert!(!Layer::ExternalApis.can_bridge_to(Layer::BasePhysics));
    }

    #[test]
    fn test_layer_state_creation() {
        let state = LayerState::new(Layer::GaiaConsciousness, "test data".to_string());
        assert_eq!(state.layer, Layer::GaiaConsciousness);
        assert_eq!(state.confidence, 1.0);
        assert_eq!(state.data::<String>(), Some(&"test data".to_string()));
    }

    #[test]
    fn test_confidence_amplification() {
        let mut state = LayerState::with_confidence(Layer::CrossDomain, 42i32, 0.8);
        assert_eq!(state.confidence, 0.8);

        state.amplify_confidence(1.2);
        assert!((state.confidence - 0.96).abs() < 0.001);
        assert_eq!(state.amplification_iterations, 1);
    }

    #[test]
    fn test_layer_signal_direction() {
        let state = LayerState::new(Layer::BasePhysics, ());

        let forward = LayerSignal::new(Layer::BasePhysics, Layer::ExtendedPhysics, state.clone());
        assert_eq!(forward.direction, FlowDirection::Forward);

        let backward = LayerSignal::new(Layer::ExternalApis, Layer::CollaborativeLearning, state);
        assert_eq!(backward.direction, FlowDirection::Backward);
    }

    #[test]
    fn test_domain_layer_mapping() {
        assert_eq!(Domain::Physics.primary_layer(), Layer::BasePhysics);
        assert_eq!(
            Domain::Consciousness.primary_layer(),
            Layer::GaiaConsciousness
        );
        assert_eq!(
            Domain::Visualization.primary_layer(),
            Layer::PreCognitiveVisualization
        );
    }
}
