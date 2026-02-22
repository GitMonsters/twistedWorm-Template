//! 8-Layer Multiplicative Integration System.
//!
//! This module implements a bidirectional layer architecture that enables
//! multiplicative confidence amplification across domains. Unlike additive
//! systems where confidence is bounded by the weakest input, this system
//! allows confidence to compound through resonance between layers.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────┐
//! │  Layer 8: Pre-Cognitive Visualization            │
//! │  ↕ (simulation, outcome projection, fidelity)  │
//! ├────────────────────────────────────────────────┤
//! │  Layer 7: Real-Time External APIs              │
//! │  ↕ (feedback)                                  │
//! ├────────────────────────────────────────────────┤
//! │  Layer 6: Collaborative Learning               │
//! │  ↕ (amplification)                             │
//! ├────────────────────────────────────────────────┤
//! │  Layer 5: Multilingual Processing              │
//! │  ↕ (perspective) ↔ (translation)              │
//! ├────────────────────────────────────────────────┤
//! │  Layer 4: GAIA Consciousness                   │
//! │  ↕ (analogical reasoning) ↔ (intuition)       │
//! ├────────────────────────────────────────────────┤
//! │  Layer 3: Cross-Domain Relationships           │
//! │  ↔ (emergence) ↔ (composition)                │
//! ├────────────────────────────────────────────────┤
//! │  Layer 2: Extended Physics Domains             │
//! │  ↕ (specialization)                           │
//! ├────────────────────────────────────────────────┤
//! │  Layer 1: Base Physics                         │
//! └────────────────────────────────────────────────┘
//! ```
//!
//! # Key Concepts
//!
//! ## Bidirectional Bridges
//!
//! Each layer connects to others through bidirectional bridges that allow:
//! - **Forward propagation**: Information flows from lower to higher layers
//! - **Backward propagation**: Refinements flow from higher to lower layers
//! - **Amplification**: Multiplicative confidence boosting through resonance
//!
//! ## Multiplicative Confidence
//!
//! Unlike traditional systems:
//! ```text
//! Traditional: confidence = min(c1, c2, c3)  // Bounded by weakest
//! Multiplicative: confidence = c1 × c2 × c3 × amplification  // Can exceed 1.0
//! ```
//!
//! ## GAIA Intuition Engine
//!
//! Layer 4 implements a weighted pattern matching system that:
//! - Detects patterns beyond explicit rules
//! - Performs cross-domain analogical reasoning
//! - Reinforces successful patterns over time
//!
//! # Usage
//!
//! ```rust,ignore
//! use rustyworm::mimicry::layers::{LayerStack, LayerState, Layer};
//!
//! // Create a layer stack
//! let mut stack = LayerStack::new();
//!
//! // Process input through the stack
//! let input = LayerState::new(Layer::BasePhysics, "input data");
//! let result = stack.process_bidirectional(input);
//!
//! // Check the amplified confidence
//! println!("Combined confidence: {}", result.combined_confidence);
//! ```
//!
//! # Feature Flags
//!
//! The layer system is always available but some integrations require features:
//! - `layers`: Core layer system (always on)
//! - `api`: External API providers for Layer 7
//! - `rl`: Reinforcement learning integration for amplification

pub mod bridge;
pub mod layer;
pub mod registry;
pub mod stack;

pub mod adaptive;
pub mod amplification;
pub mod bridges;
// dead_code is allowed at the module level since metrics are opt-in for consumers
pub mod compounding;
pub mod distribution_resonance;
pub mod domains;
pub mod emergence;
pub mod external;
pub mod gaia;
pub mod integration;
#[allow(dead_code)]
pub mod metrics;
pub mod octo_braid;
pub mod reserve;
pub mod visualization;

// Re-export primary types
pub use bridge::{
    compute_multiplicative_confidence, AmplificationResult, BidirectionalBridge, BridgeConnection,
    BridgeError, BridgeNetwork, BridgeResult,
};
pub use compounding::{BridgeMetrics, CompoundingAnalysis, CompoundingMetrics, LayerMetrics};
pub use domains::{DomainConfig, DomainFactory, DomainLayer, DomainProcessor};
pub use emergence::{
    EmergenceAnalysis, EmergenceConfig, EmergenceFramework, EmergenceMechanism, EmergenceStats,
};
pub use integration::{IntegrationConfig, IntegrationResult, IntegrationStats, LayerIntegration};
pub use layer::{Domain, FlowDirection, Layer, LayerConfig, LayerSignal, LayerState};
pub use registry::{
    LayerHandler, LayerProcessError, LayerRegistry, PassthroughHandler, RegistryStats,
};
pub use stack::{LayerStack, LayerStackConfig, StackProcessResult, StackStats};

// Phase 3: New module re-exports
pub use distribution_resonance::{
    distribution_resonance, distribution_resonance_raw, ConfidenceDistribution,
    DistributionResonanceConfig, DistributionResonanceSystem,
};
pub use reserve::{
    ConfidenceDecomposition, FractionalReserve, LayerReserve, ReserveConfig, ReserveStats,
};
pub use visualization::{
    EngineSummary, FidelityConfig, FidelityMeasurement, FidelityStats, FidelityTracker,
    OutcomeProjection, OutcomeProjector, ProjectorConfig, SimulationResult, SimulatorConfig,
    SubGoal, TaskSimulator, VisualizationConfig, VisualizationEngine, VisualizationResult,
};

// Phase 5: Adaptive systems
pub use adaptive::{
    AdaptiveCapConfig, AdaptiveConfidenceCap, BridgeWeight, DynamicBridgeWeighting,
    DynamicWeightConfig, GlobalLearningState, LayerEffectiveness, OnlineLearningConfig,
    OnlineLearningSystem,
};

// Phase 5D: Quality metrics
pub use metrics::{
    compare_quality, compute_quality, MetricsAnalyzer, QualityComparison, QualityDelta,
    QualityMetrics, QualityMetricsConfig,
};

// Phase 6: OCTO Braiding
pub use octo_braid::{BraidSignals, BraidStats, OctoBraid, OctoBraidConfig, PathwayEmphasis};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use super::{
        // Adaptive types (Phase 5)
        AdaptiveConfidenceCap,
        AmplificationResult,
        BidirectionalBridge,
        BridgeConnection,
        BridgeError,
        BridgeNetwork,
        BridgeResult,
        // Compounding types
        CompoundingAnalysis,
        CompoundingMetrics,
        // Distribution resonance types
        ConfidenceDistribution,
        DistributionResonanceSystem,
        Domain,
        // Domain types
        DomainConfig,
        DomainFactory,
        DomainProcessor,
        DynamicBridgeWeighting,
        // Emergence types
        EmergenceAnalysis,
        EmergenceFramework,
        EmergenceMechanism,
        FlowDirection,
        // Reserve types
        FractionalReserve,
        // Integration types
        IntegrationConfig,
        IntegrationResult,
        IntegrationStats,
        Layer,
        LayerConfig,
        LayerHandler,
        LayerIntegration,
        LayerProcessError,
        LayerRegistry,
        LayerSignal,
        LayerStack,
        LayerStackConfig,
        LayerState,
        // Quality metrics (Phase 5D)
        MetricsAnalyzer,
        // Phase 6: OCTO Braiding
        OctoBraid,
        OctoBraidConfig,
        OnlineLearningSystem,
        QualityMetrics,
        StackProcessResult,
        // Visualization types
        VisualizationConfig,
        VisualizationEngine,
        VisualizationResult,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_exports() {
        // Verify all primary types are accessible
        let _ = Layer::BasePhysics;
        let _ = Layer::GaiaConsciousness;
        let _ = Layer::PreCognitiveVisualization;
        let _ = Domain::Physics;
        let _ = Domain::Visualization;
        let _ = FlowDirection::Forward;
    }

    #[test]
    fn test_layer_state_creation() {
        let state = LayerState::new(Layer::CrossDomain, vec![1, 2, 3]);
        assert_eq!(state.layer, Layer::CrossDomain);
        assert_eq!(state.confidence, 1.0);
    }

    #[test]
    fn test_layer_stack_creation() {
        let stack = LayerStack::new();
        assert!(stack.bridge_network().bridges().is_empty());
    }

    #[test]
    fn test_registry_creation() {
        let registry = LayerRegistry::new();
        assert_eq!(registry.stats().total_layers, 8);
    }

    #[test]
    fn test_bridge_network() {
        let network = BridgeNetwork::new();
        assert_eq!(network.total_resonance(), 0.0);
    }
}
