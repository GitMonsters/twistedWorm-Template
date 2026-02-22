//! Inter-layer bridge implementations.
//!
//! This module contains concrete implementations of bidirectional bridges
//! that connect the 8 layers of the multiplicative integration system.

pub mod base_extended;
pub mod collaborative_external;
pub mod consciousness_external;
pub mod consciousness_language;
pub mod cross_domain;
pub mod crossdomain_consciousness;
pub mod individual_collective;
pub mod internal_external;
pub mod language_collaborative;
pub mod physics_consciousness;
pub mod physics_language;
pub mod visualization_base;
pub mod visualization_collaborative;
pub mod visualization_consciousness;
pub mod visualization_external;

// Re-export bridge implementations
pub use base_extended::BaseExtendedBridge;
pub use collaborative_external::CollaborativeExternalBridge;
pub use consciousness_external::ConsciousnessExternalBridge;
pub use consciousness_language::ConsciousnessLanguageBridge;
pub use cross_domain::CrossDomainBridge;
pub use crossdomain_consciousness::CrossDomainConsciousnessBridge;
pub use individual_collective::IndividualCollectiveBridge;
pub use internal_external::InternalExternalBridge;
pub use language_collaborative::LanguageCollaborativeBridge;
pub use physics_consciousness::PhysicsConsciousnessBridge;
pub use physics_language::PhysicsLanguageBridge;
pub use visualization_base::VisualizationBaseBridge;
pub use visualization_collaborative::VisualizationCollaborativeBridge;
pub use visualization_consciousness::VisualizationConsciousnessBridge;
pub use visualization_external::VisualizationExternalBridge;

use super::{BidirectionalBridge, BridgeNetwork};
use std::sync::Arc;

/// Create a fully connected bridge network with all standard bridges.
pub fn create_default_bridge_network() -> BridgeNetwork {
    let mut network = BridgeNetwork::new();

    // Register default bridges (8 total: 7 original + visualization-external)
    network.register(Arc::new(BaseExtendedBridge::new()));
    network.register(Arc::new(CrossDomainBridge::new()));
    network.register(Arc::new(PhysicsConsciousnessBridge::new()));
    network.register(Arc::new(PhysicsLanguageBridge::new()));
    network.register(Arc::new(IndividualCollectiveBridge::new()));
    network.register(Arc::new(InternalExternalBridge::new()));
    network.register(Arc::new(VisualizationExternalBridge::new()));
    network.register(Arc::new(VisualizationBaseBridge::new()));

    network
}

/// Bridge builder for creating custom bridge configurations.
pub struct BridgeBuilder {
    bridges: Vec<Arc<dyn BidirectionalBridge>>,
    global_amplification: f32,
}

impl BridgeBuilder {
    /// Create a new bridge builder.
    pub fn new() -> Self {
        Self {
            bridges: Vec::new(),
            global_amplification: 1.1,
        }
    }

    /// Add the base-extended physics bridge.
    pub fn with_base_extended(mut self) -> Self {
        self.bridges.push(Arc::new(BaseExtendedBridge::new()));
        self
    }

    /// Add the cross-domain bridge.
    pub fn with_cross_domain(mut self) -> Self {
        self.bridges.push(Arc::new(CrossDomainBridge::new()));
        self
    }

    /// Add the physics-consciousness bridge.
    pub fn with_physics_consciousness(mut self) -> Self {
        self.bridges
            .push(Arc::new(PhysicsConsciousnessBridge::new()));
        self
    }

    /// Add the physics-language bridge.
    pub fn with_physics_language(mut self) -> Self {
        self.bridges.push(Arc::new(PhysicsLanguageBridge::new()));
        self
    }

    /// Add the individual-collective bridge.
    pub fn with_individual_collective(mut self) -> Self {
        self.bridges
            .push(Arc::new(IndividualCollectiveBridge::new()));
        self
    }

    /// Add the internal-external bridge.
    pub fn with_internal_external(mut self) -> Self {
        self.bridges.push(Arc::new(InternalExternalBridge::new()));
        self
    }

    /// Add the consciousness-language bridge (L4↔L5).
    pub fn with_consciousness_language(mut self) -> Self {
        self.bridges
            .push(Arc::new(ConsciousnessLanguageBridge::new()));
        self
    }

    /// Add the language-collaborative bridge (L5↔L6).
    pub fn with_language_collaborative(mut self) -> Self {
        self.bridges
            .push(Arc::new(LanguageCollaborativeBridge::new()));
        self
    }

    /// Add the collaborative-external bridge (L6↔L7).
    pub fn with_collaborative_external(mut self) -> Self {
        self.bridges
            .push(Arc::new(CollaborativeExternalBridge::new()));
        self
    }

    /// Add the crossdomain-consciousness bridge (L3↔L4).
    pub fn with_crossdomain_consciousness(mut self) -> Self {
        self.bridges
            .push(Arc::new(CrossDomainConsciousnessBridge::new()));
        self
    }

    /// Add the consciousness-external bridge (L4↔L7).
    pub fn with_consciousness_external(mut self) -> Self {
        self.bridges
            .push(Arc::new(ConsciousnessExternalBridge::new()));
        self
    }

    /// Add the visualization-external bridge (L7↔L8).
    pub fn with_visualization_external(mut self) -> Self {
        self.bridges
            .push(Arc::new(VisualizationExternalBridge::new()));
        self
    }

    /// Add the visualization-consciousness bridge (L4↔L8).
    pub fn with_visualization_consciousness(mut self) -> Self {
        self.bridges
            .push(Arc::new(VisualizationConsciousnessBridge::new()));
        self
    }

    /// Add the visualization-collaborative bridge (L6↔L8).
    pub fn with_visualization_collaborative(mut self) -> Self {
        self.bridges
            .push(Arc::new(VisualizationCollaborativeBridge::new()));
        self
    }

    /// Add the visualization-base bridge (L8↔L1).
    pub fn with_visualization_base(mut self) -> Self {
        self.bridges.push(Arc::new(VisualizationBaseBridge::new()));
        self
    }

    /// Add all standard bridges (8 total, including visualization-external and visualization-base).
    pub fn with_all_bridges(self) -> Self {
        self.with_base_extended()
            .with_cross_domain()
            .with_physics_consciousness()
            .with_physics_language()
            .with_individual_collective()
            .with_internal_external()
            .with_visualization_external()
            .with_visualization_base()
    }

    /// Add all bridges including extended bridges (15 total).
    pub fn with_all_extended_bridges(self) -> Self {
        self.with_all_bridges()
            .with_consciousness_language()
            .with_language_collaborative()
            .with_collaborative_external()
            .with_crossdomain_consciousness()
            .with_consciousness_external()
            .with_visualization_consciousness()
            .with_visualization_collaborative()
    }

    /// Build all standard bridges and return as a vector.
    pub fn build_all() -> Vec<Arc<dyn BidirectionalBridge>> {
        vec![
            Arc::new(BaseExtendedBridge::new()),
            Arc::new(CrossDomainBridge::new()),
            Arc::new(PhysicsConsciousnessBridge::new()),
            Arc::new(PhysicsLanguageBridge::new()),
            Arc::new(IndividualCollectiveBridge::new()),
            Arc::new(InternalExternalBridge::new()),
            Arc::new(ConsciousnessLanguageBridge::new()),
            Arc::new(LanguageCollaborativeBridge::new()),
            Arc::new(CollaborativeExternalBridge::new()),
            Arc::new(CrossDomainConsciousnessBridge::new()),
            Arc::new(ConsciousnessExternalBridge::new()),
            Arc::new(VisualizationExternalBridge::new()),
            Arc::new(VisualizationConsciousnessBridge::new()),
            Arc::new(VisualizationCollaborativeBridge::new()),
            Arc::new(VisualizationBaseBridge::new()),
        ]
    }

    /// Set global amplification factor.
    pub fn with_global_amplification(mut self, factor: f32) -> Self {
        self.global_amplification = factor;
        self
    }

    /// Add a custom bridge.
    pub fn with_custom_bridge(mut self, bridge: Arc<dyn BidirectionalBridge>) -> Self {
        self.bridges.push(bridge);
        self
    }

    /// Build the bridge network.
    pub fn build(self) -> BridgeNetwork {
        let mut network = BridgeNetwork::new();
        network.set_global_amplification(self.global_amplification);

        for bridge in self.bridges {
            network.register(bridge);
        }

        network
    }
}

impl Default for BridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_bridge_network() {
        let network = create_default_bridge_network();
        assert_eq!(network.bridges().len(), 8);
    }

    #[test]
    fn test_extended_bridge_network() {
        let network = BridgeBuilder::new().with_all_extended_bridges().build();
        assert_eq!(network.bridges().len(), 15);
    }

    #[test]
    fn test_build_all() {
        let bridges = BridgeBuilder::build_all();
        assert_eq!(bridges.len(), 15);
    }

    #[test]
    fn test_bridge_builder() {
        let network = BridgeBuilder::new()
            .with_base_extended()
            .with_physics_consciousness()
            .with_global_amplification(1.5)
            .build();

        assert_eq!(network.bridges().len(), 2);
        assert_eq!(network.global_amplification(), 1.5);
    }

    #[test]
    fn test_builder_all_bridges() {
        let network = BridgeBuilder::new().with_all_bridges().build();
        assert_eq!(network.bridges().len(), 8);
    }
}
