//! Pre-Cognitive Visualization Engine - Layer 8 Implementation.
//!
//! The Visualization Engine fires BEFORE the main forward pass, building
//! a structural model of the task and projecting what a successful output
//! looks like. It consists of three components:
//!
//! - **TaskSimulator**: Decomposes tasks into sub-goals with estimated confidence
//! - **OutcomeProjector**: Projects expected per-layer outputs for success
//! - **FidelityTracker**: Compares predictions vs actuals, learns bias corrections
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │              Pre-Cognitive Visualization Engine               │
//! ├──────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
//! │  │  Task Simulator  │─▶│ Outcome Projector│─▶│  Fidelity   │ │
//! │  │  (decompose,    │  │ (project success,│  │  Tracker    │ │
//! │  │   estimate)     │  │  per-layer conf) │  │ (learn bias)│ │
//! │  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
//! │           │                    │                   │        │
//! │           ▼                    ▼                   │        │
//! │  ┌─────────────────────────────────────┐          │        │
//! │  │         Pre-Warm Signals             │◀─────────┘        │
//! │  │  (layer → confidence boost)          │                   │
//! │  └─────────────────────────────────────┘                    │
//! └──────────────────────────────────────────────────────────────┘
//!         │                              ▲
//!         ▼ (before forward pass)        │ (after forward pass)
//!   [Main Forward Pass] ──────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rustyworm::mimicry::layers::visualization::{VisualizationEngine, VisualizationConfig};
//! use rustyworm::mimicry::layers::layer::{Layer, LayerState};
//!
//! let mut engine = VisualizationEngine::with_defaults();
//!
//! // Pre-visualization pass (before forward)
//! let input = LayerState::with_confidence(Layer::BasePhysics, "task data", 0.8);
//! let vis_result = engine.visualize(&input);
//!
//! // Use pre-warm signals during forward pass
//! for (layer, boost) in &vis_result.pre_warm_signals {
//!     println!("Pre-warm {}: +{:.3}", layer, boost);
//! }
//!
//! // After forward pass, record actuals for learning
//! use std::collections::HashMap;
//! let mut actuals = HashMap::new();
//! actuals.insert(Layer::BasePhysics, 0.78);
//! engine.record_outcome(vis_result.visualization_confidence, 0.78, &actuals);
//! ```

pub mod engine;
pub mod fidelity;
pub mod projector;
pub mod simulator;

// Re-export primary types
pub use engine::{EngineSummary, VisualizationConfig, VisualizationEngine, VisualizationResult};
pub use fidelity::{FidelityConfig, FidelityMeasurement, FidelityStats, FidelityTracker};
pub use projector::{OutcomeProjection, OutcomeProjector, ProjectorConfig};
pub use simulator::{SimulationResult, SimulatorConfig, SubGoal, TaskSimulator};

/// Error types for visualization operations.
#[derive(Debug, Clone)]
pub enum VisualizationError {
    /// Simulation failed.
    SimulationFailed(String),
    /// Projection failed.
    ProjectionFailed(String),
    /// Fidelity computation failed.
    FidelityError(String),
    /// Engine is disabled.
    EngineDisabled,
    /// Configuration error.
    ConfigError(String),
}

impl std::fmt::Display for VisualizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisualizationError::SimulationFailed(msg) => {
                write!(f, "Simulation failed: {}", msg)
            }
            VisualizationError::ProjectionFailed(msg) => {
                write!(f, "Projection failed: {}", msg)
            }
            VisualizationError::FidelityError(msg) => {
                write!(f, "Fidelity error: {}", msg)
            }
            VisualizationError::EngineDisabled => write!(f, "Visualization engine is disabled"),
            VisualizationError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for VisualizationError {}

/// Result type for visualization operations.
pub type VisualizationOpResult<T> = Result<T, VisualizationError>;

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        EngineSummary, FidelityConfig, FidelityMeasurement, FidelityStats, FidelityTracker,
        OutcomeProjection, OutcomeProjector, ProjectorConfig, SimulationResult, SimulatorConfig,
        SubGoal, TaskSimulator, VisualizationConfig, VisualizationEngine, VisualizationError,
        VisualizationOpResult, VisualizationResult,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimicry::layers::layer::{Layer, LayerState};

    #[test]
    fn test_visualization_error_display() {
        let err = VisualizationError::SimulationFailed("timeout".into());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_engine_creation() {
        let engine = VisualizationEngine::with_defaults();
        assert_eq!(engine.total_passes(), 0);
    }

    #[test]
    fn test_full_pipeline() {
        let mut engine = VisualizationEngine::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);

        // Visualization pass
        let result = engine.visualize(&input);
        assert!(!result.simulation.task_id.is_empty());
        assert!(!result.projections.is_empty());
        assert!(result.visualization_confidence > 0.0);

        // Convert to layer state
        let state = result.to_layer_state();
        assert_eq!(state.layer, Layer::PreCognitiveVisualization);
    }

    #[test]
    fn test_simulator_standalone() {
        let mut sim = TaskSimulator::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "task".to_string(), 0.9);

        let result = sim.simulate(&input);
        assert!(!result.sub_goals.is_empty());
        assert!(result.overall_confidence > 0.0);
    }

    #[test]
    fn test_projector_standalone() {
        let mut proj = OutcomeProjector::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "task".to_string(), 0.9);

        let projections = proj.project(&input);
        assert!(!projections.is_empty());
        assert!(projections.iter().any(|p| p.is_primary));
    }

    #[test]
    fn test_fidelity_standalone() {
        let mut tracker = FidelityTracker::with_defaults();
        let needs_recal = tracker.record_simple("test", 0.8, 0.75);
        assert!(!needs_recal);
    }
}
