//! OCTO Integration Module
//!
//! Provides integration with OctoTetrahedral AGI's RNA Editing Layer
//! for intelligent System 1/System 2 routing decisions.
//!
//! # Architecture
//!
//! The OCTO bridge uses PyO3 to call Python code that runs the
//! RNA Editing neural network. This provides:
//!
//! 1. **Temperature**: Uncertainty measure (0.1-5.0)
//!    - High temperature = uncertain = use System 2 (deep reasoning)
//!    - Low temperature = confident = use System 1 (fast path)
//!
//! 2. **Head Gating**: Controls which attention heads are active
//!    - Maps to personality trait modulation
//!    - 8 heads = 8 controllable dimensions
//!
//! 3. **Pathway Routing**: Routes to specialized personas
//!    - Pathway 0: Perception (input understanding)
//!    - Pathway 1: Reasoning (analysis)
//!    - Pathway 2: Action (output generation)
//!
//! # Usage
//!
//! ```ignore
//! use rustyworm::mimicry::octo::{OctoRNABridge, TextEmbedder};
//!
//! let bridge = OctoRNABridge::new()?;
//! let embedder = TextEmbedder::default();
//!
//! let embedding = embedder.embed("What is recursion?");
//! let routing = bridge.get_routing(&embedding)?;
//! ```

mod embedding;
mod rna_bridge;

pub use embedding::TextEmbedder;
pub use rna_bridge::{OctoConfig, OctoRNABridge, RNAEditingResult, RoutingDecision};
