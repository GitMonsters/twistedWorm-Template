//! Integration module for connecting LayerStack to MimicryEngine.
//!
//! This module provides the bridge between the 8-layer multiplicative system
//! and the existing dual-process MimicryEngine architecture.

use super::bridges::BridgeBuilder;
use super::gaia::intuition::{GaiaConfig, GaiaIntuitionEngine};
use super::layer::{Domain, Layer, LayerState};
use super::stack::{LayerStack, LayerStackConfig, StackProcessResult};

/// Integration layer connecting the 8-layer system to MimicryEngine.
///
/// This struct wraps the LayerStack and provides convenience methods
/// for processing mimicry inputs through the multiplicative amplification
/// pipeline.
pub struct LayerIntegration {
    /// The underlying layer stack.
    stack: LayerStack,
    /// GAIA intuition engine for Layer 4 processing.
    gaia_engine: GaiaIntuitionEngine,
    /// Integration configuration.
    config: IntegrationConfig,
    /// Processing statistics.
    stats: IntegrationStats,
}

/// Configuration for the integration layer.
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Whether to use Layer 4 GAIA processing.
    pub enable_gaia: bool,
    /// Whether to use external API layer (Layer 7).
    pub enable_external_apis: bool,
    /// Minimum confidence required for amplification benefit.
    pub min_amplification_benefit: f32,
    /// Whether to track detailed statistics.
    pub track_statistics: bool,
    /// Maximum processing time in milliseconds.
    pub max_processing_time_ms: u64,
    /// Whether to enable Phase 3 subsystems (visualization, reserve, distribution resonance).
    pub enable_phase3: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_gaia: true,
            enable_external_apis: false, // Disabled by default (requires API keys)
            min_amplification_benefit: 0.05,
            track_statistics: true,
            max_processing_time_ms: 5000,
            enable_phase3: false, // Phase 3 off by default for backward compat
        }
    }
}

/// Statistics for layer integration processing.
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Total number of inputs processed.
    pub total_processed: u64,
    /// Number of times GAIA layer was invoked.
    pub gaia_invocations: u64,
    /// Number of times external APIs were used.
    pub external_api_calls: u64,
    /// Average confidence improvement from amplification.
    pub avg_confidence_improvement: f32,
    /// Maximum confidence achieved.
    pub max_confidence: f32,
    /// Number of convergence events.
    pub convergence_count: u64,
    /// Total amplification factor accumulated.
    pub total_amplification: f32,
    /// Phase 3: Number of visualization passes.
    pub visualization_passes: u64,
    /// Phase 3: Total reserve burst events.
    pub reserve_burst_events: u64,
}

/// Result of processing through the integration layer.
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// The underlying stack result.
    pub stack_result: StackProcessResult,
    /// Confidence before amplification.
    pub initial_confidence: f32,
    /// Confidence after amplification.
    pub final_confidence: f32,
    /// Whether GAIA provided a contribution.
    pub gaia_contributed: bool,
    /// GAIA intuition confidence (if used).
    pub gaia_confidence: Option<f32>,
    /// Domains that were activated.
    pub active_domains: Vec<Domain>,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
    /// Phase 3: Whether visualization pre-pass was active.
    pub visualization_active: bool,
    /// Phase 3: Visualization confidence (if active).
    pub visualization_confidence: Option<f32>,
    /// Phase 3: Whether any reserve bursts occurred.
    pub reserve_bursts_occurred: bool,
}

impl LayerIntegration {
    /// Create a new layer integration with default settings.
    pub fn new() -> Self {
        Self::with_config(IntegrationConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: IntegrationConfig) -> Self {
        let stack_config = if config.enable_phase3 {
            LayerStackConfig::new().with_phase3()
        } else {
            LayerStackConfig::new()
        };

        let mut stack = LayerStack::with_config(stack_config);

        // Register all standard bridges
        let bridges = BridgeBuilder::build_all();
        for bridge in bridges {
            stack.register_bridge(bridge);
        }

        Self {
            stack,
            gaia_engine: GaiaIntuitionEngine::new(GaiaConfig::default()),
            config,
            stats: IntegrationStats::default(),
        }
    }

    /// Create with custom stack configuration.
    pub fn with_stack_config(config: IntegrationConfig, stack_config: LayerStackConfig) -> Self {
        let effective_stack_config = if config.enable_phase3 {
            // Ensure Phase 3 flags are set even if caller forgot
            stack_config.with_phase3()
        } else {
            stack_config
        };

        let mut stack = LayerStack::with_config(effective_stack_config);

        let bridges = BridgeBuilder::build_all();
        for bridge in bridges {
            stack.register_bridge(bridge);
        }

        Self {
            stack,
            gaia_engine: GaiaIntuitionEngine::new(GaiaConfig::default()),
            config,
            stats: IntegrationStats::default(),
        }
    }

    /// Get a reference to the underlying stack.
    pub fn stack(&self) -> &LayerStack {
        &self.stack
    }

    /// Get a mutable reference to the underlying stack.
    pub fn stack_mut(&mut self) -> &mut LayerStack {
        &mut self.stack
    }

    /// Get a reference to the GAIA engine.
    pub fn gaia_engine(&self) -> &GaiaIntuitionEngine {
        &self.gaia_engine
    }

    /// Get a mutable reference to the GAIA engine.
    pub fn gaia_engine_mut(&mut self) -> &mut GaiaIntuitionEngine {
        &mut self.gaia_engine
    }

    /// Get current statistics.
    pub fn stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = IntegrationStats::default();
    }

    /// Process an input through the full 8-layer system.
    ///
    /// This is the main entry point for integrating with MimicryEngine.
    /// It takes a text input and optional context, processes through all
    /// enabled layers, and returns the amplified result.
    pub fn process(&mut self, input: &str, context: Option<&str>) -> IntegrationResult {
        let start_time = std::time::Instant::now();

        // Create initial state at Layer 1 (Base Physics)
        let initial_state = LayerState::new(Layer::BasePhysics, input.to_string());
        let initial_confidence = initial_state.confidence;

        // Process through the stack
        let stack_result = self.stack.process_bidirectional(initial_state);

        // Optionally enhance with GAIA
        let (gaia_contributed, gaia_confidence) = if self.config.enable_gaia {
            self.apply_gaia_enhancement(input, context, &stack_result)
        } else {
            (false, None)
        };

        // Collect active domains
        let active_domains = self.collect_active_domains(&stack_result);

        // Calculate final confidence with GAIA boost
        let final_confidence = if let Some(gaia_conf) = gaia_confidence {
            // Multiplicative combination with GAIA
            stack_result.combined_confidence * (1.0 + gaia_conf * 0.5)
        } else {
            stack_result.combined_confidence
        };

        // Phase 3: Extract visualization and reserve info
        let visualization_active = stack_result.visualization.is_some();
        let visualization_confidence = stack_result
            .visualization
            .as_ref()
            .map(|v| v.visualization_confidence);
        let reserve_bursts_occurred = !stack_result.reserve_decompositions.is_empty()
            && stack_result
                .reserve_decompositions
                .values()
                .any(|d| d.burst > 0.0);

        // Update statistics
        self.update_stats(
            initial_confidence,
            final_confidence,
            &stack_result,
            gaia_contributed,
        );

        // Phase 3: Update Phase 3 stats
        if visualization_active {
            self.stats.visualization_passes += 1;
        }
        if reserve_bursts_occurred {
            self.stats.reserve_burst_events += 1;
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        IntegrationResult {
            stack_result,
            initial_confidence,
            final_confidence,
            gaia_contributed,
            gaia_confidence,
            active_domains,
            processing_time_ms,
            visualization_active,
            visualization_confidence,
            reserve_bursts_occurred,
        }
    }

    /// Process with explicit domain hints.
    pub fn process_with_domain(
        &mut self,
        input: &str,
        primary_domain: Domain,
        context: Option<&str>,
    ) -> IntegrationResult {
        let start_time = std::time::Instant::now();

        // Start at the primary layer for this domain
        let start_layer = primary_domain.primary_layer();
        let initial_state = LayerState::new(start_layer, input.to_string());
        let initial_confidence = initial_state.confidence;

        // Process through stack
        let stack_result = self.stack.process_bidirectional(initial_state);

        // GAIA enhancement with domain hint
        let (gaia_contributed, gaia_confidence) = if self.config.enable_gaia {
            self.apply_gaia_with_domain(input, context, primary_domain, &stack_result)
        } else {
            (false, None)
        };

        let active_domains = self.collect_active_domains(&stack_result);
        let final_confidence = if let Some(gaia_conf) = gaia_confidence {
            stack_result.combined_confidence * (1.0 + gaia_conf * 0.5)
        } else {
            stack_result.combined_confidence
        };

        // Phase 3 info
        let visualization_active = stack_result.visualization.is_some();
        let visualization_confidence = stack_result
            .visualization
            .as_ref()
            .map(|v| v.visualization_confidence);
        let reserve_bursts_occurred = !stack_result.reserve_decompositions.is_empty()
            && stack_result
                .reserve_decompositions
                .values()
                .any(|d| d.burst > 0.0);

        self.update_stats(
            initial_confidence,
            final_confidence,
            &stack_result,
            gaia_contributed,
        );

        if visualization_active {
            self.stats.visualization_passes += 1;
        }
        if reserve_bursts_occurred {
            self.stats.reserve_burst_events += 1;
        }

        IntegrationResult {
            stack_result,
            initial_confidence,
            final_confidence,
            gaia_contributed,
            gaia_confidence,
            active_domains,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            visualization_active,
            visualization_confidence,
            reserve_bursts_occurred,
        }
    }

    /// Apply GAIA intuition enhancement.
    fn apply_gaia_enhancement(
        &mut self,
        input: &str,
        _context: Option<&str>,
        _stack_result: &StackProcessResult,
    ) -> (bool, Option<f32>) {
        self.stats.gaia_invocations += 1;

        // Convert text input to a simple feature vector for GAIA
        // Using character-based hashing for now
        let features = Self::text_to_features(input);

        // Query GAIA for intuition
        if let Ok(query_result) = self.gaia_engine.query(&features) {
            if query_result.confidence > 0.3 {
                return (true, Some(query_result.confidence));
            }
        }

        (false, None)
    }

    /// Apply GAIA with domain hint.
    fn apply_gaia_with_domain(
        &mut self,
        input: &str,
        _context: Option<&str>,
        _domain: Domain,
        _stack_result: &StackProcessResult,
    ) -> (bool, Option<f32>) {
        self.stats.gaia_invocations += 1;

        // Convert text input to a simple feature vector
        let features = Self::text_to_features(input);

        // Query GAIA for intuition
        if let Ok(query_result) = self.gaia_engine.query(&features) {
            if query_result.confidence > 0.3 {
                return (true, Some(query_result.confidence));
            }
        }

        (false, None)
    }

    /// Convert text to a simple feature vector.
    fn text_to_features(input: &str) -> Vec<f32> {
        // Simple character-frequency based features
        let mut features = vec![0.0f32; 26];
        let input_lower = input.to_lowercase();
        let total = input_lower.len().max(1) as f32;

        for c in input_lower.chars() {
            if c.is_ascii_alphabetic() {
                let idx = (c as u8 - b'a') as usize;
                if idx < 26 {
                    features[idx] += 1.0 / total;
                }
            }
        }

        features
    }

    /// Collect active domains from stack result.
    fn collect_active_domains(&self, result: &StackProcessResult) -> Vec<Domain> {
        let mut domains = Vec::new();

        for layer in result.layer_states.keys() {
            let domain = match layer {
                Layer::BasePhysics | Layer::ExtendedPhysics => Domain::Physics,
                Layer::MultilingualProcessing => Domain::Language,
                Layer::GaiaConsciousness => Domain::Consciousness,
                Layer::CollaborativeLearning => Domain::Social,
                Layer::ExternalApis => Domain::External,
                Layer::CrossDomain => Domain::Emergent,
                Layer::PreCognitiveVisualization => Domain::Visualization,
            };
            if !domains.contains(&domain) {
                domains.push(domain);
            }
        }

        domains
    }

    /// Update integration statistics.
    fn update_stats(
        &mut self,
        initial: f32,
        final_conf: f32,
        result: &StackProcessResult,
        gaia_contributed: bool,
    ) {
        if !self.config.track_statistics {
            return;
        }

        self.stats.total_processed += 1;

        if gaia_contributed {
            // Already incremented in apply_gaia_enhancement
        }

        let improvement = final_conf - initial;
        let n = self.stats.total_processed as f32;
        self.stats.avg_confidence_improvement =
            (self.stats.avg_confidence_improvement * (n - 1.0) + improvement) / n;

        if final_conf > self.stats.max_confidence {
            self.stats.max_confidence = final_conf;
        }

        if result.converged {
            self.stats.convergence_count += 1;
        }

        self.stats.total_amplification += result.total_amplification;
    }

    /// Feed a successful pattern to GAIA for learning.
    ///
    /// This uses GAIA's feedback mechanism to reinforce successful patterns.
    pub fn learn_pattern(&mut self, input: &str, _output: &str, success: bool) {
        if self.config.enable_gaia {
            // Convert input to features and register as a pattern
            let features = Self::text_to_features(input);

            // Try to find matching patterns and provide feedback
            if let Ok(result) = self.gaia_engine.query(&features) {
                if let Some(best_match) = result.best_match() {
                    let _ = self.gaia_engine.feedback(&best_match.pattern_id, success);
                }
            }
        }
    }

    /// Get a summary of the integration state.
    pub fn summary(&self) -> String {
        format!(
            "LayerIntegration Summary:\n\
             Processed: {}\n\
             GAIA invocations: {}\n\
             Avg confidence improvement: {:.2}%\n\
             Max confidence: {:.2}\n\
             Convergence rate: {:.1}%\n\
             Total amplification: {:.2}x\n\
             Phase 3 — Visualization passes: {}\n\
             Phase 3 — Reserve burst events: {}",
            self.stats.total_processed,
            self.stats.gaia_invocations,
            self.stats.avg_confidence_improvement * 100.0,
            self.stats.max_confidence,
            if self.stats.total_processed > 0 {
                (self.stats.convergence_count as f32 / self.stats.total_processed as f32) * 100.0
            } else {
                0.0
            },
            if self.stats.total_processed > 0 {
                self.stats.total_amplification / self.stats.total_processed as f32
            } else {
                1.0
            },
            self.stats.visualization_passes,
            self.stats.reserve_burst_events,
        )
    }
}

impl Default for LayerIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for integrating layers with mimicry sessions.
pub trait LayerEnhanced {
    /// Process input with layer amplification.
    fn process_with_layers(&mut self, input: &str, integration: &mut LayerIntegration) -> f32;

    /// Get the amplification benefit from last processing.
    fn last_amplification_benefit(&self) -> Option<f32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_creation() {
        let integration = LayerIntegration::new();
        assert!(integration.config.enable_gaia);
        assert!(!integration.config.enable_external_apis);
    }

    #[test]
    fn test_basic_processing() {
        let mut integration = LayerIntegration::new();
        let result = integration.process("test input", None);

        assert!(result.final_confidence > 0.0);
        assert!(result.processing_time_ms < 5000);
    }

    #[test]
    fn test_domain_processing() {
        let mut integration = LayerIntegration::new();
        let result = integration.process_with_domain("physics calculation", Domain::Physics, None);

        assert!(result.active_domains.contains(&Domain::Physics));
    }

    #[test]
    fn test_statistics_tracking() {
        let mut integration = LayerIntegration::new();

        integration.process("test 1", None);
        integration.process("test 2", None);

        assert_eq!(integration.stats().total_processed, 2);
    }

    #[test]
    fn test_pattern_learning() {
        let mut integration = LayerIntegration::new();

        integration.learn_pattern("input", "output", true);
        // No panic = success
    }
}
