//! Layer stack orchestrator for the 8-layer multiplicative integration system.
//!
//! The stack coordinates information flow through all layers, managing
//! bidirectional propagation and multiplicative amplification.
//!
//! # Phase 3 Integration
//!
//! The stack optionally integrates three Phase 3 subsystems:
//! - **VisualizationEngine**: Pre-cognitive visualization fires BEFORE forward pass
//! - **FractionalReserve**: Splits confidence into active/held, with burst releases
//! - **DistributionResonanceSystem**: Variance-penalized resonance for amplification

use std::collections::HashMap;
use std::sync::Arc;

use super::adaptive::{
    AdaptiveCapConfig, AdaptiveConfidenceCap, DynamicBridgeWeighting, DynamicWeightConfig,
    OnlineLearningConfig, OnlineLearningSystem,
};
use super::bridge::{BidirectionalBridge, BridgeNetwork};
use super::distribution_resonance::DistributionResonanceSystem;
use super::layer::{Layer, LayerConfig, LayerSignal, LayerState};
use super::octo_braid::{BraidSignals, OctoBraid, OctoBraidConfig};
use super::registry::LayerRegistry;
use super::reserve::{ConfidenceDecomposition, FractionalReserve};
use super::visualization::{VisualizationEngine, VisualizationResult};

/// Configuration for the layer stack.
#[derive(Debug, Clone)]
pub struct LayerStackConfig {
    /// Maximum number of full-stack amplification cycles.
    pub max_stack_iterations: u32,
    /// Convergence threshold for stopping iteration.
    pub convergence_threshold: f32,
    /// Global amplification factor.
    pub global_amplification: f32,
    /// Whether to enable backward propagation by default.
    pub enable_backward_propagation: bool,
    /// Minimum confidence to continue propagation.
    pub min_propagation_confidence: f32,
    /// Per-layer configuration overrides.
    pub layer_configs: HashMap<Layer, LayerConfig>,
    /// Maximum allowed confidence value (prevents divergence).
    pub max_confidence: f32,
    /// Maximum allowed total amplification factor.
    pub max_total_amplification: f32,
    /// Damping factor applied to amplification (0.0 - 1.0).
    pub amplification_damping: f32,
    /// Whether to enable the Pre-Cognitive Visualization Engine (Phase 3).
    pub enable_visualization: bool,
    /// Whether to enable the Fractional Reserve Model (Phase 3).
    pub enable_reserve: bool,
    /// Whether to enable Distribution-based Resonance (Phase 3).
    pub enable_distribution_resonance: bool,
    /// Whether to enable the Adaptive Confidence Cap (Phase 5).
    pub enable_adaptive_cap: bool,
    /// Configuration for the adaptive confidence cap (Phase 5).
    pub adaptive_cap_config: AdaptiveCapConfig,
    /// Whether to enable Dynamic Bridge Weighting (Phase 5).
    pub enable_dynamic_weighting: bool,
    /// Configuration for dynamic bridge weighting (Phase 5).
    pub dynamic_weight_config: DynamicWeightConfig,
    /// Whether to enable the Online Learning System (Phase 5).
    pub enable_online_learning: bool,
    /// Configuration for online learning (Phase 5).
    pub online_learning_config: OnlineLearningConfig,
    /// Whether to enable the OCTO Braid cross-modulation (Phase 6).
    pub enable_octo_braid: bool,
    /// Configuration for the OCTO Braid (Phase 6).
    pub octo_braid_config: OctoBraidConfig,
}

impl Default for LayerStackConfig {
    fn default() -> Self {
        Self {
            max_stack_iterations: 5,
            convergence_threshold: 0.01,
            global_amplification: 1.1,
            enable_backward_propagation: true,
            min_propagation_confidence: 0.1,
            layer_configs: HashMap::new(),
            max_confidence: 2.0,                  // Cap confidence at 2.0
            max_total_amplification: 10.0,        // Cap total amplification at 10x
            amplification_damping: 0.8,           // Dampen amplification to 80%
            enable_visualization: false,          // Phase 3: off by default for backward compat
            enable_reserve: false,                // Phase 3: off by default
            enable_distribution_resonance: false, // Phase 3: off by default
            enable_adaptive_cap: false,           // Phase 5: off by default
            adaptive_cap_config: AdaptiveCapConfig::default(),
            enable_dynamic_weighting: false, // Phase 5: off by default
            dynamic_weight_config: DynamicWeightConfig::default(),
            enable_online_learning: false, // Phase 5: off by default
            online_learning_config: OnlineLearningConfig::default(),
            enable_octo_braid: false, // Phase 6: off by default
            octo_braid_config: OctoBraidConfig::default(),
        }
    }
}

impl LayerStackConfig {
    /// Create a new configuration with the given settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the global amplification factor.
    pub fn with_global_amplification(mut self, factor: f32) -> Self {
        self.global_amplification = factor;
        self
    }

    /// Set max stack iterations.
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_stack_iterations = max;
        self
    }

    /// Add a layer-specific configuration.
    pub fn with_layer_config(mut self, config: LayerConfig) -> Self {
        self.layer_configs.insert(config.layer, config);
        self
    }

    /// Disable backward propagation.
    pub fn without_backward_propagation(mut self) -> Self {
        self.enable_backward_propagation = false;
        self
    }

    /// Set maximum confidence value.
    pub fn with_max_confidence(mut self, max: f32) -> Self {
        self.max_confidence = max;
        self
    }

    /// Set maximum total amplification.
    pub fn with_max_total_amplification(mut self, max: f32) -> Self {
        self.max_total_amplification = max;
        self
    }

    /// Set amplification damping factor.
    pub fn with_amplification_damping(mut self, damping: f32) -> Self {
        self.amplification_damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Enable the Pre-Cognitive Visualization Engine.
    pub fn with_visualization(mut self) -> Self {
        self.enable_visualization = true;
        self
    }

    /// Enable the Fractional Reserve Model.
    pub fn with_reserve(mut self) -> Self {
        self.enable_reserve = true;
        self
    }

    /// Enable Distribution-based Resonance.
    pub fn with_distribution_resonance(mut self) -> Self {
        self.enable_distribution_resonance = true;
        self
    }

    /// Enable all Phase 3 subsystems.
    pub fn with_phase3(mut self) -> Self {
        self.enable_visualization = true;
        self.enable_reserve = true;
        self.enable_distribution_resonance = true;
        self
    }

    /// Enable the Adaptive Confidence Cap (Phase 5).
    pub fn with_adaptive_cap(mut self) -> Self {
        self.enable_adaptive_cap = true;
        self.adaptive_cap_config.enabled = true;
        self
    }

    /// Enable the Adaptive Confidence Cap with custom config (Phase 5).
    pub fn with_adaptive_cap_config(mut self, config: AdaptiveCapConfig) -> Self {
        self.enable_adaptive_cap = true;
        self.adaptive_cap_config = config;
        self.adaptive_cap_config.enabled = true;
        self
    }

    /// Enable Dynamic Bridge Weighting (Phase 5).
    pub fn with_dynamic_weighting(mut self) -> Self {
        self.enable_dynamic_weighting = true;
        self.dynamic_weight_config.enabled = true;
        self
    }

    /// Enable Dynamic Bridge Weighting with custom config (Phase 5).
    pub fn with_dynamic_weight_config(mut self, config: DynamicWeightConfig) -> Self {
        self.enable_dynamic_weighting = true;
        self.dynamic_weight_config = config;
        self.dynamic_weight_config.enabled = true;
        self
    }

    /// Enable the Online Learning System (Phase 5).
    pub fn with_online_learning(mut self) -> Self {
        self.enable_online_learning = true;
        self.online_learning_config.enabled = true;
        self
    }

    /// Enable the Online Learning System with custom config (Phase 5).
    pub fn with_online_learning_config(mut self, config: OnlineLearningConfig) -> Self {
        self.enable_online_learning = true;
        self.online_learning_config = config;
        self.online_learning_config.enabled = true;
        self
    }

    /// Enable all Phase 5 adaptive subsystems.
    pub fn with_phase5(mut self) -> Self {
        self.enable_adaptive_cap = true;
        self.adaptive_cap_config.enabled = true;
        self.enable_dynamic_weighting = true;
        self.dynamic_weight_config.enabled = true;
        self.enable_online_learning = true;
        self.online_learning_config.enabled = true;
        self
    }

    /// Enable ALL subsystems (Phase 3 + Phase 5).
    pub fn with_all_subsystems(self) -> Self {
        self.with_phase3().with_phase5()
    }

    /// Enable the OCTO Braid cross-modulation (Phase 6).
    pub fn with_octo_braid(mut self) -> Self {
        self.enable_octo_braid = true;
        self.octo_braid_config.enabled = true;
        self
    }

    /// Enable the OCTO Braid with custom config (Phase 6).
    pub fn with_octo_braid_config(mut self, config: OctoBraidConfig) -> Self {
        self.enable_octo_braid = true;
        self.octo_braid_config = config;
        self.octo_braid_config.enabled = true;
        self
    }

    /// Enable ALL subsystems including Phase 6 OCTO Braid (Phase 3 + Phase 5 + Phase 6).
    pub fn with_all_phases(self) -> Self {
        self.with_phase3().with_phase5().with_octo_braid()
    }

    /// Clamp a confidence value to the configured maximum.
    #[inline]
    pub fn clamp_confidence(&self, confidence: f32) -> f32 {
        confidence.clamp(0.0, self.max_confidence)
    }

    /// Clamp a confidence value to a specific cap.
    #[inline]
    pub fn clamp_confidence_to(&self, confidence: f32, cap: f32) -> f32 {
        confidence.clamp(0.0, cap)
    }
}

/// Result of processing through the entire stack.
#[derive(Debug, Clone)]
pub struct StackProcessResult {
    /// Final states for each layer.
    pub layer_states: HashMap<Layer, LayerState>,
    /// Combined confidence across all layers.
    pub combined_confidence: f32,
    /// Total amplification achieved.
    pub total_amplification: f32,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Per-layer confidence values.
    pub layer_confidences: HashMap<Layer, f32>,
    /// Trace of signals for debugging.
    pub signal_trace: Vec<LayerSignal>,
    /// Visualization result from the pre-pass (Phase 3, None if disabled).
    pub visualization: Option<VisualizationResult>,
    /// Per-layer confidence decompositions from the reserve system (Phase 3).
    pub reserve_decompositions: HashMap<Layer, ConfidenceDecomposition>,
    /// Per-layer-pair distribution resonance values (Phase 3, keyed by (layer_a_num, layer_b_num)).
    pub distribution_resonances: HashMap<(u8, u8), f32>,
    /// The effective confidence cap used for this result (Phase 5).
    pub effective_cap: f32,
    /// Per-bridge dynamic weights at time of processing (Phase 5).
    pub bridge_weights: HashMap<(u8, u8), f32>,
    /// Per-layer recommended amplification from online learning (Phase 5).
    pub learning_amplifications: HashMap<Layer, f32>,
}

impl StackProcessResult {
    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            layer_states: HashMap::new(),
            combined_confidence: 0.0,
            total_amplification: 1.0,
            iterations: 0,
            converged: false,
            layer_confidences: HashMap::new(),
            signal_trace: Vec::new(),
            visualization: None,
            reserve_decompositions: HashMap::new(),
            distribution_resonances: HashMap::new(),
            effective_cap: 0.0,
            bridge_weights: HashMap::new(),
            learning_amplifications: HashMap::new(),
        }
    }

    /// Get the state for a specific layer.
    pub fn get_state(&self, layer: Layer) -> Option<&LayerState> {
        self.layer_states.get(&layer)
    }

    /// Check if processing was successful.
    pub fn is_successful(&self, min_confidence: f32) -> bool {
        self.combined_confidence >= min_confidence
    }
}

/// The main orchestrator for the 8-layer system.
pub struct LayerStack {
    /// Layer registry.
    registry: LayerRegistry,
    /// Bridge network.
    bridge_network: BridgeNetwork,
    /// Stack configuration.
    config: LayerStackConfig,
    /// Current layer states.
    current_states: HashMap<Layer, LayerState>,
    /// Processing statistics.
    stats: StackStats,
    /// Phase 3: Visualization Engine (optional).
    visualization_engine: Option<VisualizationEngine>,
    /// Phase 3: Fractional Reserve system (optional).
    fractional_reserve: Option<FractionalReserve>,
    /// Phase 3: Distribution Resonance system (optional).
    distribution_resonance: Option<DistributionResonanceSystem>,
    /// Phase 5: Adaptive Confidence Cap (optional).
    adaptive_cap: Option<AdaptiveConfidenceCap>,
    /// Phase 5: Dynamic Bridge Weighting (optional).
    dynamic_weighting: Option<DynamicBridgeWeighting>,
    /// Phase 5: Online Learning System (optional).
    online_learning: Option<OnlineLearningSystem>,
    /// Phase 6: OCTO Braid cross-modulation (optional).
    octo_braid: Option<OctoBraid>,
}

/// Statistics for stack processing.
#[derive(Debug, Clone, Default)]
pub struct StackStats {
    pub total_forward_propagations: u64,
    pub total_backward_propagations: u64,
    pub total_amplifications: u64,
    pub average_confidence: f32,
    pub max_confidence_achieved: f32,
    pub convergence_count: u64,
    pub non_convergence_count: u64,
    /// Phase 3: Total visualization passes.
    pub visualization_passes: u64,
    /// Phase 3: Total reserve burst events.
    pub reserve_burst_events: u64,
    /// Phase 3: Total reserve burst confidence released.
    pub reserve_burst_confidence: f32,
    /// Phase 5: Total adaptive cap adjustments.
    pub adaptive_cap_adjustments: u64,
    /// Phase 5: Total dynamic weight updates.
    pub dynamic_weight_updates: u64,
    /// Phase 5: Total online learning observations.
    pub online_learning_observations: u64,
    /// Phase 6: Total braid modulation calls.
    pub braid_modulations: u64,
    /// Phase 6: Total System 2 activations from braid.
    pub braid_system2_activations: u64,
}

impl LayerStack {
    /// Create a new layer stack with default configuration.
    pub fn new() -> Self {
        Self::with_config(LayerStackConfig::default())
    }

    /// Create a new layer stack with custom configuration.
    pub fn with_config(config: LayerStackConfig) -> Self {
        let mut registry = LayerRegistry::new();

        // Apply layer-specific configs
        for (layer, layer_config) in &config.layer_configs {
            registry.configure(*layer, layer_config.clone());
        }

        let mut bridge_network = BridgeNetwork::new();
        bridge_network.set_global_amplification(config.global_amplification);

        // Initialize Phase 3 subsystems if enabled
        let visualization_engine = if config.enable_visualization {
            Some(VisualizationEngine::with_defaults())
        } else {
            None
        };

        let fractional_reserve = if config.enable_reserve {
            Some(FractionalReserve::with_defaults())
        } else {
            None
        };

        let distribution_resonance = if config.enable_distribution_resonance {
            Some(DistributionResonanceSystem::with_defaults())
        } else {
            None
        };

        // Initialize Phase 5 subsystems if enabled
        let adaptive_cap = if config.enable_adaptive_cap {
            Some(AdaptiveConfidenceCap::with_config(
                config.adaptive_cap_config.clone(),
            ))
        } else {
            None
        };

        let dynamic_weighting = if config.enable_dynamic_weighting {
            Some(DynamicBridgeWeighting::with_config(
                config.dynamic_weight_config.clone(),
            ))
        } else {
            None
        };

        let online_learning = if config.enable_online_learning {
            Some(OnlineLearningSystem::with_config(
                config.online_learning_config.clone(),
            ))
        } else {
            None
        };

        let octo_braid = if config.enable_octo_braid {
            Some(OctoBraid::with_config(config.octo_braid_config.clone()))
        } else {
            None
        };

        Self {
            registry,
            bridge_network,
            config,
            current_states: HashMap::new(),
            stats: StackStats::default(),
            visualization_engine,
            fractional_reserve,
            distribution_resonance,
            adaptive_cap,
            dynamic_weighting,
            online_learning,
            octo_braid,
        }
    }

    /// Get a reference to the registry.
    pub fn registry(&self) -> &LayerRegistry {
        &self.registry
    }

    /// Get a mutable reference to the registry.
    pub fn registry_mut(&mut self) -> &mut LayerRegistry {
        &mut self.registry
    }

    /// Get a reference to the bridge network.
    pub fn bridge_network(&self) -> &BridgeNetwork {
        &self.bridge_network
    }

    /// Get a mutable reference to the bridge network.
    pub fn bridge_network_mut(&mut self) -> &mut BridgeNetwork {
        &mut self.bridge_network
    }

    /// Register a bridge in the network.
    pub fn register_bridge(&mut self, bridge: Arc<dyn BidirectionalBridge>) {
        self.bridge_network.register(bridge);
    }

    /// Get the current configuration.
    pub fn config(&self) -> &LayerStackConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: LayerStackConfig) {
        self.bridge_network
            .set_global_amplification(config.global_amplification);
        self.config = config;
    }

    /// Get processing statistics.
    pub fn stats(&self) -> &StackStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = StackStats::default();
    }

    /// Get a reference to the visualization engine (Phase 3).
    pub fn visualization_engine(&self) -> Option<&VisualizationEngine> {
        self.visualization_engine.as_ref()
    }

    /// Get a mutable reference to the visualization engine (Phase 3).
    pub fn visualization_engine_mut(&mut self) -> Option<&mut VisualizationEngine> {
        self.visualization_engine.as_mut()
    }

    /// Get a reference to the fractional reserve system (Phase 3).
    pub fn fractional_reserve(&self) -> Option<&FractionalReserve> {
        self.fractional_reserve.as_ref()
    }

    /// Get a mutable reference to the fractional reserve system (Phase 3).
    pub fn fractional_reserve_mut(&mut self) -> Option<&mut FractionalReserve> {
        self.fractional_reserve.as_mut()
    }

    /// Get a reference to the distribution resonance system (Phase 3).
    pub fn distribution_resonance(&self) -> Option<&DistributionResonanceSystem> {
        self.distribution_resonance.as_ref()
    }

    /// Get a mutable reference to the distribution resonance system (Phase 3).
    pub fn distribution_resonance_mut(&mut self) -> Option<&mut DistributionResonanceSystem> {
        self.distribution_resonance.as_mut()
    }

    /// Get a reference to the adaptive confidence cap (Phase 5).
    pub fn adaptive_cap(&self) -> Option<&AdaptiveConfidenceCap> {
        self.adaptive_cap.as_ref()
    }

    /// Get a mutable reference to the adaptive confidence cap (Phase 5).
    pub fn adaptive_cap_mut(&mut self) -> Option<&mut AdaptiveConfidenceCap> {
        self.adaptive_cap.as_mut()
    }

    /// Get a reference to the dynamic bridge weighting (Phase 5).
    pub fn dynamic_weighting(&self) -> Option<&DynamicBridgeWeighting> {
        self.dynamic_weighting.as_ref()
    }

    /// Get a mutable reference to the dynamic bridge weighting (Phase 5).
    pub fn dynamic_weighting_mut(&mut self) -> Option<&mut DynamicBridgeWeighting> {
        self.dynamic_weighting.as_mut()
    }

    /// Get a reference to the online learning system (Phase 5).
    pub fn online_learning(&self) -> Option<&OnlineLearningSystem> {
        self.online_learning.as_ref()
    }

    /// Get a mutable reference to the online learning system (Phase 5).
    pub fn online_learning_mut(&mut self) -> Option<&mut OnlineLearningSystem> {
        self.online_learning.as_mut()
    }

    /// Get a reference to the OCTO braid (Phase 6).
    pub fn octo_braid(&self) -> Option<&OctoBraid> {
        self.octo_braid.as_ref()
    }

    /// Get a mutable reference to the OCTO braid (Phase 6).
    pub fn octo_braid_mut(&mut self) -> Option<&mut OctoBraid> {
        self.octo_braid.as_mut()
    }

    /// Set the difficulty hint on the OCTO braid (Phase 6).
    /// This allows external callers (e.g. contest) to pass scenario difficulty.
    pub fn set_difficulty_hint(&mut self, hint: Option<f32>) {
        if let Some(ref mut braid) = self.octo_braid {
            braid.set_difficulty_hint(hint);
        }
    }

    /// Get the current state for a layer.
    pub fn get_current_state(&self, layer: Layer) -> Option<&LayerState> {
        self.current_states.get(&layer)
    }

    /// Set the state for a layer.
    pub fn set_state(&mut self, state: LayerState) {
        self.current_states.insert(state.layer, state);
    }

    /// Clear all current states.
    pub fn clear_states(&mut self) {
        self.current_states.clear();
    }

    /// Process input through the entire stack with forward propagation.
    pub fn process_forward(&mut self, input: LayerState) -> StackProcessResult {
        let mut result = StackProcessResult::empty();
        let start_layer = input.layer;

        // Phase 5: Compute effective confidence cap
        let effective_cap = if let Some(ref cap) = self.adaptive_cap {
            // Use 0.0 variance initially (no layer spread yet); will refine after propagation
            let cap_val = cap.effective_cap(input.confidence, 0.0);
            result.effective_cap = cap_val;
            cap_val
        } else {
            result.effective_cap = self.config.max_confidence;
            self.config.max_confidence
        };

        // Phase 5: Snapshot bridge weights into result
        if let Some(ref dw) = self.dynamic_weighting {
            for (&key, bw) in dw.all_weights() {
                result.bridge_weights.insert(key, bw.weight);
            }
        }

        // Phase 3: Pre-Cognitive Visualization pass (fires BEFORE forward)
        if let Some(vis_engine) = &mut self.visualization_engine {
            let vis_result = vis_engine.visualize(&input);
            self.stats.visualization_passes += 1;

            // Apply pre-warm signals: boost the input confidence for each targeted layer
            // The visualization result is stored in the StackProcessResult for downstream use
            result.visualization = Some(vis_result);
        }

        // Phase 3: Apply reserve splitting to the input confidence
        let effective_input = if let Some(reserve) = &mut self.fractional_reserve {
            let resonance_val = self
                .distribution_resonance
                .as_ref()
                .map(|dr| dr.resonance(start_layer, start_layer))
                .unwrap_or(0.5);
            let decomp = reserve.process(start_layer, input.confidence, resonance_val);
            if decomp.burst > 0.0 {
                self.stats.reserve_burst_events += 1;
                self.stats.reserve_burst_confidence += decomp.burst;
            }
            result.reserve_decompositions.insert(start_layer, decomp);
            let mut modified_input = input.clone();
            modified_input.confidence = self
                .config
                .clamp_confidence_to(decomp.effective, effective_cap);
            modified_input
        } else {
            input.clone()
        };

        // Phase 3: Observe confidence in distribution resonance system
        if let Some(dr) = &mut self.distribution_resonance {
            dr.observe(start_layer, effective_input.confidence);
        }

        // Set the initial state
        self.current_states
            .insert(start_layer, effective_input.clone());
        result
            .layer_states
            .insert(start_layer, effective_input.clone());
        result
            .layer_confidences
            .insert(start_layer, effective_input.confidence);

        // Get layers to process (from start layer upward)
        let layers_to_process: Vec<Layer> = Layer::all()
            .iter()
            .filter(|&&l| l.number() > start_layer.number() && self.registry.is_enabled(l))
            .copied()
            .collect();

        // Forward propagation through each layer
        let mut current_state = effective_input;
        for target_layer in layers_to_process {
            // Find bridge from current layer to target
            if let Some(bridge) = self
                .bridge_network
                .bridge_between(current_state.layer, target_layer)
            {
                match bridge.forward(&current_state) {
                    Ok(mut new_state) => {
                        self.stats.total_forward_propagations += 1;

                        // Phase 3: Apply pre-warm boost from visualization
                        if let Some(ref vis) = result.visualization {
                            if let Some(&boost) = vis.pre_warm_signals.get(&target_layer) {
                                new_state.confidence = self.config.clamp_confidence_to(
                                    new_state.confidence * (1.0 + boost * 0.1),
                                    effective_cap,
                                );
                            }
                        }

                        // Phase 3: Apply reserve splitting to this layer's confidence
                        if let Some(reserve) = &mut self.fractional_reserve {
                            let resonance_val = self
                                .distribution_resonance
                                .as_ref()
                                .map(|dr| dr.resonance(current_state.layer, target_layer))
                                .unwrap_or(bridge.resonance());
                            let decomp =
                                reserve.process(target_layer, new_state.confidence, resonance_val);
                            if decomp.burst > 0.0 {
                                self.stats.reserve_burst_events += 1;
                                self.stats.reserve_burst_confidence += decomp.burst;
                            }
                            result.reserve_decompositions.insert(target_layer, decomp);
                            new_state.confidence = self
                                .config
                                .clamp_confidence_to(decomp.effective, effective_cap);
                        }

                        // Phase 3: Observe confidence in distribution resonance
                        if let Some(dr) = &mut self.distribution_resonance {
                            dr.observe(target_layer, new_state.confidence);
                        }

                        // Create signal for trace
                        let signal =
                            LayerSignal::new(current_state.layer, target_layer, new_state.clone());
                        result.signal_trace.push(signal);

                        // Update states
                        self.current_states.insert(target_layer, new_state.clone());
                        result.layer_states.insert(target_layer, new_state.clone());
                        result
                            .layer_confidences
                            .insert(target_layer, new_state.confidence);

                        current_state = new_state;
                    }
                    Err(_) => {
                        // Bridge failed, stop propagation to this path
                        break;
                    }
                }
            }
        }

        // Phase 3: Compute pairwise distribution resonances for the result
        if let Some(dr) = &self.distribution_resonance {
            let active_layers: Vec<Layer> = result.layer_states.keys().copied().collect();
            result.distribution_resonances = dr.pairwise_resonances(&active_layers);
        }

        // Calculate combined confidence
        result.combined_confidence = self.calculate_combined_confidence(&result.layer_confidences);
        // Phase 5: Re-clamp to effective cap (may be lower than config.max_confidence)
        result.combined_confidence = self
            .config
            .clamp_confidence_to(result.combined_confidence, effective_cap);
        result.iterations = 1;

        // Phase 5: Observe layers in online learning system
        if let Some(ref mut ol) = self.online_learning {
            for (&layer, &conf) in &result.layer_confidences {
                ol.observe_layer(layer, conf, result.combined_confidence);
            }
            // Snapshot learning amplifications
            for &layer in result.layer_confidences.keys() {
                let amp = ol.recommended_amplification(layer);
                result.learning_amplifications.insert(layer, amp);
            }
        }

        result
    }

    /// Process with bidirectional amplification.
    pub fn process_bidirectional(&mut self, input: LayerState) -> StackProcessResult {
        let mut result = self.process_forward(input);

        if !self.config.enable_backward_propagation {
            return result;
        }

        let mut previous_confidence = result.combined_confidence;

        for iteration in 0..self.config.max_stack_iterations {
            // Phase 6: Compute braid signals at the START of each iteration
            let braid_signals = self.compute_braid_signals(&result, iteration);

            // Phase 6: Feed braid effective_cap into reserve threshold updates
            if let Some(ref mut reserve) = self.fractional_reserve {
                if let Some(ref signals) = braid_signals {
                    // Update thresholds: lower cap → higher quality requirement → raise threshold
                    let quality_hint = signals.effective_cap / self.config.max_confidence;
                    for &layer in result.layer_confidences.keys() {
                        reserve.update_threshold(layer, quality_hint);
                    }
                }
            }

            // Backward propagation (with braid signals)
            self.propagate_backward(&mut result, &braid_signals);

            // Forward propagation again (with braid signals)
            self.propagate_forward_from_states(&mut result, &braid_signals);

            // Amplification across all bridges (with braid signals)
            self.amplify_all_bridges(&mut result, &braid_signals);

            // Check convergence
            let confidence_change = (result.combined_confidence - previous_confidence).abs();
            if confidence_change < self.config.convergence_threshold {
                result.converged = true;
                self.stats.convergence_count += 1;
                break;
            }

            previous_confidence = result.combined_confidence;
            result.iterations = iteration + 2; // +1 for initial forward, +1 for 0-indexed
        }

        if !result.converged {
            self.stats.non_convergence_count += 1;
        }

        // Phase 3: Record outcome back to visualization engine for fidelity tracking
        if let Some(vis_engine) = &mut self.visualization_engine {
            if let Some(ref vis_result) = result.visualization {
                let layer_actuals: HashMap<Layer, f32> = result.layer_confidences.clone();
                vis_engine.record_outcome(
                    vis_result.visualization_confidence,
                    result.combined_confidence,
                    &layer_actuals,
                );
            }
        }

        // Phase 3: Update pairwise distribution resonances with final values
        if let Some(dr) = &self.distribution_resonance {
            let active_layers: Vec<Layer> = result.layer_states.keys().copied().collect();
            result.distribution_resonances = dr.pairwise_resonances(&active_layers);
        }

        // Phase 5: Recompute effective cap with actual layer variance
        if let Some(ref cap) = self.adaptive_cap {
            let layer_variance = Self::compute_layer_variance(&result.layer_confidences);
            let input_confidence = result
                .layer_confidences
                .get(&Layer::BasePhysics)
                .copied()
                .unwrap_or(0.5);
            let effective = cap.effective_cap(input_confidence, layer_variance);
            result.effective_cap = effective;
            // Re-clamp combined confidence to the refined effective cap
            result.combined_confidence = self
                .config
                .clamp_confidence_to(result.combined_confidence, effective);
            self.stats.adaptive_cap_adjustments += 1;
        }

        // Phase 6: Final clamp to braid's effective_cap (which may be lower than Phase 5's cap)
        if let Some(ref mut braid) = self.octo_braid {
            let input_confidence = result
                .layer_confidences
                .get(&Layer::BasePhysics)
                .copied()
                .unwrap_or(0.5);
            let final_signals = braid.modulate(
                input_confidence,
                &result.layer_confidences,
                result.effective_cap,
                result.iterations,
            );
            let braid_cap = final_signals.effective_cap;
            if braid_cap < result.effective_cap {
                result.effective_cap = braid_cap;
                result.combined_confidence = self
                    .config
                    .clamp_confidence_to(result.combined_confidence, braid_cap);
                // Also clamp individual layer confidences to braid cap
                for conf in result.layer_confidences.values_mut() {
                    *conf = conf.min(braid_cap);
                }
            }
        }

        // Phase 5: Record result in adaptive cap for saturation tracking
        if let Some(ref mut cap) = self.adaptive_cap {
            cap.record_result(result.combined_confidence, result.effective_cap);
        }

        // Phase 5: Update online learning with global observation
        if let Some(ref mut ol) = self.online_learning {
            // Observe each layer
            for (&layer, &conf) in &result.layer_confidences {
                ol.observe_layer(layer, conf, result.combined_confidence);
            }
            ol.observe_global(result.combined_confidence);
            self.stats.online_learning_observations += 1;

            // Snapshot learning amplifications
            for &layer in result.layer_confidences.keys() {
                let amp = ol.recommended_amplification(layer);
                result.learning_amplifications.insert(layer, amp);
            }
        }

        // Phase 5: Snapshot bridge weights into result
        if let Some(ref dw) = self.dynamic_weighting {
            for (&key, bw) in dw.all_weights() {
                result.bridge_weights.insert(key, bw.weight);
            }
        }

        // Update statistics
        self.update_stats(&result);

        result
    }

    /// Propagate backward through the stack.
    fn propagate_backward(
        &mut self,
        result: &mut StackProcessResult,
        braid_signals: &Option<BraidSignals>,
    ) {
        let effective_cap = self.effective_cap_from_braid(braid_signals, result);

        let layers: Vec<Layer> = result
            .layer_states
            .keys()
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        for (i, &source_layer) in layers.iter().enumerate() {
            if i + 1 >= layers.len() {
                break;
            }

            let target_layer = layers[i + 1];

            if let Some(bridge) = self
                .bridge_network
                .bridge_between(source_layer, target_layer)
            {
                if let Some(source_state) = result.layer_states.get(&source_layer) {
                    if let Ok(refined_state) = bridge.backward(source_state) {
                        self.stats.total_backward_propagations += 1;

                        let signal =
                            LayerSignal::new(source_layer, target_layer, refined_state.clone());
                        result.signal_trace.push(signal);

                        // Merge refined state with existing
                        if let Some(existing) = result.layer_states.get_mut(&target_layer) {
                            let raw_confidence =
                                (existing.confidence + refined_state.confidence) / 2.0;

                            // Phase 6: Apply reserve splitting during backward pass
                            let after_reserve = if let Some(reserve) = &mut self.fractional_reserve
                            {
                                let resonance_val = self
                                    .distribution_resonance
                                    .as_ref()
                                    .map(|dr| dr.resonance(source_layer, target_layer))
                                    .unwrap_or(0.5);
                                // Scale resonance by braid sensitivity
                                let scaled_resonance = braid_signals
                                    .as_ref()
                                    .map(|s| resonance_val * s.resonance_sensitivity)
                                    .unwrap_or(resonance_val);
                                let decomp =
                                    reserve.process(target_layer, raw_confidence, scaled_resonance);
                                if decomp.burst > 0.0 {
                                    self.stats.reserve_burst_events += 1;
                                    self.stats.reserve_burst_confidence += decomp.burst;
                                }
                                result.reserve_decompositions.insert(target_layer, decomp);
                                decomp.effective
                            } else {
                                raw_confidence
                            };

                            // Phase 6: Clamp to effective_cap (NOT config.max_confidence)
                            existing.confidence = self
                                .config
                                .clamp_confidence_to(after_reserve, effective_cap);
                            existing.increment_amplification();
                        }
                    }
                }
            }
        }
    }

    /// Propagate forward from current states.
    fn propagate_forward_from_states(
        &mut self,
        result: &mut StackProcessResult,
        braid_signals: &Option<BraidSignals>,
    ) {
        let effective_cap = self.effective_cap_from_braid(braid_signals, result);
        let damping = self.damping_from_braid(braid_signals);

        let layers: Vec<Layer> = result.layer_states.keys().copied().collect();

        for &source_layer in &layers {
            for &target_layer in &layers {
                if target_layer.number() <= source_layer.number() {
                    continue;
                }

                if let Some(bridge) = self
                    .bridge_network
                    .bridge_between(source_layer, target_layer)
                {
                    if let Some(source_state) = result.layer_states.get(&source_layer) {
                        if let Ok(new_state) = bridge.forward(source_state) {
                            self.stats.total_forward_propagations += 1;

                            // Merge with existing state (with braid damping and clamping)
                            if let Some(existing) = result.layer_states.get_mut(&target_layer) {
                                // Phase 6: Use braid's global_damping instead of config damping
                                let raw_confidence = (existing.confidence * 0.7
                                    + new_state.confidence * 0.3)
                                    * (1.0 + (self.config.global_amplification - 1.0) * damping);

                                // Phase 6: Apply reserve splitting during forward re-pass
                                let after_reserve =
                                    if let Some(reserve) = &mut self.fractional_reserve {
                                        let resonance_val = self
                                            .distribution_resonance
                                            .as_ref()
                                            .map(|dr| dr.resonance(source_layer, target_layer))
                                            .unwrap_or(bridge.resonance());
                                        let scaled_resonance = braid_signals
                                            .as_ref()
                                            .map(|s| resonance_val * s.resonance_sensitivity)
                                            .unwrap_or(resonance_val);
                                        let decomp = reserve.process(
                                            target_layer,
                                            raw_confidence,
                                            scaled_resonance,
                                        );
                                        if decomp.burst > 0.0 {
                                            self.stats.reserve_burst_events += 1;
                                            self.stats.reserve_burst_confidence += decomp.burst;
                                        }
                                        result.reserve_decompositions.insert(target_layer, decomp);
                                        decomp.effective
                                    } else {
                                        raw_confidence
                                    };

                                // Phase 6: Clamp to effective_cap (NOT config.max_confidence)
                                existing.confidence = self
                                    .config
                                    .clamp_confidence_to(after_reserve, effective_cap);
                                existing.increment_amplification();
                            }
                        }
                    }
                }
            }
        }

        // Recalculate combined confidence
        result.layer_confidences = result
            .layer_states
            .iter()
            .map(|(l, s)| (*l, s.confidence))
            .collect();
        result.combined_confidence = self.calculate_combined_confidence(&result.layer_confidences);
        // Phase 6: Re-clamp to effective_cap
        result.combined_confidence = self
            .config
            .clamp_confidence_to(result.combined_confidence, effective_cap);
    }

    /// Run amplification across all bridges.
    fn amplify_all_bridges(
        &mut self,
        result: &mut StackProcessResult,
        braid_signals: &Option<BraidSignals>,
    ) {
        let effective_cap = self.effective_cap_from_braid(braid_signals, result);
        let damping = self.damping_from_braid(braid_signals);

        let bridges = self.bridge_network.bridges().to_vec();

        for bridge in bridges {
            let source = bridge.source_layer();
            let target = bridge.target_layer();

            let (source_state, target_state) = {
                let s = result.layer_states.get(&source).cloned();
                let t = result.layer_states.get(&target).cloned();
                match (s, t) {
                    (Some(s), Some(t)) => (s, t),
                    _ => continue,
                }
            };

            let layer_config = self.config.layer_configs.get(&source);
            let max_iterations = layer_config
                .map(|c| c.max_amplification_iterations)
                .unwrap_or(10);

            if let Ok(amp_result) = bridge.amplify(&source_state, &target_state, max_iterations) {
                self.stats.total_amplifications += 1;

                // Update states with amplified versions
                let mut up_state = amp_result.up_state;
                let mut down_state = amp_result.down_state;

                // Phase 6: Apply reserve splitting to amplified states
                if let Some(reserve) = &mut self.fractional_reserve {
                    let resonance_val = self
                        .distribution_resonance
                        .as_ref()
                        .map(|dr| dr.resonance(source, target))
                        .unwrap_or(bridge.resonance());
                    let scaled_resonance = braid_signals
                        .as_ref()
                        .map(|s| resonance_val * s.resonance_sensitivity)
                        .unwrap_or(resonance_val);

                    let decomp_up = reserve.process(source, up_state.confidence, scaled_resonance);
                    let decomp_down =
                        reserve.process(target, down_state.confidence, scaled_resonance);

                    if decomp_up.burst > 0.0 {
                        self.stats.reserve_burst_events += 1;
                        self.stats.reserve_burst_confidence += decomp_up.burst;
                    }
                    if decomp_down.burst > 0.0 {
                        self.stats.reserve_burst_events += 1;
                        self.stats.reserve_burst_confidence += decomp_down.burst;
                    }
                    result.reserve_decompositions.insert(source, decomp_up);
                    result.reserve_decompositions.insert(target, decomp_down);

                    up_state.confidence = decomp_up.effective;
                    down_state.confidence = decomp_down.effective;
                }

                // Phase 6: Clamp to effective_cap (NOT config.max_confidence)
                up_state.confidence = self
                    .config
                    .clamp_confidence_to(up_state.confidence, effective_cap);
                down_state.confidence = self
                    .config
                    .clamp_confidence_to(down_state.confidence, effective_cap);

                // Phase 3: Observe updated confidences in distribution resonance
                if let Some(dr) = &mut self.distribution_resonance {
                    dr.observe(source, up_state.confidence);
                    dr.observe(target, down_state.confidence);
                }

                // Compute confidence delta for dynamic weighting update
                let pre_source = source_state.confidence;
                let pre_target = target_state.confidence;
                let post_avg = (up_state.confidence + down_state.confidence) / 2.0;
                let pre_avg = (pre_source + pre_target) / 2.0;
                let bridge_usefulness = if pre_avg > 0.0 {
                    (post_avg - pre_avg) / pre_avg
                } else {
                    0.0
                };

                result.layer_states.insert(source, up_state);
                result.layer_states.insert(target, down_state);

                // Track amplification with damping and capping
                // Phase 3: Use distribution resonance to modulate amplification factor
                let dist_res_factor = self
                    .distribution_resonance
                    .as_ref()
                    .map(|dr| dr.resonance(source, target))
                    .unwrap_or(1.0);

                // Phase 6: Scale distribution resonance by braid sensitivity
                let scaled_dist_res = braid_signals
                    .as_ref()
                    .map(|s| dist_res_factor * s.resonance_sensitivity)
                    .unwrap_or(dist_res_factor);

                // Phase 5: Apply dynamic bridge weight as a multiplier
                let bridge_weight = self
                    .dynamic_weighting
                    .as_ref()
                    .map(|dw| dw.weight_for(source, target))
                    .unwrap_or(1.0);

                // Phase 6: Use braid damping instead of config damping
                let damped_factor = 1.0
                    + (amp_result.amplification_factor - 1.0)
                        * damping
                        * scaled_dist_res
                        * bridge_weight;
                result.total_amplification = (result.total_amplification * damped_factor)
                    .min(self.config.max_total_amplification);

                // Phase 5: Update dynamic bridge weighting with observed usefulness
                if let Some(ref mut dw) = self.dynamic_weighting {
                    dw.update(source, target, bridge_usefulness);
                    self.stats.dynamic_weight_updates += 1;
                }
            }
        }

        // Recalculate combined confidence
        result.layer_confidences = result
            .layer_states
            .iter()
            .map(|(l, s)| (*l, s.confidence))
            .collect();
        result.combined_confidence = self.calculate_combined_confidence(&result.layer_confidences);
        // Phase 6: Re-clamp to effective_cap
        result.combined_confidence = self
            .config
            .clamp_confidence_to(result.combined_confidence, effective_cap);
    }

    /// Calculate combined confidence using multiplicative formula.
    fn calculate_combined_confidence(&self, confidences: &HashMap<Layer, f32>) -> f32 {
        if confidences.is_empty() {
            return 0.0;
        }

        // Multiplicative combination (geometric mean with damped amplification)
        let product: f32 = confidences.values().product();
        let n = confidences.len() as f32;
        let damped_amp =
            1.0 + (self.config.global_amplification - 1.0) * self.config.amplification_damping;
        let raw_confidence = product.powf(1.0 / n) * damped_amp;

        // Clamp to prevent divergence
        self.config.clamp_confidence(raw_confidence)
    }

    /// Compute variance across layer confidence values (Phase 5 helper).
    fn compute_layer_variance(confidences: &HashMap<Layer, f32>) -> f32 {
        if confidences.len() < 2 {
            return 0.0;
        }
        let n = confidences.len() as f32;
        let mean: f32 = confidences.values().sum::<f32>() / n;
        let variance: f32 = confidences
            .values()
            .map(|&c| (c - mean).powi(2))
            .sum::<f32>()
            / n;
        variance
    }

    /// Compute braid signals for the current iteration (Phase 6).
    ///
    /// Returns `Some(BraidSignals)` if the braid is enabled, `None` otherwise.
    fn compute_braid_signals(
        &mut self,
        result: &StackProcessResult,
        iteration: u32,
    ) -> Option<BraidSignals> {
        let braid = self.octo_braid.as_mut()?;

        let input_confidence = result
            .layer_confidences
            .get(&Layer::BasePhysics)
            .copied()
            .unwrap_or(0.5);

        let current_effective_cap = result.effective_cap;

        let signals = braid.modulate(
            input_confidence,
            &result.layer_confidences,
            current_effective_cap,
            iteration,
        );

        // Update stats
        self.stats.braid_modulations += 1;
        if signals.system2_active {
            self.stats.braid_system2_activations += 1;
        }

        Some(signals)
    }

    /// Get the effective cap to use: braid's cap if available, otherwise config default.
    #[inline]
    fn effective_cap_from_braid(
        &self,
        braid_signals: &Option<BraidSignals>,
        result: &StackProcessResult,
    ) -> f32 {
        braid_signals
            .as_ref()
            .map(|s| s.effective_cap)
            .unwrap_or(result.effective_cap.max(self.config.max_confidence))
    }

    /// Get the damping factor: braid's damping if available, otherwise config default.
    #[inline]
    fn damping_from_braid(&self, braid_signals: &Option<BraidSignals>) -> f32 {
        braid_signals
            .as_ref()
            .map(|s| s.global_damping)
            .unwrap_or(self.config.amplification_damping)
    }

    /// Update statistics after processing.
    fn update_stats(&mut self, result: &StackProcessResult) {
        // Update running average
        let current_count = self.stats.convergence_count + self.stats.non_convergence_count;
        if current_count > 0 {
            self.stats.average_confidence = (self.stats.average_confidence
                * (current_count - 1) as f32
                + result.combined_confidence)
                / current_count as f32;
        } else {
            self.stats.average_confidence = result.combined_confidence;
        }

        // Update max confidence
        if result.combined_confidence > self.stats.max_confidence_achieved {
            self.stats.max_confidence_achieved = result.combined_confidence;
        }
    }

    /// Inject a state directly into a layer (for testing or external input).
    pub fn inject_state(&mut self, state: LayerState) {
        self.current_states.insert(state.layer, state);
    }

    /// Get all current layer states.
    pub fn all_states(&self) -> &HashMap<Layer, LayerState> {
        &self.current_states
    }
}

impl Default for LayerStack {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LayerStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerStack")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .field("current_states", &self.current_states.len())
            .field("bridge_count", &self.bridge_network.bridges().len())
            .field(
                "visualization_enabled",
                &self.visualization_engine.is_some(),
            )
            .field("reserve_enabled", &self.fractional_reserve.is_some())
            .field(
                "distribution_resonance_enabled",
                &self.distribution_resonance.is_some(),
            )
            .field("adaptive_cap_enabled", &self.adaptive_cap.is_some())
            .field(
                "dynamic_weighting_enabled",
                &self.dynamic_weighting.is_some(),
            )
            .field("online_learning_enabled", &self.online_learning.is_some())
            .field("octo_braid_enabled", &self.octo_braid.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_creation() {
        let stack = LayerStack::new();
        assert_eq!(stack.config().max_stack_iterations, 5);
        assert!(stack.bridge_network().bridges().is_empty());
        assert!(stack.visualization_engine().is_none());
        assert!(stack.fractional_reserve().is_none());
        assert!(stack.distribution_resonance().is_none());
    }

    #[test]
    fn test_stack_config() {
        let config = LayerStackConfig::new()
            .with_global_amplification(1.5)
            .with_max_iterations(10);

        let stack = LayerStack::with_config(config);
        assert_eq!(stack.config().global_amplification, 1.5);
        assert_eq!(stack.config().max_stack_iterations, 10);
    }

    #[test]
    fn test_phase3_config() {
        let config = LayerStackConfig::new().with_phase3();
        assert!(config.enable_visualization);
        assert!(config.enable_reserve);
        assert!(config.enable_distribution_resonance);

        let stack = LayerStack::with_config(config);
        assert!(stack.visualization_engine().is_some());
        assert!(stack.fractional_reserve().is_some());
        assert!(stack.distribution_resonance().is_some());
    }

    #[test]
    fn test_state_injection() {
        let mut stack = LayerStack::new();
        let state = LayerState::new(Layer::GaiaConsciousness, "test".to_string());

        stack.inject_state(state);

        assert!(stack.get_current_state(Layer::GaiaConsciousness).is_some());
    }

    #[test]
    fn test_forward_processing_no_bridges() {
        let mut stack = LayerStack::new();
        let input = LayerState::with_confidence(Layer::BasePhysics, "input".to_string(), 0.8);

        let result = stack.process_forward(input);

        // Without bridges, only the input layer should have a state
        assert_eq!(result.layer_states.len(), 1);
        assert!(result.layer_states.contains_key(&Layer::BasePhysics));
        assert!(result.visualization.is_none());
    }

    #[test]
    fn test_forward_with_phase3() {
        let config = LayerStackConfig::new().with_phase3();
        let mut stack = LayerStack::with_config(config);

        let input = LayerState::with_confidence(Layer::BasePhysics, "input".to_string(), 0.8);
        let result = stack.process_forward(input);

        // Visualization should have run
        assert!(result.visualization.is_some());
        // Reserve decompositions should exist for processed layers
        assert!(!result.reserve_decompositions.is_empty());
        assert!(result
            .reserve_decompositions
            .contains_key(&Layer::BasePhysics));
        // Stats should reflect visualization pass
        assert_eq!(stack.stats().visualization_passes, 1);
    }

    #[test]
    fn test_stack_result() {
        let result = StackProcessResult::empty();
        assert!(!result.is_successful(0.5));
        assert!(result.layer_states.is_empty());
        assert!(result.visualization.is_none());
        assert!(result.reserve_decompositions.is_empty());
        assert!(result.distribution_resonances.is_empty());
    }

    #[test]
    fn test_combined_confidence_calculation() {
        let stack = LayerStack::new();
        let mut confidences = HashMap::new();
        confidences.insert(Layer::BasePhysics, 0.8);
        confidences.insert(Layer::ExtendedPhysics, 0.8);

        let combined = stack.calculate_combined_confidence(&confidences);
        // Geometric mean of 0.8, 0.8 = 0.8
        // Damped amplification: 1.0 + (1.1 - 1.0) * 0.8 = 1.08
        // Combined: 0.8 * 1.08 = 0.864
        assert!((combined - 0.864).abs() < 0.01);
    }

    #[test]
    fn test_confidence_clamping() {
        let config = LayerStackConfig::new().with_max_confidence(1.5);
        let stack = LayerStack::with_config(config);

        // High confidence values should be clamped
        let mut confidences = HashMap::new();
        confidences.insert(Layer::BasePhysics, 1.8);
        confidences.insert(Layer::ExtendedPhysics, 1.8);

        let combined = stack.calculate_combined_confidence(&confidences);
        assert!(
            combined <= 1.5,
            "Combined confidence {} exceeds max 1.5",
            combined
        );
    }

    #[test]
    fn test_amplification_damping_config() {
        let config = LayerStackConfig::new()
            .with_amplification_damping(0.5)
            .with_max_total_amplification(5.0);

        assert_eq!(config.amplification_damping, 0.5);
        assert_eq!(config.max_total_amplification, 5.0);
    }

    // ---- Phase 5 unit tests ----

    #[test]
    fn test_phase5_config() {
        let config = LayerStackConfig::new().with_phase5();
        assert!(config.enable_adaptive_cap);
        assert!(config.enable_dynamic_weighting);
        assert!(config.enable_online_learning);
        assert!(config.adaptive_cap_config.enabled);
        assert!(config.dynamic_weight_config.enabled);
        assert!(config.online_learning_config.enabled);

        let stack = LayerStack::with_config(config);
        assert!(stack.adaptive_cap().is_some());
        assert!(stack.dynamic_weighting().is_some());
        assert!(stack.online_learning().is_some());
    }

    #[test]
    fn test_phase5_individual_config() {
        // Test enabling each subsystem individually
        let config_cap = LayerStackConfig::new().with_adaptive_cap();
        assert!(config_cap.enable_adaptive_cap);
        assert!(!config_cap.enable_dynamic_weighting);
        assert!(!config_cap.enable_online_learning);

        let config_dw = LayerStackConfig::new().with_dynamic_weighting();
        assert!(!config_dw.enable_adaptive_cap);
        assert!(config_dw.enable_dynamic_weighting);
        assert!(!config_dw.enable_online_learning);

        let config_ol = LayerStackConfig::new().with_online_learning();
        assert!(!config_ol.enable_adaptive_cap);
        assert!(!config_ol.enable_dynamic_weighting);
        assert!(config_ol.enable_online_learning);
    }

    #[test]
    fn test_all_subsystems_config() {
        let config = LayerStackConfig::new().with_all_subsystems();
        // Phase 3
        assert!(config.enable_visualization);
        assert!(config.enable_reserve);
        assert!(config.enable_distribution_resonance);
        // Phase 5
        assert!(config.enable_adaptive_cap);
        assert!(config.enable_dynamic_weighting);
        assert!(config.enable_online_learning);
    }

    #[test]
    fn test_phase5_disabled_by_default() {
        let stack = LayerStack::new();
        assert!(stack.adaptive_cap().is_none());
        assert!(stack.dynamic_weighting().is_none());
        assert!(stack.online_learning().is_none());
    }

    #[test]
    fn test_phase5_result_fields_empty() {
        let result = StackProcessResult::empty();
        assert_eq!(result.effective_cap, 0.0);
        assert!(result.bridge_weights.is_empty());
        assert!(result.learning_amplifications.is_empty());
    }

    #[test]
    fn test_phase5_forward_without_phase5_sets_default_cap() {
        let mut stack = LayerStack::new();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.8);
        let result = stack.process_forward(input);
        // Without Phase 5, effective_cap should equal config default (2.0)
        assert_eq!(result.effective_cap, 2.0);
        assert!(result.bridge_weights.is_empty());
        assert!(result.learning_amplifications.is_empty());
    }

    #[test]
    fn test_layer_variance_computation() {
        let mut confidences = HashMap::new();
        confidences.insert(Layer::BasePhysics, 1.0);
        confidences.insert(Layer::ExtendedPhysics, 1.0);
        let var = LayerStack::compute_layer_variance(&confidences);
        assert_eq!(var, 0.0, "Identical values should have zero variance");

        let mut confidences2 = HashMap::new();
        confidences2.insert(Layer::BasePhysics, 0.0);
        confidences2.insert(Layer::ExtendedPhysics, 1.0);
        let var2 = LayerStack::compute_layer_variance(&confidences2);
        assert!(
            (var2 - 0.25).abs() < 0.001,
            "Variance of [0,1] should be 0.25"
        );
    }

    #[test]
    fn test_clamp_confidence_to() {
        let config = LayerStackConfig::new().with_max_confidence(2.0);
        assert_eq!(config.clamp_confidence_to(1.5, 1.0), 1.0);
        assert_eq!(config.clamp_confidence_to(0.5, 1.0), 0.5);
        assert_eq!(config.clamp_confidence_to(-0.5, 1.0), 0.0);
    }
}
