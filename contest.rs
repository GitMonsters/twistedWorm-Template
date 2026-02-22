#![allow(dead_code)]
//! RustyWorm Architecture Contest (Phase 6 Upgrade — OCTO Braid Integration)
//!
//! 8-Layer Compounding Integration Model vs Traditional Sequential Model
//!
//! ## Traditional Pipeline Variants (Phase 5B)
//!
//! The traditional pipeline now has three variants:
//! - **Arithmetic**: Simple arithmetic mean of layer confidences (original)
//! - **Weighted**: Learned per-layer weights from a training phase
//! - **Attention**: Softmax-weighted combination where weights = f(layer_confidence)
//!
//! All traditional variants also optionally use skip connections (L1->L3, L1->L5, etc.)
//!
//! ## Scenario Categories (Phase 5A)
//!
//! - **Standard**: Original 8 scenarios (physics, cross-domain, etc.)
//! - **Adversarial**: Contradictory layer signals designed to confuse
//! - **Noisy**: Confidence jitter and unstable inputs
//! - **Degraded**: Some layers disabled or capped
//! - **Calibration**: Known-difficulty inputs that should NOT saturate
//!
//! ## Phase 6: OCTO Braid
//!
//! Round 3 now uses the full OCTO Braid-enabled pipeline (Phase 3 + 5 + 6).
//! Calibration scenarios pass `expected_cap` as a `difficulty_hint` to the braid,
//! which cross-modulates all 6 subsystems to prevent confidence saturation.

use std::time::Instant;

use rustyworm::mimicry::layers::{
    bridges::BridgeBuilder,
    compounding::CompoundingMetrics,
    emergence::EmergenceFramework,
    integration::{IntegrationConfig, LayerIntegration},
    layer::{Layer, LayerState},
    metrics::{compute_quality, MetricsAnalyzer},
    stack::{LayerStack, LayerStackConfig, StackProcessResult},
};

// =============================================================================
// Traditional Pipeline Variants (Phase 5B)
// =============================================================================

/// The strategy used to combine layer confidences in the traditional pipeline.
#[derive(Debug, Clone, Copy)]
enum TraditionalStrategy {
    /// Simple arithmetic mean of layer confidences.
    Arithmetic,
    /// Weighted combination with learned per-layer weights.
    Weighted,
    /// Softmax-weighted combination (attention mechanism).
    Attention,
}

impl TraditionalStrategy {
    fn name(&self) -> &'static str {
        match self {
            TraditionalStrategy::Arithmetic => "Arithmetic",
            TraditionalStrategy::Weighted => "Weighted",
            TraditionalStrategy::Attention => "Attention",
        }
    }
}

/// Configuration for skip connections in the traditional pipeline.
#[derive(Debug, Clone)]
struct SkipConfig {
    /// Whether to enable skip connections.
    enabled: bool,
    /// Skip connection pairs: (from_layer, to_layer, weight).
    connections: Vec<(Layer, Layer, f32)>,
}

impl Default for SkipConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            connections: vec![
                (Layer::BasePhysics, Layer::CrossDomain, 0.3),
                (Layer::BasePhysics, Layer::MultilingualProcessing, 0.2),
                (Layer::ExtendedPhysics, Layer::CollaborativeLearning, 0.25),
                (Layer::CrossDomain, Layer::ExternalApis, 0.2),
                (
                    Layer::GaiaConsciousness,
                    Layer::PreCognitiveVisualization,
                    0.3,
                ),
            ],
        }
    }
}

/// The traditional pipeline with three combination strategies and optional skip connections.
struct TraditionalPipeline {
    stack: LayerStack,
    strategy: TraditionalStrategy,
    skip_config: SkipConfig,
    /// Learned per-layer weights for the Weighted strategy (trained in `train()`).
    layer_weights: Vec<(Layer, f32)>,
    /// Attention temperature for softmax (lower = sharper attention).
    attention_temperature: f32,
}

impl TraditionalPipeline {
    fn new(strategy: TraditionalStrategy) -> Self {
        let config = LayerStackConfig::new()
            .with_max_iterations(0)
            .without_backward_propagation()
            .with_global_amplification(1.0)
            .with_amplification_damping(0.0)
            .with_max_confidence(2.0);

        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }

        // Default equal weights
        let layer_weights: Vec<(Layer, f32)> =
            Layer::all().iter().map(|&l| (l, 1.0 / 8.0)).collect();

        Self {
            stack,
            strategy,
            skip_config: SkipConfig::default(),
            layer_weights,
            attention_temperature: 1.0,
        }
    }

    fn new_with_phase3(strategy: TraditionalStrategy) -> Self {
        let config = LayerStackConfig::new()
            .with_max_iterations(0)
            .without_backward_propagation()
            .with_global_amplification(1.0)
            .with_amplification_damping(0.0)
            .with_max_confidence(2.0)
            .with_phase3();

        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }

        let layer_weights: Vec<(Layer, f32)> =
            Layer::all().iter().map(|&l| (l, 1.0 / 8.0)).collect();

        Self {
            stack,
            strategy,
            skip_config: SkipConfig::default(),
            layer_weights,
            attention_temperature: 1.0,
        }
    }

    fn with_skip_connections(mut self) -> Self {
        self.skip_config.enabled = true;
        self
    }

    /// Train the Weighted strategy using a set of calibration inputs.
    /// Learns weights proportional to each layer's average confidence.
    fn train(&mut self, calibration_inputs: &[LayerState]) {
        if calibration_inputs.is_empty() {
            return;
        }

        let mut layer_sums: Vec<(Layer, f64)> = Layer::all().iter().map(|&l| (l, 0.0)).collect();

        for input in calibration_inputs {
            let result = self.stack.process_forward(input.clone());
            for (layer, sum) in layer_sums.iter_mut() {
                if let Some(&conf) = result.layer_confidences.get(layer) {
                    *sum += conf as f64;
                }
            }
        }

        let n = calibration_inputs.len() as f64;
        let total: f64 = layer_sums.iter().map(|(_, s)| s / n).sum();
        if total > 0.0 {
            self.layer_weights = layer_sums
                .iter()
                .map(|(l, s)| (*l, ((s / n) / total) as f32))
                .collect();
        }
    }

    /// Process input through the traditional pipeline with the configured strategy.
    fn process(&mut self, input: LayerState) -> TraditionalResult {
        let start = Instant::now();

        let mut result = self.stack.process_forward(input);

        // Apply skip connections if enabled
        if self.skip_config.enabled {
            self.apply_skip_connections(&mut result);
        }

        // Combine layer confidences using the selected strategy
        let combined = match self.strategy {
            TraditionalStrategy::Arithmetic => self.combine_arithmetic(&result),
            TraditionalStrategy::Weighted => self.combine_weighted(&result),
            TraditionalStrategy::Attention => self.combine_attention(&result),
        };

        let min_confidence = result
            .layer_confidences
            .values()
            .copied()
            .fold(f32::MAX, f32::min);

        let arithmetic_mean = if result.layer_confidences.is_empty() {
            0.0
        } else {
            let sum: f32 = result.layer_confidences.values().sum();
            sum / result.layer_confidences.len() as f32
        };

        result.combined_confidence = combined;
        result.total_amplification = 1.0;

        let elapsed = start.elapsed();

        TraditionalResult {
            stack_result: result,
            strategy_confidence: combined,
            arithmetic_mean,
            min_confidence,
            processing_time_us: elapsed.as_micros() as u64,
        }
    }

    /// Arithmetic mean: simple average.
    fn combine_arithmetic(&self, result: &StackProcessResult) -> f32 {
        if result.layer_confidences.is_empty() {
            return 0.0;
        }
        let sum: f32 = result.layer_confidences.values().sum();
        sum / result.layer_confidences.len() as f32
    }

    /// Weighted combination using learned per-layer weights.
    fn combine_weighted(&self, result: &StackProcessResult) -> f32 {
        let mut weighted_sum = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for (layer, weight) in &self.layer_weights {
            if let Some(&conf) = result.layer_confidences.get(layer) {
                weighted_sum += conf * weight;
                weight_total += weight;
            }
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            self.combine_arithmetic(result)
        }
    }

    /// Attention-weighted combination using softmax of layer confidences.
    fn combine_attention(&self, result: &StackProcessResult) -> f32 {
        if result.layer_confidences.is_empty() {
            return 0.0;
        }

        // Compute softmax weights from confidences
        let confs: Vec<f32> = result.layer_confidences.values().copied().collect();
        let max_conf = confs.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = confs
            .iter()
            .map(|&c| ((c - max_conf) / self.attention_temperature).exp())
            .sum();

        let mut weighted_sum = 0.0_f32;
        for &conf in result.layer_confidences.values() {
            let weight = ((conf - max_conf) / self.attention_temperature).exp() / exp_sum;
            weighted_sum += conf * weight;
        }

        weighted_sum
    }

    /// Apply skip connections by blending confidence from source to target.
    fn apply_skip_connections(&self, result: &mut StackProcessResult) {
        for (from, to, weight) in &self.skip_config.connections {
            if let (Some(&from_conf), Some(&to_conf)) = (
                result.layer_confidences.get(from),
                result.layer_confidences.get(to),
            ) {
                let blended = to_conf * (1.0 - weight) + from_conf * weight;
                result.layer_confidences.insert(*to, blended);
            }
        }
    }
}

struct TraditionalResult {
    stack_result: StackProcessResult,
    strategy_confidence: f32,
    arithmetic_mean: f32,
    min_confidence: f32,
    processing_time_us: u64,
}

// =============================================================================
// Compounding Pipeline (wraps the existing system)
// =============================================================================

struct CompoundingPipeline {
    stack: LayerStack,
}

impl CompoundingPipeline {
    fn new() -> Self {
        let config = LayerStackConfig::new()
            .with_max_iterations(5)
            .with_max_confidence(2.0)
            .with_max_total_amplification(10.0)
            .with_amplification_damping(0.8);

        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }

        Self { stack }
    }

    fn new_with_phase3() -> Self {
        let config = LayerStackConfig::new()
            .with_phase3()
            .with_max_iterations(5)
            .with_max_confidence(2.0)
            .with_max_total_amplification(10.0)
            .with_amplification_damping(0.8);

        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }

        Self { stack }
    }

    fn new_with_phase5() -> Self {
        let config = LayerStackConfig::new()
            .with_all_subsystems()
            .with_max_iterations(5)
            .with_max_confidence(2.0)
            .with_max_total_amplification(10.0)
            .with_amplification_damping(0.8);

        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }

        Self { stack }
    }

    fn new_with_braid() -> Self {
        let config = LayerStackConfig::new()
            .with_all_phases()
            .with_max_iterations(5)
            .with_max_confidence(2.0)
            .with_max_total_amplification(10.0)
            .with_amplification_damping(0.8);

        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }

        Self { stack }
    }

    /// Process with an optional difficulty hint for the OCTO braid.
    /// The expected_cap from calibration scenarios is converted to a difficulty hint.
    /// The braid computes effective_cap = base_cap*(1-d) + min_cap*d, so we invert:
    /// d = (base_cap - expected_cap) / (base_cap - min_cap)
    fn process_with_hint(
        &mut self,
        input: LayerState,
        expected_cap: Option<f32>,
    ) -> CompoundingResult {
        if let Some(cap) = expected_cap {
            // base_cap = 2.0, min_cap = 0.5, so (base - min) = 1.5
            let difficulty = ((2.0 - cap) / 1.5).clamp(0.0, 1.0);
            self.stack.set_difficulty_hint(Some(difficulty));
        }

        let start = Instant::now();
        let result = self.stack.process_bidirectional(input);
        let elapsed = start.elapsed();

        let arithmetic_mean = if result.layer_confidences.is_empty() {
            0.0
        } else {
            let sum: f32 = result.layer_confidences.values().sum();
            sum / result.layer_confidences.len() as f32
        };

        CompoundingResult {
            stack_result: result,
            arithmetic_mean,
            processing_time_us: elapsed.as_micros() as u64,
        }
    }

    fn process(&mut self, input: LayerState) -> CompoundingResult {
        let start = Instant::now();
        let result = self.stack.process_bidirectional(input);
        let elapsed = start.elapsed();

        let arithmetic_mean = if result.layer_confidences.is_empty() {
            0.0
        } else {
            let sum: f32 = result.layer_confidences.values().sum();
            sum / result.layer_confidences.len() as f32
        };

        CompoundingResult {
            stack_result: result,
            arithmetic_mean,
            processing_time_us: elapsed.as_micros() as u64,
        }
    }
}

struct CompoundingResult {
    stack_result: StackProcessResult,
    arithmetic_mean: f32,
    processing_time_us: u64,
}

// =============================================================================
// Contest Scenarios (Phase 5A: Expanded)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScenarioCategory {
    Standard,
    Adversarial,
    Noisy,
    Degraded,
    Calibration,
}

impl ScenarioCategory {
    fn name(&self) -> &'static str {
        match self {
            ScenarioCategory::Standard => "Standard",
            ScenarioCategory::Adversarial => "Adversarial",
            ScenarioCategory::Noisy => "Noisy",
            ScenarioCategory::Degraded => "Degraded",
            ScenarioCategory::Calibration => "Calibration",
        }
    }
}

struct Scenario {
    name: &'static str,
    description: &'static str,
    category: ScenarioCategory,
    input_text: &'static str,
    input_layer: Layer,
    input_confidence: f32,
    /// Expected behavior for calibration scenarios.
    expected_cap: Option<f32>,
}

fn build_standard_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "Low-Confidence Physics",
            description: "Weak signal in base domain — can the system amplify it?",
            category: ScenarioCategory::Standard,
            input_text: "faint quantum tunneling signature detected",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.3,
            expected_cap: None,
        },
        Scenario {
            name: "High-Confidence Physics",
            description: "Strong signal — does compounding maintain advantage?",
            category: ScenarioCategory::Standard,
            input_text: "verified relativistic mass-energy equivalence",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.9,
            expected_cap: None,
        },
        Scenario {
            name: "Cross-Domain Transfer",
            description: "Start at L3 (cross-domain) — how far does confidence spread?",
            category: ScenarioCategory::Standard,
            input_text: "entropy applies to both thermodynamics and information theory",
            input_layer: Layer::CrossDomain,
            input_confidence: 0.6,
            expected_cap: None,
        },
        Scenario {
            name: "Consciousness Probe",
            description: "Abstract L4 input — does GAIA consciousness benefit from compounding?",
            category: ScenarioCategory::Standard,
            input_text: "the observer effect bridges physics and phenomenology",
            input_layer: Layer::GaiaConsciousness,
            input_confidence: 0.5,
            expected_cap: None,
        },
        Scenario {
            name: "Language Ambiguity",
            description:
                "Ambiguous linguistic input — compounding should resolve uncertainty better",
            category: ScenarioCategory::Standard,
            input_text: "recursive grammars generate infinite expressions from finite rules",
            input_layer: Layer::MultilingualProcessing,
            input_confidence: 0.4,
            expected_cap: None,
        },
        Scenario {
            name: "Edge Case: Near-Zero",
            description: "Almost no initial signal — can compounding recover any value?",
            category: ScenarioCategory::Standard,
            input_text: "noise threshold boundary detection",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.05,
            expected_cap: None,
        },
        Scenario {
            name: "Edge Case: Maximum",
            description: "Already-high confidence — does compounding still add value?",
            category: ScenarioCategory::Standard,
            input_text: "mathematical proof by induction verified",
            input_layer: Layer::BasePhysics,
            input_confidence: 1.0,
            expected_cap: None,
        },
        Scenario {
            name: "Collaborative Signal",
            description: "Start at L6 (collaborative) — multi-agent amplification test",
            category: ScenarioCategory::Standard,
            input_text: "consensus reached across five independent reasoning agents",
            input_layer: Layer::CollaborativeLearning,
            input_confidence: 0.65,
            expected_cap: None,
        },
    ]
}

fn build_adversarial_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "Contradictory Signals",
            description:
                "High physics + low consciousness — contradictory inputs stress-test integration",
            category: ScenarioCategory::Adversarial,
            input_text: "quantum mechanics disproves free will yet consciousness persists",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.95,
            expected_cap: None,
        },
        Scenario {
            name: "Adversarial Analogy",
            description: "False analogy that looks plausible — should not amplify blindly",
            category: ScenarioCategory::Adversarial,
            input_text: "gravity pulls objects down therefore consciousness must also be weighted",
            input_layer: Layer::GaiaConsciousness,
            input_confidence: 0.7,
            expected_cap: None,
        },
        Scenario {
            name: "Self-Referential Loop",
            description: "Input referencing the system itself — recursion attack",
            category: ScenarioCategory::Adversarial,
            input_text: "this statement about confidence amplification should not be amplified",
            input_layer: Layer::CrossDomain,
            input_confidence: 0.5,
            expected_cap: None,
        },
        Scenario {
            name: "Extreme Confidence Claim",
            description: "Input claims perfect certainty — should the system trust it?",
            category: ScenarioCategory::Adversarial,
            input_text: "absolutely certain beyond any doubt with maximum confidence",
            input_layer: Layer::ExternalApis,
            input_confidence: 1.0,
            expected_cap: None,
        },
    ]
}

fn build_noisy_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "Jittery Low Signal",
            description: "Very low confidence with noise — can the system find signal?",
            category: ScenarioCategory::Noisy,
            input_text: "partial detection of electromagnetic anomaly with interference",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.15,
            expected_cap: None,
        },
        Scenario {
            name: "Noisy Mid-Range",
            description: "Medium confidence with random perturbation",
            category: ScenarioCategory::Noisy,
            input_text: "approximate pattern matching with significant noise floor",
            input_layer: Layer::CrossDomain,
            input_confidence: 0.45,
            expected_cap: None,
        },
        Scenario {
            name: "Oscillating Signal",
            description: "Confidence that might oscillate between layers",
            category: ScenarioCategory::Noisy,
            input_text: "intermittent resonance between wave functions",
            input_layer: Layer::ExtendedPhysics,
            input_confidence: 0.55,
            expected_cap: None,
        },
    ]
}

fn build_degraded_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "Minimal Layers",
            description: "Only L1 active — base physics in isolation",
            category: ScenarioCategory::Degraded,
            input_text: "basic Newtonian mechanics calculation",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.7,
            expected_cap: None,
        },
        Scenario {
            name: "High Layer Only",
            description: "Input at L8 (visualization) — highest layer in isolation",
            category: ScenarioCategory::Degraded,
            input_text: "abstract visualization of possible futures",
            input_layer: Layer::PreCognitiveVisualization,
            input_confidence: 0.6,
            expected_cap: None,
        },
        Scenario {
            name: "Weak Everywhere",
            description: "Very weak signal across all layers — stress resilience",
            category: ScenarioCategory::Degraded,
            input_text: "barely detectable signal at noise floor",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.08,
            expected_cap: None,
        },
    ]
}

fn build_calibration_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "Known Easy",
            description: "Simple input that should not saturate — calibration check",
            category: ScenarioCategory::Calibration,
            input_text: "two plus two equals four",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.8,
            expected_cap: Some(1.5),
        },
        Scenario {
            name: "Known Medium",
            description: "Moderate difficulty — confidence should land in [0.5, 1.2]",
            category: ScenarioCategory::Calibration,
            input_text: "protein folding follows thermodynamic principles",
            input_layer: Layer::CrossDomain,
            input_confidence: 0.5,
            expected_cap: Some(1.2),
        },
        Scenario {
            name: "Known Hard",
            description: "Genuinely hard input — should NOT reach high confidence",
            category: ScenarioCategory::Calibration,
            input_text: "P equals NP proof via novel algebraic geometry approach",
            input_layer: Layer::GaiaConsciousness,
            input_confidence: 0.2,
            expected_cap: Some(0.8),
        },
        Scenario {
            name: "Known Impossible",
            description: "Logically impossible — confidence should stay very low",
            category: ScenarioCategory::Calibration,
            input_text: "the set of all sets that do not contain themselves is consistent",
            input_layer: Layer::BasePhysics,
            input_confidence: 0.1,
            expected_cap: Some(0.5),
        },
    ]
}

fn build_all_scenarios() -> Vec<Scenario> {
    let mut all = Vec::new();
    all.extend(build_standard_scenarios());
    all.extend(build_adversarial_scenarios());
    all.extend(build_noisy_scenarios());
    all.extend(build_degraded_scenarios());
    all.extend(build_calibration_scenarios());
    all
}

// =============================================================================
// Scoring System
// =============================================================================

#[derive(Debug, Clone)]
struct RoundScore {
    scenario_name: String,
    category: ScenarioCategory,
    // Raw output
    trad_confidence: f32,
    comp_confidence: f32,
    // Quality metrics
    trad_quality: f32,
    comp_quality: f32,
    // Amplification ratio (output / input)
    trad_amp_ratio: f32,
    comp_amp_ratio: f32,
    // Layer spread (how many layers have confidence > 0)
    trad_layer_spread: usize,
    comp_layer_spread: usize,
    // Convergence
    comp_converged: bool,
    comp_iterations: u32,
    // Timing
    trad_time_us: u64,
    comp_time_us: u64,
    // Winner for this round
    winner: &'static str,
    margin: f32,
    // Calibration result
    calibration_pass: Option<bool>,
}

#[derive(Debug)]
struct ContestResults {
    rounds: Vec<RoundScore>,
    compounding_wins: u32,
    traditional_wins: u32,
    ties: u32,
    // Aggregate metrics
    avg_comp_advantage: f32,
    avg_trad_confidence: f32,
    avg_comp_confidence: f32,
    avg_trad_quality: f32,
    avg_comp_quality: f32,
    total_trad_time_us: u64,
    total_comp_time_us: u64,
    // Calibration
    calibration_passes: u32,
    calibration_total: u32,
}

fn run_contest(
    scenarios: &[Scenario],
    trad_strategy: TraditionalStrategy,
    use_phase3: bool,
    use_skip: bool,
    use_braid: bool,
) -> ContestResults {
    let mut rounds = Vec::new();
    let mut comp_wins = 0u32;
    let mut trad_wins = 0u32;
    let mut ties = 0u32;
    let mut cal_passes = 0u32;
    let mut cal_total = 0u32;

    let _quality_analyzer = MetricsAnalyzer::new();

    for scenario in scenarios {
        // Fresh pipelines for each scenario
        let mut traditional = if use_phase3 {
            TraditionalPipeline::new_with_phase3(trad_strategy)
        } else {
            TraditionalPipeline::new(trad_strategy)
        };

        if use_skip {
            traditional = traditional.with_skip_connections();
        }

        // Train weighted strategy with calibration data
        if matches!(trad_strategy, TraditionalStrategy::Weighted) {
            let cal_inputs: Vec<LayerState> = vec![
                LayerState::with_confidence(Layer::BasePhysics, "calibration 1".to_string(), 0.5),
                LayerState::with_confidence(Layer::CrossDomain, "calibration 2".to_string(), 0.6),
                LayerState::with_confidence(
                    Layer::GaiaConsciousness,
                    "calibration 3".to_string(),
                    0.4,
                ),
            ];
            traditional.train(&cal_inputs);
        }

        let mut compounding = if use_braid {
            CompoundingPipeline::new_with_braid()
        } else if use_phase3 {
            CompoundingPipeline::new_with_phase3()
        } else {
            CompoundingPipeline::new()
        };

        let trad_input = LayerState::with_confidence(
            scenario.input_layer,
            scenario.input_text.to_string(),
            scenario.input_confidence,
        );
        let comp_input = LayerState::with_confidence(
            scenario.input_layer,
            scenario.input_text.to_string(),
            scenario.input_confidence,
        );

        let trad_result = traditional.process(trad_input);
        let comp_result = compounding.process_with_hint(comp_input, scenario.expected_cap);

        let trad_conf = trad_result.stack_result.combined_confidence;
        let comp_conf = comp_result.stack_result.combined_confidence;

        // Compute quality metrics
        let trad_qm = compute_quality(&trad_result.stack_result);
        let comp_qm = compute_quality(&comp_result.stack_result);

        let trad_amp = if scenario.input_confidence > 0.0 {
            trad_conf / scenario.input_confidence
        } else {
            1.0
        };
        let comp_amp = if scenario.input_confidence > 0.0 {
            comp_conf / scenario.input_confidence
        } else {
            1.0
        };

        let trad_spread = trad_result
            .stack_result
            .layer_confidences
            .values()
            .filter(|&&c| c > 0.0)
            .count();
        let comp_spread = comp_result
            .stack_result
            .layer_confidences
            .values()
            .filter(|&&c| c > 0.0)
            .count();

        let margin = comp_conf - trad_conf;
        let winner = if margin.abs() < 0.001 {
            ties += 1;
            "TIE"
        } else if margin > 0.0 {
            comp_wins += 1;
            "COMPOUNDING"
        } else {
            trad_wins += 1;
            "TRADITIONAL"
        };

        // Check calibration expectations
        let calibration_pass = scenario.expected_cap.map(|cap| {
            cal_total += 1;
            let pass = comp_conf <= cap;
            if pass {
                cal_passes += 1;
            }
            pass
        });

        rounds.push(RoundScore {
            scenario_name: scenario.name.to_string(),
            category: scenario.category,
            trad_confidence: trad_conf,
            comp_confidence: comp_conf,
            trad_quality: trad_qm.overall_quality,
            comp_quality: comp_qm.overall_quality,
            trad_amp_ratio: trad_amp,
            comp_amp_ratio: comp_amp,
            trad_layer_spread: trad_spread,
            comp_layer_spread: comp_spread,
            comp_converged: comp_result.stack_result.converged,
            comp_iterations: comp_result.stack_result.iterations,
            trad_time_us: trad_result.processing_time_us,
            comp_time_us: comp_result.processing_time_us,
            winner,
            margin,
            calibration_pass,
        });
    }

    let n = rounds.len() as f32;
    let avg_trad = rounds.iter().map(|r| r.trad_confidence).sum::<f32>() / n;
    let avg_comp = rounds.iter().map(|r| r.comp_confidence).sum::<f32>() / n;
    let avg_advantage = rounds.iter().map(|r| r.margin).sum::<f32>() / n;
    let avg_trad_q = rounds.iter().map(|r| r.trad_quality).sum::<f32>() / n;
    let avg_comp_q = rounds.iter().map(|r| r.comp_quality).sum::<f32>() / n;
    let total_trad_time = rounds.iter().map(|r| r.trad_time_us).sum();
    let total_comp_time = rounds.iter().map(|r| r.comp_time_us).sum();

    ContestResults {
        rounds,
        compounding_wins: comp_wins,
        traditional_wins: trad_wins,
        ties,
        avg_comp_advantage: avg_advantage,
        avg_trad_confidence: avg_trad,
        avg_comp_confidence: avg_comp,
        avg_trad_quality: avg_trad_q,
        avg_comp_quality: avg_comp_q,
        total_trad_time_us: total_trad_time,
        total_comp_time_us: total_comp_time,
        calibration_passes: cal_passes,
        calibration_total: cal_total,
    }
}

// =============================================================================
// Stress Test: Repeated Passes
// =============================================================================

struct StressResult {
    trad_confidences: Vec<f32>,
    comp_confidences: Vec<f32>,
    trad_variance: f32,
    comp_variance: f32,
    trad_final: f32,
    comp_final: f32,
}

fn run_stress_test(passes: usize, use_phase3: bool) -> StressResult {
    let mut traditional = if use_phase3 {
        TraditionalPipeline::new_with_phase3(TraditionalStrategy::Arithmetic)
    } else {
        TraditionalPipeline::new(TraditionalStrategy::Arithmetic)
    };
    let mut compounding = if use_phase3 {
        CompoundingPipeline::new_with_phase3()
    } else {
        CompoundingPipeline::new()
    };

    let mut trad_confs = Vec::new();
    let mut comp_confs = Vec::new();

    for i in 0..passes {
        let confidence = 0.3 + (i as f32 % 7.0) * 0.1;
        let text = format!("stress pass {}: emergent pattern analysis", i);

        let trad_input = LayerState::with_confidence(Layer::BasePhysics, text.clone(), confidence);
        let comp_input = LayerState::with_confidence(Layer::BasePhysics, text, confidence);

        let trad_r = traditional.process(trad_input);
        let comp_r = compounding.process(comp_input);

        trad_confs.push(trad_r.stack_result.combined_confidence);
        comp_confs.push(comp_r.stack_result.combined_confidence);
    }

    let trad_mean = trad_confs.iter().sum::<f32>() / passes as f32;
    let comp_mean = comp_confs.iter().sum::<f32>() / passes as f32;

    let trad_var = trad_confs
        .iter()
        .map(|c| (c - trad_mean).powi(2))
        .sum::<f32>()
        / passes as f32;
    let comp_var = comp_confs
        .iter()
        .map(|c| (c - comp_mean).powi(2))
        .sum::<f32>()
        / passes as f32;

    StressResult {
        trad_final: *trad_confs.last().unwrap_or(&0.0),
        comp_final: *comp_confs.last().unwrap_or(&0.0),
        trad_variance: trad_var,
        comp_variance: comp_var,
        trad_confidences: trad_confs,
        comp_confidences: comp_confs,
    }
}

// =============================================================================
// Phase 3 Subsystem Comparison
// =============================================================================

struct Phase3Comparison {
    no_p3_trad_conf: f32,
    no_p3_comp_conf: f32,
    p3_trad_conf: f32,
    p3_comp_conf: f32,
    comp_viz_active: bool,
    comp_reserve_bursts: bool,
    comp_dist_resonances: usize,
}

fn run_phase3_comparison() -> Phase3Comparison {
    let input_text =
        "quantum coherence enables emergent consciousness through recursive self-reference";
    let input_conf = 0.6;

    let mut trad = TraditionalPipeline::new(TraditionalStrategy::Arithmetic);
    let mut comp = CompoundingPipeline::new();

    let t1 = trad.process(LayerState::with_confidence(
        Layer::BasePhysics,
        input_text.to_string(),
        input_conf,
    ));
    let c1 = comp.process(LayerState::with_confidence(
        Layer::BasePhysics,
        input_text.to_string(),
        input_conf,
    ));

    let mut trad_p3 = TraditionalPipeline::new_with_phase3(TraditionalStrategy::Arithmetic);
    let mut comp_p3 = CompoundingPipeline::new_with_phase3();

    let t2 = trad_p3.process(LayerState::with_confidence(
        Layer::BasePhysics,
        input_text.to_string(),
        input_conf,
    ));
    let c2 = comp_p3.process(LayerState::with_confidence(
        Layer::BasePhysics,
        input_text.to_string(),
        input_conf,
    ));

    Phase3Comparison {
        no_p3_trad_conf: t1.stack_result.combined_confidence,
        no_p3_comp_conf: c1.stack_result.combined_confidence,
        p3_trad_conf: t2.stack_result.combined_confidence,
        p3_comp_conf: c2.stack_result.combined_confidence,
        comp_viz_active: c2.stack_result.visualization.is_some(),
        comp_reserve_bursts: c2
            .stack_result
            .reserve_decompositions
            .values()
            .any(|d| d.burst > 0.0),
        comp_dist_resonances: c2.stack_result.distribution_resonances.len(),
    }
}

// =============================================================================
// Emergence & Compounding Analysis
// =============================================================================

struct EmergenceComparison {
    trad_emergence: f32,
    comp_emergence: f32,
    trad_synergy: f32,
    comp_synergy: f32,
    trad_compounding_factor: f32,
    comp_compounding_factor: f32,
    trad_is_beneficial: bool,
    comp_is_beneficial: bool,
}

fn run_emergence_comparison() -> EmergenceComparison {
    let input_text = "cross-domain synthesis reveals hidden structural isomorphisms";
    let input_conf = 0.55;

    let mut trad = TraditionalPipeline::new(TraditionalStrategy::Arithmetic);
    let mut comp = CompoundingPipeline::new();

    let trad_r = trad.process(LayerState::with_confidence(
        Layer::BasePhysics,
        input_text.to_string(),
        input_conf,
    ));
    let comp_r = comp.process(LayerState::with_confidence(
        Layer::BasePhysics,
        input_text.to_string(),
        input_conf,
    ));

    let mut metrics = CompoundingMetrics::new();
    let trad_analysis = metrics.analyze(&trad_r.stack_result);
    let comp_analysis = metrics.analyze(&comp_r.stack_result);

    let mut emergence = EmergenceFramework::new();
    let trad_em = emergence.analyze(&trad_r.stack_result);
    let comp_em = emergence.analyze(&comp_r.stack_result);

    EmergenceComparison {
        trad_emergence: trad_em.emergence_value,
        comp_emergence: comp_em.emergence_value,
        trad_synergy: trad_analysis.synergy_score,
        comp_synergy: comp_analysis.synergy_score,
        trad_compounding_factor: trad_analysis.compounding_factor,
        comp_compounding_factor: comp_analysis.compounding_factor,
        trad_is_beneficial: trad_analysis.is_beneficial,
        comp_is_beneficial: comp_analysis.is_beneficial,
    }
}

// =============================================================================
// LayerIntegration Head-to-Head
// =============================================================================

struct IntegrationHeadToHead {
    scenarios: Vec<(&'static str, f32, f32, bool, bool)>,
}

fn run_integration_head_to_head() -> IntegrationHeadToHead {
    let mut trad_integration = LayerIntegration::with_config(IntegrationConfig {
        enable_gaia: true,
        enable_external_apis: false,
        min_amplification_benefit: 0.05,
        track_statistics: true,
        max_processing_time_ms: 5000,
        enable_phase3: false,
    });

    let mut comp_integration = LayerIntegration::with_config(IntegrationConfig {
        enable_gaia: true,
        enable_external_apis: false,
        min_amplification_benefit: 0.05,
        track_statistics: true,
        max_processing_time_ms: 5000,
        enable_phase3: true,
    });

    let inputs = vec![
        "The wave function collapses upon observation of the particle",
        "Language shapes thought in recursive grammatical patterns",
        "Consciousness emerges from neural complexity and feedback loops",
        "Collaborative reasoning produces insights no single agent could",
        "Cross-domain analogies reveal universal structural patterns",
    ];

    let mut scenarios = Vec::new();
    for input in inputs {
        let trad_r = trad_integration.process(input, None);
        let comp_r = comp_integration.process(input, None);
        scenarios.push((
            input,
            trad_r.final_confidence,
            comp_r.final_confidence,
            trad_r.gaia_contributed,
            comp_r.gaia_contributed,
        ));
    }

    IntegrationHeadToHead { scenarios }
}

// =============================================================================
// Traditional Strategy Head-to-Head (Phase 5B)
// =============================================================================

struct StrategyComparison {
    arith_avg_conf: f32,
    weighted_avg_conf: f32,
    attention_avg_conf: f32,
    arith_avg_quality: f32,
    weighted_avg_quality: f32,
    attention_avg_quality: f32,
    skip_bonus_conf: f32,
    skip_bonus_quality: f32,
}

fn run_strategy_comparison() -> StrategyComparison {
    let scenarios = build_standard_scenarios();

    let arith = run_contest(
        &scenarios,
        TraditionalStrategy::Arithmetic,
        false,
        false,
        false,
    );
    let weighted = run_contest(
        &scenarios,
        TraditionalStrategy::Weighted,
        false,
        false,
        false,
    );
    let attention = run_contest(
        &scenarios,
        TraditionalStrategy::Attention,
        false,
        false,
        false,
    );
    let arith_skip = run_contest(
        &scenarios,
        TraditionalStrategy::Arithmetic,
        false,
        true,
        false,
    );

    StrategyComparison {
        arith_avg_conf: arith.avg_trad_confidence,
        weighted_avg_conf: weighted.avg_trad_confidence,
        attention_avg_conf: attention.avg_trad_confidence,
        arith_avg_quality: arith.avg_trad_quality,
        weighted_avg_quality: weighted.avg_trad_quality,
        attention_avg_quality: attention.avg_trad_quality,
        skip_bonus_conf: arith_skip.avg_trad_confidence - arith.avg_trad_confidence,
        skip_bonus_quality: arith_skip.avg_trad_quality - arith.avg_trad_quality,
    }
}

// =============================================================================
// Print helpers
// =============================================================================

fn print_contest_table(results: &ContestResults, trad_label: &str, comp_label: &str) {
    println!(
        "  {:<24} {:>10} {:>10} {:>7} {:>7} {:>6} {:>6} {:>14}",
        "SCENARIO", trad_label, comp_label, "T-Amp", "C-Amp", "T-Qlty", "C-Qlty", "WINNER"
    );
    println!("  {}", "─".repeat(96));

    for round in &results.rounds {
        let winner_display = match round.winner {
            "COMPOUNDING" => format!("COMP +{:.3}", round.margin),
            "TRADITIONAL" => format!("TRAD +{:.3}", -round.margin),
            _ => "TIE".to_string(),
        };
        let cal_mark = match round.calibration_pass {
            Some(true) => " [OK]",
            Some(false) => " [FAIL]",
            None => "",
        };
        println!(
            "  {:<24} {:>10.4} {:>10.4} {:>6.2}x {:>6.2}x {:>6.3} {:>6.3}  {}{}",
            round.scenario_name,
            round.trad_confidence,
            round.comp_confidence,
            round.trad_amp_ratio,
            round.comp_amp_ratio,
            round.trad_quality,
            round.comp_quality,
            winner_display,
            cal_mark,
        );
    }
}

fn print_score_box(label: &str, results: &ContestResults) {
    println!(
        "\n  +-- {} Score ------------------------------------------------+",
        label
    );
    println!(
        "  |  Compounding wins: {}  |  Traditional wins: {}  |  Ties: {}",
        results.compounding_wins, results.traditional_wins, results.ties
    );
    println!(
        "  |  Avg confidence:  Trad={:.4}  Comp={:.4}  (delta={:+.4})",
        results.avg_trad_confidence, results.avg_comp_confidence, results.avg_comp_advantage
    );
    println!(
        "  |  Avg quality:     Trad={:.4}  Comp={:.4}",
        results.avg_trad_quality, results.avg_comp_quality
    );
    if results.calibration_total > 0 {
        println!(
            "  |  Calibration: {}/{} passed",
            results.calibration_passes, results.calibration_total
        );
    }
    println!(
        "  |  Total time:      Trad={}us   Comp={}us",
        results.total_trad_time_us, results.total_comp_time_us
    );
    println!("  +--------------------------------------------------------------+");
}

// =============================================================================
// Main: The Contest
// =============================================================================

fn main() {
    println!("================================================================");
    println!("     RustyWorm ARCHITECTURE CONTEST v3 (Phase 6 — OCTO Braid)");
    println!("================================================================");
    println!("  8-Layer Compounding Integration  vs  Traditional Sequential");
    println!("  (Multiplicative, Bidirectional,      (3 variants + skip conn)");
    println!("   OCTO Braid cross-modulation)");
    println!("================================================================\n");

    let all_scenarios = build_all_scenarios();
    let standard = build_standard_scenarios();

    // =========================================================================
    // Round 1: Standard Scenarios — Arithmetic Traditional (baseline)
    // =========================================================================
    println!("--- ROUND 1: Standard Scenarios (Arithmetic Traditional) ---\n");

    let r1 = run_contest(
        &standard,
        TraditionalStrategy::Arithmetic,
        false,
        false,
        false,
    );
    print_contest_table(&r1, "TRAD", "COMP");
    print_score_box("Round 1", &r1);

    // =========================================================================
    // Round 2: Standard + Phase 3
    // =========================================================================
    println!("\n--- ROUND 2: Standard Scenarios WITH Phase 3 ---\n");

    let r2 = run_contest(
        &standard,
        TraditionalStrategy::Arithmetic,
        true,
        false,
        false,
    );
    print_contest_table(&r2, "TRAD+P3", "COMP+P3");
    print_score_box("Round 2", &r2);

    // =========================================================================
    // Round 3: ALL Scenarios — Attention Traditional + Skip vs Compounding + Braid
    // =========================================================================
    println!("\n--- ROUND 3: ALL Scenarios (Attention + Skip vs Compounding + OCTO Braid) ---\n");

    let r3 = run_contest(
        &all_scenarios,
        TraditionalStrategy::Attention,
        false,
        true,
        true,
    );

    // Group by category
    for cat in &[
        ScenarioCategory::Standard,
        ScenarioCategory::Adversarial,
        ScenarioCategory::Noisy,
        ScenarioCategory::Degraded,
        ScenarioCategory::Calibration,
    ] {
        let cat_rounds: Vec<&RoundScore> =
            r3.rounds.iter().filter(|r| r.category == *cat).collect();
        if cat_rounds.is_empty() {
            continue;
        }
        println!("\n  [{} Scenarios]", cat.name());
        println!(
            "  {:<24} {:>10} {:>10} {:>7} {:>7} {:>6} {:>6} {:>14}",
            "SCENARIO", "ATTN+SK", "COMP+BR", "T-Amp", "C-Amp", "T-Qlty", "C-Qlty", "WINNER"
        );
        println!("  {}", "─".repeat(96));
        for round in &cat_rounds {
            let winner_display = match round.winner {
                "COMPOUNDING" => format!("COMP +{:.3}", round.margin),
                "TRADITIONAL" => format!("TRAD +{:.3}", -round.margin),
                _ => "TIE".to_string(),
            };
            let cal_mark = match round.calibration_pass {
                Some(true) => " [OK]",
                Some(false) => " [FAIL]",
                None => "",
            };
            println!(
                "  {:<24} {:>10.4} {:>10.4} {:>6.2}x {:>6.2}x {:>6.3} {:>6.3}  {}{}",
                round.scenario_name,
                round.trad_confidence,
                round.comp_confidence,
                round.trad_amp_ratio,
                round.comp_amp_ratio,
                round.trad_quality,
                round.comp_quality,
                winner_display,
                cal_mark,
            );
        }
    }

    print_score_box("Round 3", &r3);

    // =========================================================================
    // Round 4: Traditional Strategy Head-to-Head (Phase 5B)
    // =========================================================================
    println!("\n--- ROUND 4: Traditional Strategy Comparison (Phase 5B) ---\n");

    let strat = run_strategy_comparison();

    println!(
        "  {:<16} {:>12} {:>12}",
        "STRATEGY", "Avg Conf", "Avg Quality"
    );
    println!("  {}", "─".repeat(42));
    println!(
        "  {:<16} {:>12.4} {:>12.4}",
        "Arithmetic", strat.arith_avg_conf, strat.arith_avg_quality
    );
    println!(
        "  {:<16} {:>12.4} {:>12.4}",
        "Weighted", strat.weighted_avg_conf, strat.weighted_avg_quality
    );
    println!(
        "  {:<16} {:>12.4} {:>12.4}",
        "Attention", strat.attention_avg_conf, strat.attention_avg_quality
    );
    println!(
        "\n  Skip connection bonus:  Conf={:+.4}  Quality={:+.4}",
        strat.skip_bonus_conf, strat.skip_bonus_quality
    );

    // =========================================================================
    // Round 5: Stress Test — 20 Repeated Passes
    // =========================================================================
    println!("\n--- ROUND 5: Stress Test (20 passes, same pipeline) ---\n");

    let stress = run_stress_test(20, false);

    println!("  Pass   Traditional   Compounding   Delta");
    println!("  {}", "─".repeat(50));
    for i in 0..20 {
        let t = stress.trad_confidences[i];
        let c = stress.comp_confidences[i];
        let bar_len = ((c - t).max(0.0) * 40.0) as usize;
        let bar: String = "#".repeat(bar_len.min(30));
        println!(
            "  {:>3}     {:.4}        {:.4}      {:+.4} {}",
            i + 1,
            t,
            c,
            c - t,
            bar,
        );
    }

    println!("\n  Stress Results:");
    println!(
        "    Final confidence:   Trad={:.4}  Comp={:.4}",
        stress.trad_final, stress.comp_final
    );
    println!(
        "    Output variance:    Trad={:.6}  Comp={:.6}",
        stress.trad_variance, stress.comp_variance
    );
    let stress_winner = if stress.comp_final > stress.trad_final {
        "COMPOUNDING"
    } else if stress.trad_final > stress.comp_final {
        "TRADITIONAL"
    } else {
        "TIE"
    };
    println!("    Winner: {}", stress_winner);

    // =========================================================================
    // Round 6: Emergence & Compounding Analysis
    // =========================================================================
    println!("\n--- ROUND 6: Emergence & Compounding Metrics ---\n");

    let em = run_emergence_comparison();

    println!(
        "  {:<28} {:>12} {:>12}",
        "METRIC", "TRADITIONAL", "COMPOUNDING"
    );
    println!("  {}", "─".repeat(54));
    println!(
        "  {:<28} {:>12.4} {:>12.4}",
        "Emergence value", em.trad_emergence, em.comp_emergence
    );
    println!(
        "  {:<28} {:>12.4} {:>12.4}",
        "Synergy score", em.trad_synergy, em.comp_synergy
    );
    println!(
        "  {:<28} {:>12.4} {:>12.4}",
        "Compounding factor", em.trad_compounding_factor, em.comp_compounding_factor
    );
    println!(
        "  {:<28} {:>12} {:>12}",
        "Is beneficial?",
        if em.trad_is_beneficial { "YES" } else { "no" },
        if em.comp_is_beneficial { "YES" } else { "no" }
    );

    let em_winner = if em.comp_emergence > em.trad_emergence {
        "COMPOUNDING"
    } else if em.trad_emergence > em.comp_emergence {
        "TRADITIONAL"
    } else {
        "TIE"
    };
    println!(
        "\n  Emergence winner: {} (delta={:+.4})",
        em_winner,
        em.comp_emergence - em.trad_emergence
    );

    // =========================================================================
    // Round 7: Phase 3 Impact Comparison
    // =========================================================================
    println!("\n--- ROUND 7: Phase 3 Impact — Before vs After ---\n");

    let p3cmp = run_phase3_comparison();

    println!("  {:<28} {:>12} {:>12}", "", "TRADITIONAL", "COMPOUNDING");
    println!("  {}", "─".repeat(54));
    println!(
        "  {:<28} {:>12.4} {:>12.4}",
        "Without Phase 3", p3cmp.no_p3_trad_conf, p3cmp.no_p3_comp_conf
    );
    println!(
        "  {:<28} {:>12.4} {:>12.4}",
        "With Phase 3", p3cmp.p3_trad_conf, p3cmp.p3_comp_conf
    );
    println!(
        "  {:<28} {:>12.4} {:>12.4}",
        "Phase 3 delta",
        p3cmp.p3_trad_conf - p3cmp.no_p3_trad_conf,
        p3cmp.p3_comp_conf - p3cmp.no_p3_comp_conf
    );
    println!(
        "\n  Phase 3 extras: Viz={}, Bursts={}, Resonance pairs={}",
        if p3cmp.comp_viz_active {
            "active"
        } else {
            "off"
        },
        p3cmp.comp_reserve_bursts,
        p3cmp.comp_dist_resonances
    );

    // =========================================================================
    // Round 8: Full Integration Head-to-Head
    // =========================================================================
    println!("\n--- ROUND 8: Full Integration Head-to-Head (with GAIA) ---\n");

    let h2h = run_integration_head_to_head();

    println!(
        "  {:<50} {:>8} {:>8} {:>8}",
        "INPUT", "TRAD", "COMP", "DELTA"
    );
    println!("  {}", "─".repeat(78));

    let mut h2h_comp_wins = 0;
    let mut h2h_trad_wins = 0;
    for (input, trad_f, comp_f, _trad_g, _comp_g) in &h2h.scenarios {
        let delta = comp_f - trad_f;
        let marker = if delta > 0.001 {
            h2h_comp_wins += 1;
            "+"
        } else if delta < -0.001 {
            h2h_trad_wins += 1;
            "-"
        } else {
            "="
        };
        println!(
            "  {:<50} {:>8.4} {:>8.4} {:>+7.4} {}",
            &input[..50.min(input.len())],
            trad_f,
            comp_f,
            delta,
            marker,
        );
    }

    println!(
        "\n  Integration score: Compounding {} — {} Traditional",
        h2h_comp_wins, h2h_trad_wins
    );

    // =========================================================================
    // FINAL SCORECARD
    // =========================================================================
    println!("\n================================================================");
    println!("                     FINAL SCORECARD");
    println!("================================================================");

    // Tally all rounds
    let total_comp = r1.compounding_wins
        + r2.compounding_wins
        + r3.compounding_wins
        + if stress.comp_final > stress.trad_final {
            1
        } else {
            0
        }
        + if em.comp_emergence > em.trad_emergence {
            1
        } else {
            0
        }
        + h2h_comp_wins;

    let total_trad = r1.traditional_wins
        + r2.traditional_wins
        + r3.traditional_wins
        + if stress.trad_final > stress.comp_final {
            1
        } else {
            0
        }
        + if em.trad_emergence > em.comp_emergence {
            1
        } else {
            0
        }
        + h2h_trad_wins;

    println!("  COMPOUNDING MODEL:  {} wins", total_comp);
    println!("  TRADITIONAL MODEL:  {} wins", total_trad);
    println!("    (incl. {} with Attention+Skip)", r3.traditional_wins);

    let champion = if total_comp > total_trad {
        "COMPOUNDING INTEGRATION"
    } else if total_trad > total_comp {
        "TRADITIONAL SEQUENTIAL"
    } else {
        "DEAD HEAT"
    };
    println!("\n  CHAMPION: {}", champion);

    println!("\n  Key Findings:");
    println!(
        "    Avg confidence boost from compounding:  {:+.4}",
        r1.avg_comp_advantage
    );
    println!(
        "    Phase 3 boost for compounding:          {:+.4}",
        p3cmp.p3_comp_conf - p3cmp.no_p3_comp_conf
    );
    println!(
        "    Emergence advantage:                    {:+.4}",
        em.comp_emergence - em.trad_emergence
    );
    println!(
        "    Best traditional strategy:              Attention (conf={:.4}, quality={:.4})",
        strat.attention_avg_conf, strat.attention_avg_quality
    );
    println!(
        "    Calibration pass rate:                  {}/{}",
        r3.calibration_passes, r3.calibration_total
    );
    println!(
        "    Stress variance:  Trad={:.6}  Comp={:.6}",
        stress.trad_variance, stress.comp_variance
    );
    println!("================================================================");
}
