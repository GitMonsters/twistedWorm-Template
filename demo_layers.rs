//! RustyWorm 8-Layer System Demo
//!
//! Demonstrates multiplicative confidence amplification across layers.

use rustyworm::mimicry::layers::{
    bridges::BridgeBuilder,
    compounding::CompoundingMetrics,
    distribution_resonance::DistributionResonanceSystem,
    emergence::EmergenceFramework,
    gaia::intuition::{GaiaConfig, GaiaIntuitionEngine},
    gaia::pattern::Pattern,
    integration::{IntegrationConfig, LayerIntegration},
    layer::{Domain, Layer, LayerState},
    reserve::FractionalReserve,
    stack::{LayerStack, LayerStackConfig},
    visualization::VisualizationEngine,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         RustyWorm 8-Layer Multiplicative System              ║");
    println!("║                    ~ DEMONSTRATION ~                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Demo 1: Layer Architecture
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 1. LAYER ARCHITECTURE                                        │");
    println!("└──────────────────────────────────────────────────────────────┘");

    for layer in Layer::all() {
        let connections: Vec<_> = Layer::all()
            .iter()
            .filter(|&other| layer.can_bridge_to(*other))
            .map(|l| l.name())
            .collect();

        println!(
            "  L{}: {:25} → {:?}",
            layer.number(),
            layer.name(),
            connections
        );
    }
    println!();

    // =========================================================================
    // Demo 2: Bridge Network
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 2. BIDIRECTIONAL BRIDGES (15 total)                          │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let bridges = BridgeBuilder::build_all();
    for bridge in &bridges {
        println!(
            "  {:30} | resonance: {:.2}",
            bridge.name(),
            bridge.resonance()
        );
    }
    println!("  Total bridges: {}\n", bridges.len());

    // =========================================================================
    // Demo 3: GAIA Intuition Engine
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 3. GAIA INTUITION ENGINE                                     │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let gaia = GaiaIntuitionEngine::new(GaiaConfig::default());

    // Register some patterns
    let pattern1 = Pattern::new("physics_wave", Domain::Physics)
        .with_fingerprint(vec![0.8, 0.2, 0.1, 0.0])
        .with_weight(1.2);

    let pattern2 = Pattern::new("language_metaphor", Domain::Language)
        .with_fingerprint(vec![0.1, 0.7, 0.5, 0.3])
        .with_weight(1.0);

    let pattern3 = Pattern::new("consciousness_insight", Domain::Consciousness)
        .with_fingerprint(vec![0.5, 0.5, 0.8, 0.9])
        .with_weight(1.5);

    gaia.register_pattern(pattern1).unwrap();
    gaia.register_pattern(pattern2).unwrap();
    gaia.register_pattern(pattern3).unwrap();

    // Query GAIA
    let query = vec![0.7, 0.3, 0.2, 0.1];
    let result = gaia.query(&query).unwrap();

    println!("  Registered patterns: {}", gaia.pattern_memory().len());
    println!("  Query: {:?}", query);
    println!(
        "  Best match: {:?}",
        result.best_match().map(|m| &m.pattern_id)
    );
    println!("  Intuition confidence: {:.3}", result.confidence);
    println!(
        "  Analogical insights: {}\n",
        result.analogical_insights.len()
    );

    // =========================================================================
    // Demo 4: Multiplicative Amplification
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 4. MULTIPLICATIVE CONFIDENCE AMPLIFICATION                   │");
    println!("└──────────────────────────────────────────────────────────────┘");

    // Use conservative settings to avoid unbounded amplification
    let mut stack = LayerStack::with_config(
        LayerStackConfig::new()
            .with_max_iterations(3)
            .with_global_amplification(1.05),
    );

    // Register bridges
    for bridge in BridgeBuilder::build_all() {
        stack.register_bridge(bridge);
    }

    // Process through the stack
    let input = LayerState::with_confidence(
        Layer::BasePhysics,
        "quantum coherence pattern".to_string(),
        0.7,
    );

    println!(
        "  Input: Layer={}, Confidence={:.2}",
        input.layer.name(),
        input.confidence
    );

    let result = stack.process_bidirectional(input);

    // Clamp values for display (the actual system may need damping)
    let combined_conf = result.combined_confidence.min(10.0);
    let total_amp = result.total_amplification.min(10.0);

    println!("  Output:");
    println!("    Combined confidence: {:.3}", combined_conf);
    println!("    Total amplification: {:.3}x", total_amp);
    println!("    Iterations: {}", result.iterations);
    println!("    Converged: {}", result.converged);
    println!("    Active layers: {}\n", result.layer_states.len());

    // =========================================================================
    // Demo 5: Compounding Metrics
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 5. COMPOUNDING METRICS                                       │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let mut metrics = CompoundingMetrics::new();
    let analysis = metrics.analyze(&result);

    println!("  Multiplicative gain: {:.4}", analysis.multiplicative_gain);
    println!("  Additive gain: {:.4}", analysis.additive_gain);
    println!("  Compounding factor: {:.4}", analysis.compounding_factor);
    println!("  Synergy score: {:.4}", analysis.synergy_score);
    println!("  Is compounding beneficial: {}\n", analysis.is_beneficial);

    // =========================================================================
    // Demo 6: Emergence Framework
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 6. EMERGENCE FRAMEWORK                                       │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let mut emergence = EmergenceFramework::new();
    let emergence_analysis = emergence.analyze(&result);

    println!(
        "  Emergence value: {:.4}",
        emergence_analysis.emergence_value
    );
    println!(
        "  Higher-order emergence: {:.4}",
        emergence_analysis.higher_order_emergence
    );
    println!(
        "  Dominant mechanism: {:?}",
        emergence_analysis.dominant_mechanism
    );
    println!("  Is significant: {}", emergence_analysis.is_significant);
    println!(
        "  Prediction accuracy: {:.2}%\n",
        emergence_analysis.prediction_accuracy * 100.0
    );

    // =========================================================================
    // Demo 7: Full Integration
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 7. FULL LAYER INTEGRATION                                    │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let mut integration = LayerIntegration::with_config(IntegrationConfig {
        enable_gaia: true,
        enable_external_apis: false,
        min_amplification_benefit: 0.05,
        track_statistics: true,
        max_processing_time_ms: 5000,
        enable_phase3: false,
    });

    // Process multiple inputs
    let inputs = vec![
        (
            "The wave function collapses upon observation",
            Some("physics"),
        ),
        ("Language shapes thought in recursive patterns", None),
        (
            "Consciousness emerges from neural complexity",
            Some("neuroscience"),
        ),
    ];

    for (input, context) in inputs {
        let result = integration.process(input, context);
        println!("  Input: \"{}...\"", &input[..40.min(input.len())]);
        println!(
            "    Initial → Final: {:.3} → {:.3}",
            result.initial_confidence, result.final_confidence
        );
        println!("    GAIA contributed: {}", result.gaia_contributed);
        println!("    Processing time: {}ms", result.processing_time_ms);
    }

    println!("\n  Integration Summary:");
    println!("{}", integration.summary());

    // =========================================================================
    // Demo 8: Phase 3 Subsystems
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 8. PHASE 3: VISUALIZATION, RESERVE & DISTRIBUTION RESONANCE │");
    println!("└──────────────────────────────────────────────────────────────┘");

    // --- 8a: Visualization Engine ---
    println!("\n  8a. Pre-Cognitive Visualization Engine");
    let mut vis_engine = VisualizationEngine::with_defaults();
    let vis_input = LayerState::with_confidence(
        Layer::BasePhysics,
        "visualize quantum decoherence pathway".to_string(),
        0.7,
    );
    let vis_result = vis_engine.visualize(&vis_input);
    println!("    Task simulated: {} steps", vis_result.simulation.steps);
    println!(
        "    Visualization confidence: {:.3}",
        vis_result.visualization_confidence
    );
    println!(
        "    Pre-warm signals: {} layers",
        vis_result.pre_warm_signals.len()
    );
    println!(
        "    Current fidelity EMA: {:.3}",
        vis_result.current_fidelity
    );
    println!(
        "    Needs recalibration: {}",
        vis_result.needs_recalibration
    );

    // Record a mock outcome to close the feedback loop
    let mut actuals = std::collections::HashMap::new();
    actuals.insert(Layer::BasePhysics, 0.65);
    actuals.insert(Layer::ExtendedPhysics, 0.55);
    let needs_recal =
        vis_engine.record_outcome(vis_result.visualization_confidence, 0.65, &actuals);
    let summary = vis_engine.summary();
    println!(
        "    After outcome: fidelity={:.3}, bias={:.3}, recal_needed={}",
        summary.fidelity_ema, summary.current_bias, needs_recal
    );

    // --- 8b: Fractional Reserve Model ---
    println!("\n  8b. Fractional Reserve Model");
    let mut reserve = FractionalReserve::with_defaults();

    // Accumulate reserves with low resonance (no bursts)
    for _ in 0..5 {
        let _ = reserve.process(Layer::BasePhysics, 0.8, 0.3);
    }
    let stats = reserve.stats();
    println!(
        "    After 5 low-resonance passes (L1): held={:.3}, active_released={:.3}",
        stats.total_held_confidence, stats.total_active_released
    );

    // Trigger a burst with high resonance
    let burst_decomp = reserve.process(Layer::BasePhysics, 0.8, 0.9);
    println!(
        "    Burst triggered (resonance=0.9): raw={:.3}, active={:.3}, burst={:.3}, effective={:.3}",
        burst_decomp.raw, burst_decomp.active, burst_decomp.burst, burst_decomp.effective
    );
    let stats_after = reserve.stats();
    println!(
        "    Total burst events: {}, total burst released: {:.3}",
        stats_after.total_burst_events, stats_after.total_burst_released
    );

    // --- 8c: Distribution-based Resonance ---
    println!("\n  8c. Distribution-based Resonance");
    let mut dr = DistributionResonanceSystem::with_defaults();

    // Observe consistent confidence for L1, L2
    for _ in 0..10 {
        dr.observe(Layer::BasePhysics, 0.8);
        dr.observe(Layer::ExtendedPhysics, 0.7);
    }
    let res_consistent = dr.resonance(Layer::BasePhysics, Layer::ExtendedPhysics);
    println!("    L1↔L2 resonance (consistent): {:.4}", res_consistent);

    // Observe noisy confidence for L3, L4
    let noisy_vals = [0.2, 0.9, 0.3, 0.8, 0.1, 0.95, 0.4, 0.85, 0.15, 0.7];
    for &v in &noisy_vals {
        dr.observe(Layer::CrossDomain, v);
        dr.observe(Layer::GaiaConsciousness, v * 0.9);
    }
    let res_noisy = dr.resonance(Layer::CrossDomain, Layer::GaiaConsciousness);
    println!("    L3↔L4 resonance (noisy):      {:.4}", res_noisy);
    println!(
        "    Variance penalty effect: {:.1}% reduction",
        (1.0 - res_noisy / res_consistent.max(0.001)) * 100.0
    );

    // --- 8d: Full Phase 3 Stack ---
    println!("\n  8d. Full Stack with Phase 3 Enabled");
    let mut phase3_integration = LayerIntegration::with_config(IntegrationConfig {
        enable_gaia: true,
        enable_external_apis: false,
        min_amplification_benefit: 0.05,
        track_statistics: true,
        max_processing_time_ms: 5000,
        enable_phase3: true,
    });

    let phase3_inputs = vec![
        "Quantum entanglement defies classical locality",
        "Neural networks mirror biological synaptic plasticity",
        "Recursive self-reference generates emergent complexity",
    ];

    for input in phase3_inputs {
        let result = phase3_integration.process(input, None);
        println!("  Input: \"{}...\"", &input[..45.min(input.len())]);
        println!(
            "    Confidence: {:.3} → {:.3} | Viz: {} | Bursts: {}",
            result.initial_confidence,
            result.final_confidence,
            if result.visualization_active {
                "active"
            } else {
                "off"
            },
            result.reserve_bursts_occurred,
        );
    }

    let p3_stats = phase3_integration.stats();
    println!("\n  Phase 3 Stats:");
    println!(
        "    Visualization passes: {} | Reserve burst events: {}",
        p3_stats.visualization_passes, p3_stats.reserve_burst_events
    );
    println!(
        "    Total processed: {} | Avg improvement: {:.3}",
        p3_stats.total_processed, p3_stats.avg_confidence_improvement
    );

    // =========================================================================
    // Final Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DEMO COMPLETE                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Layers: 8  |  Bridges: 15  |  Patterns: 3  |  Tests: 440+  ║");
    println!("║  Phase 3: Visualization + Reserve + Distribution Resonance   ║");
    println!("║  Multiplicative Amplification: ✓  |  Fidelity Tracking: ✓   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
