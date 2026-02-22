use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustyworm::*;

fn bench_system1_cache_lookup(c: &mut Criterion) {
    let store = AiProfileStore::default();
    let _profile = store.get("gpt4o").unwrap().clone();
    let mut cache = SignatureCache::new();
    // Pre-warm the cache
    let mut analyzer = BehaviorAnalyzer::new();
    let responses =
        vec!["I'd be happy to help with that. Here's what you need to know...".to_string()];
    let sig = analyzer.build_signature("gpt4o", &responses);
    cache.compile_from(&sig);

    c.bench_function("system1_cache_lookup", |b| {
        b.iter(|| {
            let result = cache.lookup("gpt4o");
            black_box(result.is_some());
        })
    });
}

fn bench_system2_analyze_response(c: &mut Criterion) {
    let analyzer = BehaviorAnalyzer::new();
    let sample = "I'd be happy to help you with that! Let me break this down step by step. First, we need to consider the key factors involved. It's worth noting that there are multiple approaches, but I think the most effective one would be...";

    c.bench_function("system2_analyze_response", |b| {
        b.iter(|| {
            black_box(analyzer.analyze_response(black_box(sample)));
        })
    });
}

fn bench_profile_similarity(c: &mut Criterion) {
    let store = AiProfileStore::default();
    let gpt4o = store.get("gpt4o").unwrap();
    let claude = store.get("claude").unwrap();

    c.bench_function("profile_similarity", |b| {
        b.iter(|| {
            black_box(gpt4o.similarity_to(black_box(claude)));
        })
    });
}

fn bench_template_generation(c: &mut Criterion) {
    let store = AiProfileStore::default();
    let profile = store.get("claude").unwrap().clone();
    let mut template_store = TemplateStore::new();
    let lib = template_store.get_or_create(&profile);

    c.bench_function("template_generation", |b| {
        b.iter(|| {
            black_box(lib.generate(
                black_box("How do neural networks work?"),
                black_box(&profile.response_style),
            ));
        })
    });
}

fn bench_persona_snapshot_roundtrip(c: &mut Criterion) {
    let store = AiProfileStore::default();
    let profile = store.get("gpt4o").unwrap();
    let persona = CompoundPersona::from_profile(profile);
    let snapshot = persona.snapshot();
    let json = serde_json::to_string(&snapshot).unwrap();

    c.bench_function("persona_snapshot_serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(black_box(&snapshot)).unwrap());
        })
    });

    c.bench_function("persona_snapshot_deserialize", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<CompoundPersonaSnapshot>(black_box(&json)).unwrap());
        })
    });
}

fn bench_system1_vs_system2(c: &mut Criterion) {
    let mut group = c.benchmark_group("system1_vs_system2");

    // Setup System 1
    let store = AiProfileStore::default();
    let _profile = store.get("gpt4o").unwrap().clone();
    let mut cache = SignatureCache::new();
    let mut analyzer = BehaviorAnalyzer::new();
    let responses = vec!["Test response for caching".to_string()];
    let sig = analyzer.build_signature("gpt4o", &responses);
    cache.compile_from(&sig);

    let sample = "I'd be happy to help! Let me explain this concept clearly...";

    group.bench_function("system1_cached_lookup", |b| {
        b.iter(|| {
            let result = cache.lookup("gpt4o");
            black_box(result.is_some());
        })
    });

    group.bench_function("system2_full_analysis", |b| {
        b.iter(|| {
            let mut a = BehaviorAnalyzer::new();
            let responses = vec![black_box(sample).to_string()];
            let sig = a.build_signature("gpt4o", &responses);
            black_box(sig);
        })
    });

    group.finish();
}

// =============================================================================
// Layer System Benchmarks (Phase 5C)
// =============================================================================

#[cfg(feature = "layers")]
mod layer_benches {
    use super::*;
    use rustyworm::mimicry::layers::{
        bridges::BridgeBuilder,
        layer::{Layer, LayerState},
        metrics::compute_quality,
        stack::{LayerStack, LayerStackConfig},
    };

    /// Benchmark: Forward-only processing (no backward propagation).
    pub fn bench_forward_only(c: &mut Criterion) {
        c.bench_function("layers/forward_only", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_max_iterations(0)
                    .without_backward_propagation()
                    .with_max_confidence(2.0);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "benchmark forward processing".to_string(),
                    0.6,
                );
                let result = stack.process_forward(input);
                black_box(result.combined_confidence);
            })
        });
    }

    /// Benchmark: Bidirectional processing (the full compounding pipeline).
    pub fn bench_bidirectional(c: &mut Criterion) {
        c.bench_function("layers/bidirectional", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "benchmark bidirectional processing".to_string(),
                    0.6,
                );
                let result = stack.process_bidirectional(input);
                black_box(result.combined_confidence);
            })
        });
    }

    /// Benchmark: Forward vs Bidirectional in a comparison group.
    pub fn bench_forward_vs_bidirectional(c: &mut Criterion) {
        let mut group = c.benchmark_group("layers/fwd_vs_bidir");

        group.bench_function("forward", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_max_iterations(0)
                    .without_backward_propagation()
                    .with_max_confidence(2.0);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "benchmark fwd".to_string(),
                    0.6,
                );
                black_box(stack.process_forward(input));
            })
        });

        group.bench_function("bidirectional", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "benchmark bidir".to_string(),
                    0.6,
                );
                black_box(stack.process_bidirectional(input));
            })
        });

        group.finish();
    }

    /// Benchmark: Phase 3 overhead (with vs without).
    pub fn bench_phase3_overhead(c: &mut Criterion) {
        let mut group = c.benchmark_group("layers/phase3_overhead");

        group.bench_function("without_phase3", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "phase3 bench".to_string(),
                    0.6,
                );
                black_box(stack.process_bidirectional(input));
            })
        });

        group.bench_function("with_phase3", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_phase3()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "phase3 bench".to_string(),
                    0.6,
                );
                black_box(stack.process_bidirectional(input));
            })
        });

        group.finish();
    }

    /// Benchmark: Phase 5 overhead (with vs without adaptive systems).
    pub fn bench_phase5_overhead(c: &mut Criterion) {
        let mut group = c.benchmark_group("layers/phase5_overhead");

        group.bench_function("without_phase5", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "phase5 bench".to_string(),
                    0.6,
                );
                black_box(stack.process_bidirectional(input));
            })
        });

        group.bench_function("with_phase5", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_phase5()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "phase5 bench".to_string(),
                    0.6,
                );
                black_box(stack.process_bidirectional(input));
            })
        });

        group.bench_function("with_all_subsystems", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new()
                    .with_all_subsystems()
                    .with_max_iterations(5)
                    .with_max_confidence(2.0)
                    .with_amplification_damping(0.8);
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                let input = LayerState::with_confidence(
                    Layer::BasePhysics,
                    "all subsystems bench".to_string(),
                    0.6,
                );
                black_box(stack.process_bidirectional(input));
            })
        });

        group.finish();
    }

    /// Benchmark: Scaling with number of bridges.
    pub fn bench_bridge_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("layers/bridge_scaling");

        let all_bridges = BridgeBuilder::build_all();

        for bridge_count in [1, 5, 10, 15] {
            let count = bridge_count.min(all_bridges.len());
            group.bench_function(format!("{}_bridges", count), |b| {
                b.iter(|| {
                    let config = LayerStackConfig::new()
                        .with_max_iterations(5)
                        .with_max_confidence(2.0)
                        .with_amplification_damping(0.8);
                    let mut stack = LayerStack::with_config(config);
                    // Register only `count` bridges
                    for bridge in BridgeBuilder::build_all().into_iter().take(count) {
                        stack.register_bridge(bridge);
                    }
                    let input = LayerState::with_confidence(
                        Layer::BasePhysics,
                        "bridge scaling bench".to_string(),
                        0.6,
                    );
                    black_box(stack.process_bidirectional(input));
                })
            });
        }

        group.finish();
    }

    /// Benchmark: Quality metrics computation overhead.
    pub fn bench_quality_metrics(c: &mut Criterion) {
        // Pre-compute a result to measure metrics overhead in isolation
        let config = LayerStackConfig::new()
            .with_max_iterations(5)
            .with_max_confidence(2.0)
            .with_amplification_damping(0.8);
        let mut stack = LayerStack::with_config(config);
        for bridge in BridgeBuilder::build_all() {
            stack.register_bridge(bridge);
        }
        let input =
            LayerState::with_confidence(Layer::BasePhysics, "metrics bench".to_string(), 0.6);
        let result = stack.process_bidirectional(input);

        c.bench_function("layers/quality_metrics", |b| {
            b.iter(|| {
                black_box(compute_quality(black_box(&result)));
            })
        });
    }

    /// Benchmark: Stack creation overhead (with all subsystems).
    pub fn bench_stack_creation(c: &mut Criterion) {
        let mut group = c.benchmark_group("layers/stack_creation");

        group.bench_function("minimal", |b| {
            b.iter(|| {
                let stack = LayerStack::new();
                black_box(stack);
            })
        });

        group.bench_function("with_bridges", |b| {
            b.iter(|| {
                let mut stack = LayerStack::new();
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                black_box(stack);
            })
        });

        group.bench_function("full_subsystems", |b| {
            b.iter(|| {
                let config = LayerStackConfig::new().with_all_subsystems();
                let mut stack = LayerStack::with_config(config);
                for bridge in BridgeBuilder::build_all() {
                    stack.register_bridge(bridge);
                }
                black_box(stack);
            })
        });

        group.finish();
    }
}

// =============================================================================
// Criterion group configuration
// =============================================================================

criterion_group!(
    core_benches,
    bench_system1_cache_lookup,
    bench_system2_analyze_response,
    bench_profile_similarity,
    bench_template_generation,
    bench_persona_snapshot_roundtrip,
    bench_system1_vs_system2,
);

#[cfg(feature = "layers")]
criterion_group!(
    layer_system_benches,
    layer_benches::bench_forward_only,
    layer_benches::bench_bidirectional,
    layer_benches::bench_forward_vs_bidirectional,
    layer_benches::bench_phase3_overhead,
    layer_benches::bench_phase5_overhead,
    layer_benches::bench_bridge_scaling,
    layer_benches::bench_quality_metrics,
    layer_benches::bench_stack_creation,
);

#[cfg(feature = "layers")]
criterion_main!(core_benches, layer_system_benches);

#[cfg(not(feature = "layers"))]
criterion_main!(core_benches);
