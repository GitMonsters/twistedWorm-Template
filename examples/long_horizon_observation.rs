//! Long-Horizon Observation Example
//!
//! This example demonstrates how to use the LongHorizonObserver to track
//! patterns across 100+ conversation turns for improved mimicry fidelity.
//!
//! Requirements:
//! - Build with `--features agentdock`
//!
//! Run:
//! ```bash
//! cargo run --features agentdock --example long_horizon_observation
//! ```

use rustyworm::mimicry::engine::MimicryEngine;

#[cfg(feature = "agentdock")]
use rustyworm::mimicry::long_horizon::{
    ContextWindow, LongHorizonConfig, LongHorizonObserver, PatternType, StrategyAdjustment,
};

fn main() {
    println!("=== RustyWorm Long-Horizon Observation Example ===\n");

    #[cfg(not(feature = "agentdock"))]
    {
        println!("This example requires the 'agentdock' feature.");
        println!("Run with: cargo run --features agentdock --example long_horizon_observation");
        return;
    }

    #[cfg(feature = "agentdock")]
    run_long_horizon_demo();
}

#[cfg(feature = "agentdock")]
fn run_long_horizon_demo() {
    // Step 1: Configure long-horizon observation
    println!("Step 1: Configuring Long-Horizon Observer");
    println!("-----------------------------------------");

    let config = LongHorizonConfig {
        max_context_turns: 200,
        pattern_detection_window: 50,
        strategy_adjustment_threshold: 0.8,
        context_compression_ratio: 0.5,
        memory_decay_rate: 0.95,
        pattern_types: vec![
            PatternType::Repetition,
            PatternType::Escalation,
            PatternType::TopicDrift,
            PatternType::StyleShift,
            PatternType::SentimentTrend,
        ],
        ..Default::default()
    };

    println!("  Max context turns: {}", config.max_context_turns);
    println!(
        "  Pattern detection window: {}",
        config.pattern_detection_window
    );
    println!(
        "  Strategy threshold: {}",
        config.strategy_adjustment_threshold
    );
    println!("  Memory decay rate: {}", config.memory_decay_rate);
    println!();

    // Step 2: Create the observer
    println!("Step 2: Creating Long-Horizon Observer");
    println!("--------------------------------------");

    let mut observer = LongHorizonObserver::new(config);
    println!("  Observer created");
    println!("  Initial context length: {}", observer.context_length());
    println!();

    // Step 3: Simulate a long conversation
    println!("Step 3: Simulating Long Conversation (120 turns)");
    println!("-------------------------------------------------");

    let conversation_turns = generate_sample_conversation();

    for (i, turn) in conversation_turns.iter().enumerate() {
        observer.observe(turn);

        // Print progress every 20 turns
        if (i + 1) % 20 == 0 {
            println!(
                "  Turn {}: Context size = {}, Patterns detected = {}",
                i + 1,
                observer.context_length(),
                observer.pattern_count()
            );
        }
    }
    println!();

    // Step 4: Analyze detected patterns
    println!("Step 4: Detected Patterns");
    println!("-------------------------");

    let patterns = observer.get_patterns();
    for pattern in &patterns {
        println!("  Pattern: {:?}", pattern.pattern_type);
        println!("    - First seen: turn {}", pattern.first_occurrence);
        println!("    - Last seen: turn {}", pattern.last_occurrence);
        println!("    - Frequency: {:.2}", pattern.frequency);
        println!("    - Confidence: {:.2}", pattern.confidence);
        println!();
    }

    // Step 5: Get strategy adjustments
    println!("Step 5: Strategy Adjustments");
    println!("----------------------------");

    let adjustments = observer.get_strategy_adjustments();
    for adj in &adjustments {
        println!("  Adjustment: {}", adj.name);
        println!("    - Reason: {}", adj.reason);
        println!("    - Priority: {:?}", adj.priority);
        println!("    - Suggested action: {}", adj.suggested_action);
        println!();
    }

    // Step 6: Context window analysis
    println!("Step 6: Context Window Analysis");
    println!("-------------------------------");

    let context = observer.get_context_window();
    println!("  Total turns observed: {}", context.total_turns);
    println!("  Active context size: {}", context.active_size);
    println!("  Compressed history size: {}", context.compressed_size);
    println!(
        "  Compression ratio: {:.2}%",
        context.compression_ratio * 100.0
    );
    println!();

    // Topic distribution
    println!("  Topic Distribution:");
    for (topic, weight) in &context.topic_weights {
        let bar = "#".repeat((weight * 20.0) as usize);
        println!("    {:<15} {:>5.1}% {}", topic, weight * 100.0, bar);
    }
    println!();

    // Step 7: Memory decay visualization
    println!("Step 7: Memory Decay Over Time");
    println!("------------------------------");

    println!("  Turn | Memory Weight | Visual");
    println!("  -----|---------------|-------");
    for i in [0, 20, 40, 60, 80, 100, 120] {
        if i < conversation_turns.len() {
            let weight = observer.get_memory_weight(i);
            let bar = "*".repeat((weight * 30.0) as usize);
            println!("  {:>4} | {:>13.2} | {}", i, weight, bar);
        }
    }
    println!();

    // Step 8: Integration with MimicryEngine
    println!("Step 8: Integration Example");
    println!("---------------------------");

    println!("  // In production code:");
    println!("  let mut engine = MimicryEngine::new();");
    println!("  engine.set_long_horizon_observer(observer);");
    println!("  ");
    println!("  // The observer now informs evolution:");
    println!("  engine.observe_and_evolve(\"user input\").await;");
    println!("  ");
    println!("  // Patterns influence persona adaptation:");
    println!("  let persona = engine.current_persona();");
    println!("  // persona.style adjusted based on detected patterns");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(feature = "agentdock")]
fn generate_sample_conversation() -> Vec<String> {
    let mut turns = Vec::new();

    // Simulate a conversation with patterns
    let topics = ["coding", "debugging", "testing", "deployment", "monitoring"];
    let sentiments = ["neutral", "curious", "frustrated", "satisfied", "excited"];

    for i in 0..120 {
        let topic = topics[i % topics.len()];
        let sentiment = sentiments[(i / 10) % sentiments.len()];

        let turn = if i % 2 == 0 {
            format!(
                "User (turn {}): Can you help me with {} (feeling {})?",
                i, topic, sentiment
            )
        } else {
            format!(
                "Assistant (turn {}): I'd be happy to help with {}. [{}]",
                i, topic, sentiment
            )
        };

        turns.push(turn);
    }

    turns
}
