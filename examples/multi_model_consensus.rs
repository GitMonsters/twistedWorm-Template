//! Multi-Model Consensus Example
//!
//! This example demonstrates how to use the MultiModelObserver to build
//! consensus across multiple AI models for higher-fidelity mimicry.
//!
//! Requirements:
//! - Build with `--features agentdock`
//!
//! Run:
//! ```bash
//! cargo run --features agentdock --example multi_model_consensus
//! ```

use rustyworm::mimicry::engine::MimicryEngine;

#[cfg(feature = "agentdock")]
use rustyworm::mimicry::multi_model::{
    AggregationStrategy, ConsensusConfig, ConsensusResult, ModelEndpoint, ModelWeight,
    MultiModelObserver,
};

fn main() {
    println!("=== RustyWorm Multi-Model Consensus Example ===\n");

    #[cfg(not(feature = "agentdock"))]
    {
        println!("This example requires the 'agentdock' feature.");
        println!("Run with: cargo run --features agentdock --example multi_model_consensus");
        return;
    }

    #[cfg(feature = "agentdock")]
    run_multi_model_demo();
}

#[cfg(feature = "agentdock")]
fn run_multi_model_demo() {
    // Step 1: Configure multi-model consensus
    println!("Step 1: Configuring Multi-Model Observer");
    println!("-----------------------------------------");

    let config = ConsensusConfig {
        min_models_for_consensus: 2,
        agreement_threshold: 0.7,
        timeout_ms: 30000,
        aggregation_strategy: AggregationStrategy::WeightedVoting,
        fallback_on_disagreement: true,
        model_weights: vec![
            ModelWeight {
                model: "gpt-4o".to_string(),
                weight: 1.0,
            },
            ModelWeight {
                model: "claude".to_string(),
                weight: 1.2,
            },
            ModelWeight {
                model: "gemini".to_string(),
                weight: 0.9,
            },
        ],
        ..Default::default()
    };

    println!("  Aggregation strategy: {:?}", config.aggregation_strategy);
    println!("  Agreement threshold: {}", config.agreement_threshold);
    println!(
        "  Min models for consensus: {}",
        config.min_models_for_consensus
    );
    println!("  Timeout: {}ms", config.timeout_ms);
    println!();

    // Step 2: Create the observer and add models
    println!("Step 2: Creating Observer and Adding Models");
    println!("--------------------------------------------");

    let mut observer = MultiModelObserver::new(config);

    // Simulated model endpoints (in production, these would be real API endpoints)
    let models = vec![
        ("gpt-4o", "https://api.openai.com/v1/chat/completions"),
        ("claude", "https://api.anthropic.com/v1/messages"),
        (
            "gemini",
            "https://generativelanguage.googleapis.com/v1/models/gemini-pro",
        ),
    ];

    for (name, endpoint) in &models {
        observer.add_model(
            name,
            ModelEndpoint {
                url: endpoint.to_string(),
                auth_header: "Authorization".to_string(),
                model_id: name.to_string(),
                max_tokens: 4096,
                temperature: 0.7,
            },
        );
        println!("  Added model: {} -> {}", name, endpoint);
    }
    println!();

    // Step 3: Simulate model responses
    println!("Step 3: Simulating Model Responses");
    println!("-----------------------------------");

    let prompt = "Explain the concept of recursion in programming.";
    println!("  Prompt: \"{}\"\n", prompt);

    // Simulated responses (in production, these would come from actual API calls)
    let responses = vec![
        ("gpt-4o", "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems. Each recursive call works on a smaller piece of the problem until reaching a base case that can be solved directly."),
        ("claude", "Recursion in programming is when a function calls itself as part of its execution. It's a powerful technique for solving problems that can be broken down into similar, smaller subproblems. The key components are: 1) A base case that stops the recursion, and 2) A recursive case that calls the function with modified arguments."),
        ("gemini", "Recursion is a method where a function invokes itself repeatedly until a specific condition (base case) is met. It's commonly used for tasks like traversing tree structures, calculating factorials, or implementing divide-and-conquer algorithms."),
    ];

    for (model, response) in &responses {
        println!(
            "  {}: \"{}...\"",
            model,
            &response[..60.min(response.len())]
        );
    }
    println!();

    // Step 4: Build consensus
    println!("Step 4: Building Consensus");
    println!("--------------------------");

    // Simulate consensus building
    let consensus = ConsensusResult {
        response: "Recursion is a programming technique where a function calls itself to solve problems by breaking them into smaller subproblems. Key elements include a base case (stopping condition) and recursive case (self-call with modified arguments). It's useful for tree traversal, divide-and-conquer algorithms, and problems with self-similar structure.".to_string(),
        agreement_score: 0.87,
        participating_models: vec!["gpt-4o".to_string(), "claude".to_string(), "gemini".to_string()],
        model_votes: vec![
            ("gpt-4o".to_string(), 0.85),
            ("claude".to_string(), 0.92),
            ("gemini".to_string(), 0.84),
        ],
        dissenting_views: vec![],
        confidence: 0.89,
        latency_ms: 1250,
    };

    println!("  Consensus Response:");
    println!("  -------------------");
    println!("  {}", consensus.response);
    println!();
    println!("  Metrics:");
    println!("    Agreement score: {:.2}", consensus.agreement_score);
    println!("    Confidence: {:.2}", consensus.confidence);
    println!("    Latency: {}ms", consensus.latency_ms);
    println!(
        "    Participating models: {:?}",
        consensus.participating_models
    );
    println!();

    // Step 5: Analyze model contributions
    println!("Step 5: Model Contributions");
    println!("---------------------------");

    for (model, vote) in &consensus.model_votes {
        let bar = "█".repeat((vote * 20.0) as usize);
        println!("  {:<10} {:>5.1}% {}", model, vote * 100.0, bar);
    }
    println!();

    // Step 6: Aggregation strategies comparison
    println!("Step 6: Aggregation Strategies");
    println!("------------------------------");

    let strategies = [
        (
            AggregationStrategy::WeightedVoting,
            "Weighted Voting",
            "Each model's vote is weighted by configured importance",
        ),
        (
            AggregationStrategy::MajorityVoting,
            "Majority Voting",
            "Simple majority wins, equal weights",
        ),
        (
            AggregationStrategy::Ensemble,
            "Ensemble",
            "Combines all responses into a unified output",
        ),
        (
            AggregationStrategy::BestOfN,
            "Best of N",
            "Selects the highest-quality response",
        ),
    ];

    for (strategy, name, description) in &strategies {
        println!("  {:?}", strategy);
        println!("    Name: {}", name);
        println!("    Description: {}", description);
        println!();
    }

    // Step 7: Handling disagreement
    println!("Step 7: Handling Disagreement");
    println!("-----------------------------");

    println!("  When models disagree (agreement < threshold):");
    println!("    1. Fallback to highest-weighted model");
    println!("    2. Request clarification from user");
    println!("    3. Use ensemble with uncertainty flag");
    println!("    4. Log disagreement for analysis");
    println!();

    // Step 8: Integration with MimicryEngine
    println!("Step 8: Integration with Mimicry");
    println!("--------------------------------");

    println!("  // In production code:");
    println!("  let mut engine = MimicryEngine::new();");
    println!("  engine.set_multi_model_observer(observer);");
    println!("  ");
    println!("  // Build persona from consensus of multiple models:");
    println!("  let consensus = engine.multi_model_observe(prompt).await?;");
    println!("  ");
    println!("  // The persona now reflects consensus behavior:");
    println!("  let persona = engine.current_persona();");
    println!("  // persona.signature blends patterns from all models");
    println!();

    // Step 9: Scheduling and rate limiting
    println!("Step 9: Model Scheduling");
    println!("------------------------");

    println!("  The MultiModelObserver includes intelligent scheduling:");
    println!("    - Rate limiting per model");
    println!("    - Retry with exponential backoff");
    println!("    - Failover to backup models");
    println!("    - Load balancing across endpoints");
    println!();

    println!("=== Example Complete ===");
}
