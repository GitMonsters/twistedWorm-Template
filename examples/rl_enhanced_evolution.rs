//! RL-Enhanced Evolution Example
//!
//! This example demonstrates how to use the reinforcement learning optimizer
//! to improve persona convergence beyond traditional evolution methods.
//!
//! Requirements:
//! - Build with `--features rl`
//! - AgentRL service running at http://localhost:8080
//! - MongoDB running at mongodb://localhost:27017
//!
//! Run:
//! ```bash
//! cargo run --features rl --example rl_enhanced_evolution
//! ```

use rustyworm::mimicry::{engine::MimicryEngine, profile::AiProfile};

#[cfg(feature = "rl")]
use rustyworm::mimicry::{
    rl_config::{RLAlgorithm, RLConfig, RewardShaping},
    rl_optimizer::{RLOptimizationRequest, ReinforcementLearningOptimizer, TrajectoryStep},
};

fn main() {
    println!("=== RustyWorm RL-Enhanced Evolution Example ===\n");

    #[cfg(not(feature = "rl"))]
    {
        println!("This example requires the 'rl' feature.");
        println!("Run with: cargo run --features rl --example rl_enhanced_evolution");
        return;
    }

    #[cfg(feature = "rl")]
    run_rl_evolution();
}

#[cfg(feature = "rl")]
fn run_rl_evolution() {
    // Step 1: Configure the RL optimizer
    println!("Step 1: Configuring RL Optimizer");
    println!("---------------------------------");

    let config = RLConfig {
        algorithm: RLAlgorithm::PPO,
        learning_rate: 0.0003,
        gamma: 0.99,
        epsilon: 0.2,
        batch_size: 64,
        reward_shaping: RewardShaping {
            convergence_weight: 1.0,
            stability_weight: 0.3,
            efficiency_weight: 0.2,
            ..Default::default()
        },
        service_url: "http://localhost:8080".to_string(),
        ..Default::default()
    };

    println!("  Algorithm: {:?}", config.algorithm);
    println!("  Learning Rate: {}", config.learning_rate);
    println!("  Gamma: {}", config.gamma);
    println!("  Service URL: {}", config.service_url);
    println!();

    // Step 2: Create the optimizer
    println!("Step 2: Creating RL Optimizer");
    println!("-----------------------------");

    let optimizer = ReinforcementLearningOptimizer::new(config);
    println!(
        "  Optimizer created with session ID: {}",
        optimizer.session_id()
    );
    println!();

    // Step 3: Create a mimicry engine
    println!("Step 3: Setting Up Mimicry Engine");
    println!("---------------------------------");

    let mut engine = MimicryEngine::new();
    let claude_profile = AiProfile::claude();

    // Observe some Claude-like behavior
    let observations = vec![
        "I appreciate you sharing that with me. Let me think through this carefully.",
        "That's an interesting question. I want to be precise in my response.",
        "I should note that there are multiple perspectives to consider here.",
        "Let me break this down step by step to ensure clarity.",
    ];

    for obs in &observations {
        engine.observe(obs);
    }
    println!("  Observed {} Claude-like responses", observations.len());
    println!();

    // Step 4: Define the state for RL optimization
    println!("Step 4: Creating Optimization Request");
    println!("-------------------------------------");

    let state = vec![
        0.7, // Current convergence score
        0.5, // Stability metric
        0.3, // Evolution velocity
        0.6, // Profile similarity
        0.4, // Template match rate
    ];

    let request = RLOptimizationRequest {
        session_id: optimizer.session_id().to_string(),
        state: state.clone(),
        available_actions: vec![
            "increase_learning_rate".to_string(),
            "decrease_learning_rate".to_string(),
            "adjust_momentum".to_string(),
            "increase_exploration".to_string(),
            "decrease_exploration".to_string(),
        ],
        context: Some(serde_json::json!({
            "target_model": "claude",
            "current_epoch": 50,
            "convergence_trend": "improving",
        })),
    };

    println!("  State vector: {:?}", state);
    println!("  Available actions: {}", request.available_actions.len());
    println!();

    // Step 5: Simulate a trajectory
    println!("Step 5: Building Learning Trajectory");
    println!("------------------------------------");

    let trajectory: Vec<TrajectoryStep> = (0..10)
        .map(|i| {
            let convergence = 0.7 + (i as f64 * 0.02); // Improving convergence
            TrajectoryStep {
                step: i,
                state: vec![convergence, 0.5 + (i as f64 * 0.01), 0.3, 0.6, 0.4],
                action: format!("action_{}", i % 5),
                reward: convergence - 0.7, // Reward based on improvement
                next_state: vec![
                    convergence + 0.02,
                    0.5 + ((i + 1) as f64 * 0.01),
                    0.3,
                    0.6,
                    0.4,
                ],
                done: i == 9,
                info: Some(serde_json::json!({
                    "epoch": i * 10,
                    "loss": 0.1 - (i as f64 * 0.005),
                })),
            }
        })
        .collect();

    println!("  Generated {} trajectory steps", trajectory.len());
    println!("  Initial convergence: {:.2}", trajectory[0].state[0]);
    println!("  Final convergence: {:.2}", trajectory[9].state[0]);
    println!(
        "  Total reward: {:.2}",
        trajectory.iter().map(|t| t.reward).sum::<f64>()
    );
    println!();

    // Step 6: Demonstrate evolution with RL
    println!("Step 6: Evolution Loop with RL Guidance");
    println!("---------------------------------------");

    println!("  [Simulation - requires running AgentRL service]");
    println!();
    println!("  In production, you would run:");
    println!("    1. Start AgentRL: docker-compose up -d");
    println!("    2. Call: engine.evolve_with_rl(&optimizer, 100).await");
    println!("    3. Monitor convergence via /rl status");
    println!();

    // Step 7: Show expected results
    println!("Step 7: Expected Results");
    println!("------------------------");
    println!("  Without RL: ~66.7% convergence (current baseline)");
    println!("  With RL:    ~90%+ convergence (target improvement)");
    println!();
    println!("  Key benefits of RL-enhanced evolution:");
    println!("    - Adaptive learning rate adjustment");
    println!("    - Exploration/exploitation balance");
    println!("    - Long-term reward optimization");
    println!("    - Trajectory-based credit assignment");
    println!();

    println!("=== Example Complete ===");
}
