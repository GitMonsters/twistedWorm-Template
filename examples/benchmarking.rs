//! AgentToLeaP Benchmarking Example
//!
//! This example demonstrates how to use the benchmarking module to evaluate
//! RustyWorm personas against standard AI agent benchmarks.
//!
//! Requirements:
//! - Build with `--features agentdock`
//!
//! Run:
//! ```bash
//! cargo run --features agentdock --example benchmarking
//! ```

use rustyworm::mimicry::engine::MimicryEngine;

#[cfg(feature = "agentdock")]
use rustyworm::mimicry::benchmarking::{
    BenchmarkConfig, BenchmarkResult, BenchmarkRunner, BenchmarkSuite, MetricType, TaskResult,
};

fn main() {
    println!("=== RustyWorm AgentToLeaP Benchmarking Example ===\n");

    #[cfg(not(feature = "agentdock"))]
    {
        println!("This example requires the 'agentdock' feature.");
        println!("Run with: cargo run --features agentdock --example benchmarking");
        return;
    }

    #[cfg(feature = "agentdock")]
    run_benchmarking_demo();
}

#[cfg(feature = "agentdock")]
fn run_benchmarking_demo() {
    // Step 1: Available benchmarks
    println!("Step 1: Available Benchmark Suites");
    println!("----------------------------------");

    let benchmarks = vec![
        (
            BenchmarkSuite::GAIA,
            "General AI Assistants",
            "Multi-step reasoning and tool use",
        ),
        (
            BenchmarkSuite::HLE,
            "Human-Like Evaluation",
            "Natural language understanding",
        ),
        (
            BenchmarkSuite::BrowseComp,
            "Browse Comprehension",
            "Web content understanding",
        ),
        (
            BenchmarkSuite::Frames,
            "Frame Reasoning",
            "Frame-based knowledge representation",
        ),
        (
            BenchmarkSuite::AssistantBench,
            "Assistant Tasks",
            "Real-world assistant capabilities",
        ),
        (
            BenchmarkSuite::WEBARENA,
            "Web Arena",
            "Web interaction and navigation",
        ),
        (
            BenchmarkSuite::OWA,
            "Open World Agents",
            "Open-ended environment interaction",
        ),
        (
            BenchmarkSuite::SWEBench,
            "Software Engineering",
            "Code understanding and generation",
        ),
        (
            BenchmarkSuite::AppWorld,
            "Application World",
            "Application interaction tasks",
        ),
    ];

    for (suite, name, description) in &benchmarks {
        println!("  {:?}", suite);
        println!("    Name: {}", name);
        println!("    Focus: {}", description);
        println!();
    }

    // Step 2: Configure benchmark runner
    println!("Step 2: Configuring Benchmark Runner");
    println!("------------------------------------");

    let config = BenchmarkConfig {
        max_concurrent_tasks: 4,
        timeout_per_task_ms: 60000,
        retry_failed_tasks: true,
        max_retries: 3,
        save_detailed_results: true,
        output_dir: "./benchmark_results".to_string(),
        metrics: vec![
            MetricType::Accuracy,
            MetricType::SuccessRate,
            MetricType::AverageLatency,
            MetricType::TokenEfficiency,
            MetricType::ToolUseAccuracy,
        ],
        ..Default::default()
    };

    println!("  Max concurrent tasks: {}", config.max_concurrent_tasks);
    println!("  Timeout per task: {}ms", config.timeout_per_task_ms);
    println!("  Retry failed: {}", config.retry_failed_tasks);
    println!("  Output directory: {}", config.output_dir);
    println!();

    // Step 3: Create benchmark runner
    println!("Step 3: Creating Benchmark Runner");
    println!("---------------------------------");

    let runner = BenchmarkRunner::new(config);
    println!("  Runner created");
    println!();

    // Step 4: Simulate benchmark results
    println!("Step 4: Simulated Benchmark Results");
    println!("-----------------------------------");

    // Simulated results for demonstration
    let results = vec![
        create_sample_result(BenchmarkSuite::GAIA, 0.72, 0.85, 1250),
        create_sample_result(BenchmarkSuite::HLE, 0.68, 0.78, 980),
        create_sample_result(BenchmarkSuite::SWEBench, 0.45, 0.62, 2100),
        create_sample_result(BenchmarkSuite::AssistantBench, 0.81, 0.92, 850),
    ];

    println!("  Suite           | Accuracy | Success | Latency");
    println!("  ----------------|----------|---------|--------");
    for result in &results {
        println!(
            "  {:15} | {:>7.1}% | {:>6.1}% | {:>5}ms",
            format!("{:?}", result.suite),
            result.accuracy * 100.0,
            result.success_rate * 100.0,
            result.avg_latency_ms
        );
    }
    println!();

    // Step 5: Detailed task breakdown
    println!("Step 5: Task Breakdown (GAIA Example)");
    println!("-------------------------------------");

    let gaia_tasks = vec![
        TaskResult {
            task_id: "gaia_001".to_string(),
            success: true,
            latency_ms: 1100,
            tokens_used: 450,
            tool_calls: 2,
        },
        TaskResult {
            task_id: "gaia_002".to_string(),
            success: true,
            latency_ms: 1350,
            tokens_used: 520,
            tool_calls: 3,
        },
        TaskResult {
            task_id: "gaia_003".to_string(),
            success: false,
            latency_ms: 1800,
            tokens_used: 680,
            tool_calls: 4,
        },
        TaskResult {
            task_id: "gaia_004".to_string(),
            success: true,
            latency_ms: 950,
            tokens_used: 380,
            tool_calls: 1,
        },
        TaskResult {
            task_id: "gaia_005".to_string(),
            success: true,
            latency_ms: 1250,
            tokens_used: 490,
            tool_calls: 2,
        },
    ];

    println!("  Task ID   | Success | Latency | Tokens | Tools");
    println!("  ----------|---------|---------|--------|------");
    for task in &gaia_tasks {
        let status = if task.success { "✓" } else { "✗" };
        println!(
            "  {:9} | {:^7} | {:>5}ms | {:>6} | {:>5}",
            task.task_id, status, task.latency_ms, task.tokens_used, task.tool_calls
        );
    }
    println!();

    // Step 6: Comparison with baselines
    println!("Step 6: Baseline Comparison");
    println!("---------------------------");

    println!("  Model       | GAIA  | HLE   | SWE   | Assist");
    println!("  ------------|-------|-------|-------|-------");
    println!("  GPT-4o      | 75.2% | 82.1% | 48.3% | 88.5%");
    println!("  Claude 3    | 73.8% | 79.4% | 51.2% | 85.2%");
    println!("  Gemini Pro  | 68.5% | 74.3% | 42.1% | 82.1%");
    println!("  RustyWorm   | 72.0% | 68.0% | 45.0% | 81.0%  <- Current");
    println!();

    println!("  Improvement targets with RL:");
    println!("    GAIA:      72.0% -> 78%+ (goal)");
    println!("    HLE:       68.0% -> 80%+ (goal)");
    println!("    SWEBench:  45.0% -> 52%+ (goal)");
    println!();

    // Step 7: Generating reports
    println!("Step 7: Report Generation");
    println!("-------------------------");

    println!("  Available report formats:");
    println!("    - JSON: Full structured data");
    println!("    - Markdown: Human-readable summary");
    println!("    - CSV: Spreadsheet-compatible");
    println!("    - HTML: Interactive dashboard");
    println!();

    println!("  // Generate reports:");
    println!("  runner.export_json(\"results.json\")?;");
    println!("  runner.export_markdown(\"results.md\")?;");
    println!("  runner.export_html(\"dashboard.html\")?;");
    println!();

    // Step 8: Continuous benchmarking
    println!("Step 8: Continuous Benchmarking");
    println!("-------------------------------");

    println!("  Set up automated benchmarking:");
    println!("    1. Schedule periodic runs");
    println!("    2. Compare against previous results");
    println!("    3. Alert on regression");
    println!("    4. Track improvement over time");
    println!();

    println!("  // Example CI integration:");
    println!("  runner.set_baseline(\"v2.0.0_results.json\");");
    println!("  let regression = runner.check_regression(0.05)?; // 5% threshold");
    println!("  if regression.found {");
    println!("      panic!(\"Benchmark regression detected!\");");
    println!("  }");
    println!();

    // Step 9: Integration with evolution
    println!("Step 9: Benchmark-Driven Evolution");
    println!("----------------------------------");

    println!("  Benchmarks can drive evolution targets:");
    println!();
    println!("  // Use benchmark scores as evolution fitness:");
    println!("  let fitness = runner.run(BenchmarkSuite::GAIA).await?;");
    println!("  engine.set_fitness_function(|persona| {");
    println!("      // Optimize for benchmark performance");
    println!("      persona.benchmark_score(BenchmarkSuite::GAIA)");
    println!("  });");
    println!("  engine.evolve_with_rl(&optimizer, 100).await?;");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(feature = "agentdock")]
fn create_sample_result(
    suite: BenchmarkSuite,
    accuracy: f64,
    success_rate: f64,
    latency: u64,
) -> BenchmarkResult {
    BenchmarkResult {
        suite,
        accuracy,
        success_rate,
        avg_latency_ms: latency,
        total_tasks: 100,
        successful_tasks: (success_rate * 100.0) as u32,
        failed_tasks: ((1.0 - success_rate) * 100.0) as u32,
        total_tokens: 45000,
        total_tool_calls: 250,
    }
}
