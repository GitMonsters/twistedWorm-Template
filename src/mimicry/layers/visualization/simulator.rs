//! Task Simulator for Pre-Cognitive Visualization.
//!
//! Simulates task-space execution before the main forward pass,
//! building a structural model of the task and identifying
//! expected sub-goals, dependencies, and failure modes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mimicry::layers::layer::{Layer, LayerState};

/// Unique identifier for a simulated task.
pub type TaskId = String;

/// A sub-goal within a simulated task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGoal {
    /// Identifier for this sub-goal.
    pub id: String,
    /// Description of the sub-goal.
    pub description: String,
    /// Estimated confidence of achieving this sub-goal.
    pub estimated_confidence: f32,
    /// Dependencies on other sub-goals (by id).
    pub dependencies: Vec<String>,
    /// Whether this sub-goal is critical (failure blocks the whole task).
    pub critical: bool,
}

impl SubGoal {
    /// Create a new sub-goal.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            estimated_confidence: 0.5,
            dependencies: Vec::new(),
            critical: false,
        }
    }

    /// Set estimated confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.estimated_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add a dependency.
    pub fn with_dependency(mut self, dep_id: impl Into<String>) -> Self {
        self.dependencies.push(dep_id.into());
        self
    }

    /// Mark as critical.
    pub fn as_critical(mut self) -> Self {
        self.critical = true;
        self
    }
}

/// Result of a task simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Identifier for this simulation.
    pub task_id: TaskId,
    /// Sub-goals identified during simulation.
    pub sub_goals: Vec<SubGoal>,
    /// Overall estimated confidence for the task.
    pub overall_confidence: f32,
    /// Identified failure modes (description, probability).
    pub failure_modes: Vec<(String, f32)>,
    /// Pre-warm signals: layer -> confidence boost.
    pub pre_warm_signals: HashMap<Layer, f32>,
    /// Number of simulation steps taken.
    pub steps: u32,
    /// Whether the simulation converged on a stable outcome.
    pub converged: bool,
}

impl SimulationResult {
    /// Create an empty simulation result.
    pub fn empty(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            sub_goals: Vec::new(),
            overall_confidence: 0.0,
            failure_modes: Vec::new(),
            pre_warm_signals: HashMap::new(),
            steps: 0,
            converged: false,
        }
    }

    /// Get the critical path confidence (product of all critical sub-goal confidences).
    pub fn critical_path_confidence(&self) -> f32 {
        let critical: Vec<&SubGoal> = self.sub_goals.iter().filter(|g| g.critical).collect();
        if critical.is_empty() {
            return self.overall_confidence;
        }
        critical
            .iter()
            .map(|g| g.estimated_confidence)
            .product::<f32>()
    }

    /// Get the highest-risk failure mode.
    pub fn highest_risk(&self) -> Option<&(String, f32)> {
        self.failure_modes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Convert the simulation result into a pre-warm LayerState.
    pub fn to_pre_warm_state(&self) -> LayerState {
        let mut state = LayerState::with_confidence(
            Layer::PreCognitiveVisualization,
            self.task_id.clone(),
            self.overall_confidence,
        );
        state.set_metadata("simulation_type", "task_pre_warm");
        state.set_metadata("sub_goal_count", &self.sub_goals.len().to_string());
        state.set_metadata("converged", &self.converged.to_string());
        if let Some((mode, prob)) = self.highest_risk() {
            state.set_metadata("highest_risk", &format!("{}:{:.3}", mode, prob));
        }
        state
    }
}

/// Configuration for the task simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorConfig {
    /// Maximum simulation steps.
    pub max_steps: u32,
    /// Convergence threshold for confidence stability.
    pub convergence_threshold: f32,
    /// Default confidence for unknown sub-goals.
    pub default_confidence: f32,
    /// Whether to propagate failure mode analysis.
    pub analyze_failure_modes: bool,
    /// Maximum sub-goals to track per task.
    pub max_sub_goals: usize,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            max_steps: 20,
            convergence_threshold: 0.005,
            default_confidence: 0.5,
            analyze_failure_modes: true,
            max_sub_goals: 50,
        }
    }
}

/// The Task Simulator component of the Visualization Engine.
///
/// Runs a lightweight simulation of the task before the main forward pass,
/// decomposing it into sub-goals, estimating confidence, and generating
/// pre-warm signals for downstream layers.
pub struct TaskSimulator {
    /// Configuration.
    config: SimulatorConfig,
    /// History of simulation results.
    history: Vec<SimulationResult>,
    /// Counter for generating unique task IDs.
    task_counter: u64,
}

impl TaskSimulator {
    /// Create a new task simulator.
    pub fn new(config: SimulatorConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            task_counter: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SimulatorConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &SimulatorConfig {
        &self.config
    }

    /// Get simulation history.
    pub fn history(&self) -> &[SimulationResult] {
        &self.history
    }

    /// Run a simulation for a given input state.
    ///
    /// This decomposes the input into sub-goals, estimates per-goal
    /// confidence, identifies failure modes, and generates pre-warm signals.
    pub fn simulate(&mut self, input: &LayerState) -> SimulationResult {
        self.task_counter += 1;
        let task_id = format!("task_{}", self.task_counter);

        // Decompose input into sub-goals based on confidence and metadata
        let sub_goals = self.decompose_task(input);

        // Estimate overall confidence from sub-goals
        let overall_confidence = self.estimate_confidence(&sub_goals);

        // Identify failure modes
        let failure_modes = if self.config.analyze_failure_modes {
            self.analyze_failures(&sub_goals)
        } else {
            Vec::new()
        };

        // Generate pre-warm signals for connected layers
        let pre_warm_signals = self.generate_pre_warm(&sub_goals, overall_confidence);

        // Run iterative refinement
        let (refined_confidence, steps, converged) =
            self.refine_simulation(overall_confidence, &sub_goals);

        let result = SimulationResult {
            task_id,
            sub_goals,
            overall_confidence: refined_confidence,
            failure_modes,
            pre_warm_signals,
            steps,
            converged,
        };

        // Store in history (bounded)
        self.history.push(result.clone());
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        result
    }

    /// Decompose a task into sub-goals.
    fn decompose_task(&self, input: &LayerState) -> Vec<SubGoal> {
        let mut goals = Vec::new();
        let base_confidence = input.confidence;

        // Perception sub-goal (always present)
        goals.push(
            SubGoal::new("perception", "Input perception and encoding")
                .with_confidence((base_confidence * 1.1).min(1.0))
                .as_critical(),
        );

        // Reasoning sub-goal
        goals.push(
            SubGoal::new("reasoning", "Core reasoning and inference")
                .with_confidence(base_confidence * 0.9)
                .with_dependency("perception")
                .as_critical(),
        );

        // Integration sub-goal
        goals.push(
            SubGoal::new("integration", "Cross-layer integration")
                .with_confidence(base_confidence * 0.85)
                .with_dependency("reasoning"),
        );

        // Output sub-goal
        goals.push(
            SubGoal::new("output", "Output generation and formatting")
                .with_confidence(base_confidence * 0.95)
                .with_dependency("integration")
                .as_critical(),
        );

        // If there are upstream refs, add a context sub-goal
        if !input.upstream_refs.is_empty() {
            goals.push(
                SubGoal::new("context", "Context integration from upstream")
                    .with_confidence(base_confidence * 0.88)
                    .with_dependency("perception"),
            );
        }

        // Truncate to max
        goals.truncate(self.config.max_sub_goals);
        goals
    }

    /// Estimate overall confidence from sub-goals.
    fn estimate_confidence(&self, sub_goals: &[SubGoal]) -> f32 {
        if sub_goals.is_empty() {
            return self.config.default_confidence;
        }

        // Weighted geometric mean: critical goals weighted 2x
        let mut log_sum = 0.0f64;
        let mut total_weight = 0.0f64;

        for goal in sub_goals {
            let weight = if goal.critical { 2.0 } else { 1.0 };
            let conf = goal.estimated_confidence.max(0.001) as f64;
            log_sum += weight * conf.ln();
            total_weight += weight;
        }

        if total_weight > 0.0 {
            (log_sum / total_weight).exp() as f32
        } else {
            self.config.default_confidence
        }
    }

    /// Analyze potential failure modes.
    fn analyze_failures(&self, sub_goals: &[SubGoal]) -> Vec<(String, f32)> {
        let mut failures = Vec::new();

        for goal in sub_goals {
            let failure_prob = 1.0 - goal.estimated_confidence;
            if failure_prob > 0.1 {
                failures.push((format!("{} failure", goal.id), failure_prob));
            }
        }

        // Check for dependency chain failures
        let critical_chain: f32 = sub_goals
            .iter()
            .filter(|g| g.critical)
            .map(|g| g.estimated_confidence)
            .product();
        let chain_failure = 1.0 - critical_chain;
        if chain_failure > 0.15 {
            failures.push(("critical_chain_failure".to_string(), chain_failure));
        }

        failures.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        failures
    }

    /// Generate pre-warm signals for downstream layers.
    fn generate_pre_warm(
        &self,
        sub_goals: &[SubGoal],
        overall_confidence: f32,
    ) -> HashMap<Layer, f32> {
        let mut signals = HashMap::new();

        // L1 (BasePhysics) always gets a pre-warm from perception confidence
        if let Some(perception) = sub_goals.iter().find(|g| g.id == "perception") {
            signals.insert(Layer::BasePhysics, perception.estimated_confidence * 0.1);
        }

        // L4 (GaiaConsciousness) gets a pre-warm from reasoning confidence
        if let Some(reasoning) = sub_goals.iter().find(|g| g.id == "reasoning") {
            signals.insert(
                Layer::GaiaConsciousness,
                reasoning.estimated_confidence * 0.08,
            );
        }

        // L6 (CollaborativeLearning) gets a pre-warm if overall confidence is high
        if overall_confidence > 0.7 {
            signals.insert(Layer::CollaborativeLearning, overall_confidence * 0.05);
        }

        // L7 (ExternalApis) gets a pre-warm if any failure risk is high
        let max_risk = sub_goals
            .iter()
            .map(|g| 1.0 - g.estimated_confidence)
            .fold(0.0f32, f32::max);
        if max_risk > 0.3 {
            signals.insert(Layer::ExternalApis, max_risk * 0.1);
        }

        signals
    }

    /// Iteratively refine the simulation to convergence.
    fn refine_simulation(
        &self,
        initial_confidence: f32,
        sub_goals: &[SubGoal],
    ) -> (f32, u32, bool) {
        let mut confidence = initial_confidence;
        let mut converged = false;

        for step in 0..self.config.max_steps {
            // Damped refinement: adjust confidence toward critical path
            let critical_conf = sub_goals
                .iter()
                .filter(|g| g.critical)
                .map(|g| g.estimated_confidence)
                .product::<f32>();

            let new_confidence = confidence * 0.7 + critical_conf * 0.3;

            if (new_confidence - confidence).abs() < self.config.convergence_threshold {
                converged = true;
                return (new_confidence, step + 1, converged);
            }

            confidence = new_confidence;
        }

        (confidence, self.config.max_steps, converged)
    }
}

impl Default for TaskSimulator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_goal_creation() {
        let goal = SubGoal::new("test", "A test goal")
            .with_confidence(0.8)
            .with_dependency("dep1")
            .as_critical();

        assert_eq!(goal.id, "test");
        assert_eq!(goal.estimated_confidence, 0.8);
        assert_eq!(goal.dependencies, vec!["dep1"]);
        assert!(goal.critical);
    }

    #[test]
    fn test_simulator_creation() {
        let sim = TaskSimulator::with_defaults();
        assert_eq!(sim.config().max_steps, 20);
        assert!(sim.history().is_empty());
    }

    #[test]
    fn test_basic_simulation() {
        let mut sim = TaskSimulator::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test task".to_string(), 0.8);

        let result = sim.simulate(&input);

        assert!(!result.task_id.is_empty());
        assert!(!result.sub_goals.is_empty());
        assert!(result.overall_confidence > 0.0);
        assert!(result.overall_confidence <= 1.0);
        assert!(result.steps > 0);
    }

    #[test]
    fn test_critical_path_confidence() {
        let result = SimulationResult {
            task_id: "test".to_string(),
            sub_goals: vec![
                SubGoal::new("a", "A").with_confidence(0.9).as_critical(),
                SubGoal::new("b", "B").with_confidence(0.8).as_critical(),
                SubGoal::new("c", "C").with_confidence(0.5), // not critical
            ],
            overall_confidence: 0.7,
            failure_modes: Vec::new(),
            pre_warm_signals: HashMap::new(),
            steps: 1,
            converged: true,
        };

        let critical = result.critical_path_confidence();
        assert!((critical - 0.72).abs() < 0.001); // 0.9 * 0.8
    }

    #[test]
    fn test_pre_warm_signals() {
        let mut sim = TaskSimulator::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "test".to_string(), 0.9);

        let result = sim.simulate(&input);

        // Should have pre-warm for L1 at minimum
        assert!(result.pre_warm_signals.contains_key(&Layer::BasePhysics));
    }

    #[test]
    fn test_simulation_history() {
        let mut sim = TaskSimulator::with_defaults();

        for i in 0..3 {
            let input = LayerState::with_confidence(Layer::BasePhysics, format!("task_{}", i), 0.8);
            sim.simulate(&input);
        }

        assert_eq!(sim.history().len(), 3);
    }

    #[test]
    fn test_to_pre_warm_state() {
        let result = SimulationResult {
            task_id: "test_task".to_string(),
            sub_goals: vec![SubGoal::new("a", "A").with_confidence(0.9)],
            overall_confidence: 0.85,
            failure_modes: vec![("risk".to_string(), 0.3)],
            pre_warm_signals: HashMap::new(),
            steps: 5,
            converged: true,
        };

        let state = result.to_pre_warm_state();
        assert_eq!(state.layer, Layer::PreCognitiveVisualization);
        assert_eq!(state.confidence, 0.85);
        assert_eq!(state.get_metadata("simulation_type"), Some("task_pre_warm"));
        assert_eq!(state.get_metadata("converged"), Some("true"));
    }

    #[test]
    fn test_failure_analysis() {
        let mut sim = TaskSimulator::with_defaults();
        // Low-confidence input should trigger failure mode detection
        let input = LayerState::with_confidence(Layer::BasePhysics, "risky".to_string(), 0.3);

        let result = sim.simulate(&input);

        // Low confidence should produce failure modes
        assert!(!result.failure_modes.is_empty());
    }

    #[test]
    fn test_convergence() {
        let mut sim = TaskSimulator::with_defaults();
        let input = LayerState::with_confidence(Layer::BasePhysics, "stable".to_string(), 0.8);

        let result = sim.simulate(&input);

        // With reasonable confidence, should converge
        assert!(result.converged);
        assert!(result.steps < sim.config().max_steps);
    }
}
