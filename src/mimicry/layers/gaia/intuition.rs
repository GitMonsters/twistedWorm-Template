//! GAIA Intuition Engine implementation.
//!
//! The core engine that orchestrates pattern matching, resonance, and
//! analogical reasoning to provide intuitive insights beyond explicit rules.

use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use super::analogical::{AnalogicalTransfer, TransferResult};
use super::pattern::{Pattern, PatternMatch, PatternMemory, PatternStats};
use super::resonance::{ResonanceConfig, ResonanceField, ResonanceResult};
use super::{GaiaError, GaiaResult};
use crate::mimicry::layers::layer::{Domain, Layer, LayerState};

/// Configuration for the GAIA Intuition Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaiaConfig {
    /// Minimum similarity threshold for pattern matching.
    pub min_similarity_threshold: f32,

    /// Maximum number of pattern matches to consider.
    pub max_pattern_matches: usize,

    /// Resonance field configuration.
    pub resonance_config: ResonanceConfig,

    /// Enable analogical transfer across domains.
    pub enable_analogical_transfer: bool,

    /// Weight for analogical insights in final result.
    pub analogical_weight: f32,

    /// Learning rate for weight updates.
    pub learning_rate: f32,

    /// Decay factor for old patterns (0.0 = no decay).
    pub pattern_decay: f32,

    /// Maximum patterns to store in memory.
    pub max_patterns: usize,

    /// Enable automatic pattern discovery.
    pub auto_discover_patterns: bool,
}

impl Default for GaiaConfig {
    fn default() -> Self {
        Self {
            min_similarity_threshold: 0.6,
            max_pattern_matches: 10,
            resonance_config: ResonanceConfig::default(),
            enable_analogical_transfer: true,
            analogical_weight: 0.3,
            learning_rate: 0.1,
            pattern_decay: 0.001,
            max_patterns: 10000,
            auto_discover_patterns: true,
        }
    }
}

impl GaiaConfig {
    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            max_pattern_matches: 5,
            max_patterns: 100,
            ..Default::default()
        }
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            max_pattern_matches: 20,
            max_patterns: 50000,
            enable_analogical_transfer: true,
            analogical_weight: 0.5,
            ..Default::default()
        }
    }
}

/// Result of an intuition query.
#[derive(Debug, Clone)]
pub struct IntuitionResult {
    /// Top pattern matches.
    pub matches: Vec<PatternMatch>,

    /// Resonance field activation result.
    pub resonance: ResonanceResult,

    /// Cross-domain insights from analogical transfer.
    pub analogical_insights: Vec<TransferResult>,

    /// Combined intuition confidence score.
    pub confidence: f32,

    /// Primary domain of the insight.
    pub primary_domain: Domain,

    /// Suggested action or interpretation.
    pub suggestion: Option<String>,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

impl IntuitionResult {
    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            matches: Vec::new(),
            resonance: ResonanceResult::empty(),
            analogical_insights: Vec::new(),
            confidence: 0.0,
            primary_domain: Domain::Emergent,
            suggestion: None,
            processing_time_ms: 0,
        }
    }

    /// Check if the result has useful content.
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty() && self.analogical_insights.is_empty()
    }

    /// Get the best match if any.
    pub fn best_match(&self) -> Option<&PatternMatch> {
        self.matches.first()
    }
}

/// The main GAIA Intuition Engine.
///
/// Orchestrates pattern matching, resonance activation, and analogical
/// reasoning to provide insights beyond explicit rules.
pub struct GaiaIntuitionEngine {
    /// Configuration.
    config: GaiaConfig,

    /// Pattern memory storage.
    pattern_memory: PatternMemory,

    /// Resonance field for activation spreading.
    resonance_field: ResonanceField,

    /// Analogical transfer engine.
    analogical_transfer: AnalogicalTransfer,

    /// Statistics and metrics.
    stats: RwLock<GaiaStats>,
}

/// Statistics for the GAIA engine.
#[derive(Debug, Clone, Default)]
pub struct GaiaStats {
    /// Total queries processed.
    pub total_queries: u64,
    /// Successful matches (confidence > threshold).
    pub successful_matches: u64,
    /// Total patterns discovered.
    pub patterns_discovered: u64,
    /// Total analogical transfers.
    pub analogical_transfers: u64,
    /// Average query time in ms.
    pub avg_query_time_ms: f64,
}

impl GaiaIntuitionEngine {
    /// Create a new GAIA engine with the given configuration.
    pub fn new(config: GaiaConfig) -> Self {
        Self {
            resonance_field: ResonanceField::new(config.resonance_config.clone()),
            analogical_transfer: AnalogicalTransfer::new(),
            config,
            pattern_memory: PatternMemory::new(),
            stats: RwLock::new(GaiaStats::default()),
        }
    }

    /// Create a GAIA engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GaiaConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &GaiaConfig {
        &self.config
    }

    /// Get a reference to the pattern memory.
    pub fn pattern_memory(&self) -> &PatternMemory {
        &self.pattern_memory
    }

    /// Get the resonance field.
    pub fn resonance_field(&self) -> &ResonanceField {
        &self.resonance_field
    }

    /// Register a new pattern.
    pub fn register_pattern(&self, pattern: Pattern) -> GaiaResult<()> {
        // Check capacity
        if self.pattern_memory.len() >= self.config.max_patterns {
            // Could implement LRU eviction here
            return Err(GaiaError::ConfigError(format!(
                "Pattern memory full ({} patterns)",
                self.config.max_patterns
            )));
        }

        self.pattern_memory.register(pattern)
    }

    /// Query the intuition engine with a feature vector.
    pub fn query(&self, features: &[f32]) -> GaiaResult<IntuitionResult> {
        let start = Self::current_time_ms();

        // Find matching patterns
        let matches = self.pattern_memory.find_matches(
            features,
            self.config.min_similarity_threshold,
            self.config.max_pattern_matches,
        );

        // Activate resonance field
        let resonance = if !matches.is_empty() {
            let pattern_ids: Vec<&str> = matches.iter().map(|m| m.pattern_id.as_str()).collect();
            self.resonance_field
                .activate(&pattern_ids, &self.pattern_memory)
        } else {
            ResonanceResult::empty()
        };

        // Perform analogical transfer if enabled
        let analogical_insights = if self.config.enable_analogical_transfer && !matches.is_empty() {
            self.perform_analogical_transfer(&matches, features)
        } else {
            Vec::new()
        };

        // Calculate combined confidence
        let confidence =
            self.calculate_combined_confidence(&matches, &resonance, &analogical_insights);

        // Determine primary domain
        let primary_domain = matches
            .first()
            .map(|m| m.domain)
            .unwrap_or(Domain::Emergent);

        // Generate suggestion
        let suggestion = self.generate_suggestion(&matches, &analogical_insights);

        let processing_time_ms = Self::current_time_ms() - start;

        // Update statistics
        self.update_stats(confidence, processing_time_ms);

        Ok(IntuitionResult {
            matches,
            resonance,
            analogical_insights,
            confidence,
            primary_domain,
            suggestion,
            processing_time_ms,
        })
    }

    /// Query with a LayerState input (for bridge integration).
    pub fn query_state(&self, state: &LayerState) -> GaiaResult<IntuitionResult> {
        // Try to extract features from the state
        if let Some(features) = state.data::<Vec<f32>>() {
            return self.query(features);
        }

        // Try to extract a single feature
        if let Some(&feature) = state.data::<f32>() {
            return self.query(&[feature]);
        }

        // Use confidence as a single-feature fallback
        self.query(&[state.confidence])
    }

    /// Provide feedback for a pattern (reinforcement learning).
    pub fn feedback(&self, pattern_id: &str, success: bool) -> GaiaResult<()> {
        if success {
            self.pattern_memory.record_success(pattern_id)?;
        } else {
            self.pattern_memory.record_failure(pattern_id)?;
        }
        Ok(())
    }

    /// Provide feedback with a reward signal.
    pub fn feedback_with_reward(&self, pattern_id: &str, reward: f32) -> GaiaResult<()> {
        let delta = reward * self.config.learning_rate;
        self.pattern_memory.adjust_weight(pattern_id, delta)
    }

    /// Get pattern statistics.
    pub fn pattern_stats(&self) -> PatternStats {
        self.pattern_memory.stats()
    }

    /// Get engine statistics.
    pub fn stats(&self) -> GaiaStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear all patterns and reset.
    pub fn reset(&self) {
        self.pattern_memory.clear();
        self.resonance_field.reset();
        *self.stats.write().unwrap() = GaiaStats::default();
    }

    /// Create a pattern from a LayerState.
    pub fn pattern_from_state(&self, id: &str, state: &LayerState) -> GaiaResult<Pattern> {
        let domain = match state.layer {
            Layer::BasePhysics | Layer::ExtendedPhysics => Domain::Physics,
            Layer::MultilingualProcessing => Domain::Language,
            Layer::GaiaConsciousness => Domain::Consciousness,
            Layer::CollaborativeLearning => Domain::Social,
            Layer::ExternalApis => Domain::External,
            Layer::CrossDomain => Domain::Emergent,
            Layer::PreCognitiveVisualization => Domain::Visualization,
        };

        let fingerprint = if let Some(features) = state.data::<Vec<f32>>() {
            features.clone()
        } else if let Some(&feature) = state.data::<f32>() {
            vec![feature]
        } else {
            vec![state.confidence]
        };

        Ok(Pattern::new(id, domain)
            .with_fingerprint(fingerprint)
            .with_weight(state.confidence))
    }

    /// Discover a new pattern from input (if auto-discovery is enabled).
    pub fn maybe_discover_pattern(
        &self,
        features: &[f32],
        domain: Domain,
    ) -> GaiaResult<Option<String>> {
        if !self.config.auto_discover_patterns {
            return Ok(None);
        }

        // Check if a similar pattern already exists
        let existing = self.pattern_memory.find_matches(features, 0.95, 1);
        if !existing.is_empty() {
            return Ok(None); // Already have a very similar pattern
        }

        // Create a new pattern
        let id = format!("auto_{:x}", Self::current_time_ms());
        let pattern = Pattern::new(&id, domain)
            .with_fingerprint(features.to_vec())
            .with_weight(0.5); // Start with low confidence

        self.register_pattern(pattern)?;

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.patterns_discovered += 1;
        }

        Ok(Some(id))
    }

    fn perform_analogical_transfer(
        &self,
        matches: &[PatternMatch],
        features: &[f32],
    ) -> Vec<TransferResult> {
        let mut results = Vec::new();

        for m in matches.iter().take(3) {
            // Only transfer from top 3 matches
            if let Some(pattern) = self.pattern_memory.get(&m.pattern_id) {
                // Try to transfer to different domains
                for &target_domain in &[
                    Domain::Physics,
                    Domain::Language,
                    Domain::Consciousness,
                    Domain::Social,
                ] {
                    if target_domain != pattern.domain() {
                        if let Ok(transfer) = self.analogical_transfer.transfer(
                            &pattern,
                            target_domain,
                            features,
                            &self.pattern_memory,
                        ) {
                            if transfer.strength.value() > 0.3 {
                                results.push(transfer);
                            }
                        }
                    }
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.analogical_transfers += results.len() as u64;
        }

        results
    }

    fn calculate_combined_confidence(
        &self,
        matches: &[PatternMatch],
        resonance: &ResonanceResult,
        analogical: &[TransferResult],
    ) -> f32 {
        if matches.is_empty() {
            return 0.0;
        }

        // Base confidence from best match
        let base_confidence = matches[0].weighted_similarity;

        // Resonance boost
        let resonance_boost = resonance.total_activation * 0.1;

        // Analogical boost
        let analogical_boost = if analogical.is_empty() {
            0.0
        } else {
            let avg_strength: f32 = analogical.iter().map(|a| a.strength.value()).sum::<f32>()
                / analogical.len() as f32;
            avg_strength * self.config.analogical_weight
        };

        // Combine multiplicatively for super-threshold confidence
        let combined = base_confidence * (1.0 + resonance_boost) * (1.0 + analogical_boost);

        // Can exceed 1.0 for high-confidence intuitions
        combined
    }

    fn generate_suggestion(
        &self,
        matches: &[PatternMatch],
        analogical: &[TransferResult],
    ) -> Option<String> {
        if matches.is_empty() {
            return None;
        }

        let best = &matches[0];

        // Simple suggestion based on best match
        let mut suggestion = format!(
            "Pattern '{}' matched with {:.1}% confidence",
            best.pattern_id,
            best.weighted_similarity * 100.0
        );

        if !analogical.is_empty() {
            let best_transfer = &analogical[0];
            suggestion.push_str(&format!(
                "; analogical insight from {:?} domain",
                best_transfer.target_domain
            ));
        }

        Some(suggestion)
    }

    fn update_stats(&self, confidence: f32, query_time_ms: u64) {
        let mut stats = self.stats.write().unwrap();
        stats.total_queries += 1;
        if confidence > self.config.min_similarity_threshold {
            stats.successful_matches += 1;
        }
        // Update running average
        let n = stats.total_queries as f64;
        stats.avg_query_time_ms =
            stats.avg_query_time_ms * ((n - 1.0) / n) + (query_time_ms as f64 / n);
    }

    fn current_time_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

impl Default for GaiaIntuitionEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaia_creation() {
        let gaia = GaiaIntuitionEngine::with_defaults();
        assert!(gaia.pattern_memory().is_empty());
    }

    #[test]
    fn test_pattern_registration() {
        let gaia = GaiaIntuitionEngine::with_defaults();
        let pattern = Pattern::new("test", Domain::Physics).with_fingerprint(vec![1.0, 0.0, 0.0]);

        gaia.register_pattern(pattern).unwrap();
        assert_eq!(gaia.pattern_memory().len(), 1);
    }

    #[test]
    fn test_basic_query() {
        let gaia = GaiaIntuitionEngine::with_defaults();

        gaia.register_pattern(
            Pattern::new("p1", Domain::Physics)
                .with_fingerprint(vec![1.0, 0.0, 0.0])
                .with_weight(1.0),
        )
        .unwrap();

        let result = gaia.query(&[1.0, 0.0, 0.0]).unwrap();
        assert!(!result.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_feedback_reinforcement() {
        let gaia = GaiaIntuitionEngine::with_defaults();

        gaia.register_pattern(
            Pattern::new("p1", Domain::Physics)
                .with_fingerprint(vec![1.0, 0.0, 0.0])
                .with_weight(1.0),
        )
        .unwrap();

        let initial_weight = gaia.pattern_memory().get("p1").unwrap().weight();

        gaia.feedback("p1", true).unwrap();
        let after_success = gaia.pattern_memory().get("p1").unwrap().weight();
        assert!(after_success > initial_weight);

        gaia.feedback("p1", false).unwrap();
        let after_failure = gaia.pattern_memory().get("p1").unwrap().weight();
        assert!(after_failure < after_success);
    }

    #[test]
    fn test_stats_update() {
        let gaia = GaiaIntuitionEngine::with_defaults();

        gaia.register_pattern(
            Pattern::new("p1", Domain::Physics).with_fingerprint(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();

        gaia.query(&[1.0, 0.0, 0.0]).unwrap();
        gaia.query(&[1.0, 0.0, 0.0]).unwrap();

        let stats = gaia.stats();
        assert_eq!(stats.total_queries, 2);
    }

    #[test]
    fn test_empty_query() {
        let gaia = GaiaIntuitionEngine::with_defaults();
        let result = gaia.query(&[1.0, 0.0, 0.0]).unwrap();

        assert!(result.is_empty());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_pattern_from_state() {
        let gaia = GaiaIntuitionEngine::with_defaults();
        let state = LayerState::with_confidence(Layer::BasePhysics, vec![0.5f32, 0.3, 0.2], 0.8);

        let pattern = gaia.pattern_from_state("from_state", &state).unwrap();
        assert_eq!(pattern.domain(), Domain::Physics);
        assert_eq!(pattern.fingerprint(), &[0.5, 0.3, 0.2]);
    }
}
