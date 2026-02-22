//! Layer registry for dynamic layer management.
//!
//! Provides registration, discovery, and lifecycle management for layers.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::layer::{Layer, LayerConfig, LayerState};

/// Callback type for layer state changes.
pub type LayerCallback = Box<dyn Fn(&LayerState) + Send + Sync>;

/// Layer handler that processes states for a specific layer.
pub trait LayerHandler: Send + Sync {
    /// Returns the layer this handler is for.
    fn layer(&self) -> Layer;

    /// Process an incoming state and produce an output state.
    fn process(&self, input: LayerState) -> Result<LayerState, LayerProcessError>;

    /// Initialize the handler with configuration.
    fn initialize(&mut self, config: &LayerConfig) -> Result<(), LayerProcessError>;

    /// Shutdown the handler gracefully.
    fn shutdown(&mut self) -> Result<(), LayerProcessError>;

    /// Check if the handler is ready to process.
    fn is_ready(&self) -> bool {
        true
    }
}

/// Error during layer processing.
#[derive(Debug, Clone)]
pub enum LayerProcessError {
    NotReady(String),
    ProcessingFailed(String),
    InvalidInput(String),
    ConfigurationError(String),
    ShutdownError(String),
}

impl std::fmt::Display for LayerProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerProcessError::NotReady(msg) => write!(f, "Layer not ready: {}", msg),
            LayerProcessError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
            LayerProcessError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            LayerProcessError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            LayerProcessError::ShutdownError(msg) => write!(f, "Shutdown error: {}", msg),
        }
    }
}

impl std::error::Error for LayerProcessError {}

/// Registration entry for a layer.
struct LayerRegistration {
    config: LayerConfig,
    handler: Option<Arc<RwLock<dyn LayerHandler>>>,
    callbacks: Vec<LayerCallback>,
    active: bool,
}

impl LayerRegistration {
    fn new(config: LayerConfig) -> Self {
        let active = config.enabled;
        Self {
            config,
            handler: None,
            callbacks: Vec::new(),
            active,
        }
    }
}

/// Registry for managing layer registrations and handlers.
pub struct LayerRegistry {
    registrations: HashMap<Layer, LayerRegistration>,
    default_config: LayerConfig,
}

impl LayerRegistry {
    /// Create a new registry with default configurations for all layers.
    pub fn new() -> Self {
        let mut registrations = HashMap::new();

        // Register all layers with default configs
        for &layer in Layer::all() {
            registrations.insert(layer, LayerRegistration::new(LayerConfig::new(layer)));
        }

        Self {
            registrations,
            default_config: LayerConfig::default(),
        }
    }

    /// Create a registry with custom default configuration.
    pub fn with_default_config(default_config: LayerConfig) -> Self {
        let mut registry = Self::new();
        registry.default_config = default_config;
        registry
    }

    /// Register a handler for a specific layer.
    pub fn register_handler(
        &mut self,
        handler: Arc<RwLock<dyn LayerHandler>>,
    ) -> Result<(), LayerProcessError> {
        let layer = {
            let guard = handler.read().map_err(|e| {
                LayerProcessError::ConfigurationError(format!("Failed to read handler: {}", e))
            })?;
            guard.layer()
        };

        if let Some(registration) = self.registrations.get_mut(&layer) {
            // Initialize the handler with the layer's config
            {
                let mut guard = handler.write().map_err(|e| {
                    LayerProcessError::ConfigurationError(format!("Failed to write handler: {}", e))
                })?;
                guard.initialize(&registration.config)?;
            }
            registration.handler = Some(handler);
            Ok(())
        } else {
            Err(LayerProcessError::ConfigurationError(format!(
                "Layer {:?} not found in registry",
                layer
            )))
        }
    }

    /// Update configuration for a layer.
    pub fn configure(&mut self, layer: Layer, config: LayerConfig) {
        if let Some(registration) = self.registrations.get_mut(&layer) {
            registration.config = config;
            registration.active = registration.config.enabled;
        }
    }

    /// Get configuration for a layer.
    pub fn get_config(&self, layer: Layer) -> Option<&LayerConfig> {
        self.registrations.get(&layer).map(|r| &r.config)
    }

    /// Enable a layer.
    pub fn enable(&mut self, layer: Layer) {
        if let Some(registration) = self.registrations.get_mut(&layer) {
            registration.config.enabled = true;
            registration.active = true;
        }
    }

    /// Disable a layer.
    pub fn disable(&mut self, layer: Layer) {
        if let Some(registration) = self.registrations.get_mut(&layer) {
            registration.config.enabled = false;
            registration.active = false;
        }
    }

    /// Check if a layer is enabled.
    pub fn is_enabled(&self, layer: Layer) -> bool {
        self.registrations
            .get(&layer)
            .map(|r| r.active)
            .unwrap_or(false)
    }

    /// Check if a layer has a handler registered.
    pub fn has_handler(&self, layer: Layer) -> bool {
        self.registrations
            .get(&layer)
            .and_then(|r| r.handler.as_ref())
            .is_some()
    }

    /// Get the handler for a layer.
    pub fn get_handler(&self, layer: Layer) -> Option<Arc<RwLock<dyn LayerHandler>>> {
        self.registrations
            .get(&layer)
            .and_then(|r| r.handler.clone())
    }

    /// Register a callback for layer state changes.
    pub fn on_state_change(&mut self, layer: Layer, callback: LayerCallback) {
        if let Some(registration) = self.registrations.get_mut(&layer) {
            registration.callbacks.push(callback);
        }
    }

    /// Notify callbacks of a state change.
    pub fn notify_state_change(&self, state: &LayerState) {
        if let Some(registration) = self.registrations.get(&state.layer) {
            for callback in &registration.callbacks {
                callback(state);
            }
        }
    }

    /// Process a state through the registered handler.
    pub fn process(&self, state: LayerState) -> Result<LayerState, LayerProcessError> {
        let layer = state.layer;

        let registration = self.registrations.get(&layer).ok_or_else(|| {
            LayerProcessError::NotReady(format!("Layer {:?} not registered", layer))
        })?;

        if !registration.active {
            return Err(LayerProcessError::NotReady(format!(
                "Layer {:?} is disabled",
                layer
            )));
        }

        let handler = registration.handler.as_ref().ok_or_else(|| {
            LayerProcessError::NotReady(format!("No handler for layer {:?}", layer))
        })?;

        let guard = handler.read().map_err(|e| {
            LayerProcessError::ProcessingFailed(format!("Failed to acquire handler lock: {}", e))
        })?;

        if !guard.is_ready() {
            return Err(LayerProcessError::NotReady(format!(
                "Handler for layer {:?} is not ready",
                layer
            )));
        }

        let result = guard.process(state)?;

        // Notify callbacks
        self.notify_state_change(&result);

        Ok(result)
    }

    /// Get all enabled layers in order.
    pub fn enabled_layers(&self) -> Vec<Layer> {
        Layer::all()
            .iter()
            .filter(|&&l| self.is_enabled(l))
            .copied()
            .collect()
    }

    /// Get all layers with registered handlers.
    pub fn layers_with_handlers(&self) -> Vec<Layer> {
        Layer::all()
            .iter()
            .filter(|&&l| self.has_handler(l))
            .copied()
            .collect()
    }

    /// Shutdown all handlers gracefully.
    pub fn shutdown_all(&mut self) -> Result<(), Vec<(Layer, LayerProcessError)>> {
        let mut errors = Vec::new();

        for (&layer, registration) in self.registrations.iter_mut() {
            if let Some(handler) = &registration.handler {
                if let Ok(mut guard) = handler.write() {
                    if let Err(e) = guard.shutdown() {
                        errors.push((layer, e));
                    }
                }
            }
            registration.active = false;
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get statistics about the registry.
    pub fn stats(&self) -> RegistryStats {
        let total = self.registrations.len();
        let enabled = self.registrations.values().filter(|r| r.active).count();
        let with_handlers = self
            .registrations
            .values()
            .filter(|r| r.handler.is_some())
            .count();
        let with_callbacks = self
            .registrations
            .values()
            .filter(|r| !r.callbacks.is_empty())
            .count();

        RegistryStats {
            total_layers: total,
            enabled_layers: enabled,
            layers_with_handlers: with_handlers,
            layers_with_callbacks: with_callbacks,
        }
    }
}

impl Default for LayerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LayerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerRegistry")
            .field("stats", &self.stats())
            .finish()
    }
}

/// Statistics about the registry.
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_layers: usize,
    pub enabled_layers: usize,
    pub layers_with_handlers: usize,
    pub layers_with_callbacks: usize,
}

/// A simple passthrough handler for testing.
pub struct PassthroughHandler {
    layer: Layer,
    ready: bool,
}

impl PassthroughHandler {
    pub fn new(layer: Layer) -> Self {
        Self {
            layer,
            ready: false,
        }
    }
}

impl LayerHandler for PassthroughHandler {
    fn layer(&self) -> Layer {
        self.layer
    }

    fn process(&self, input: LayerState) -> Result<LayerState, LayerProcessError> {
        Ok(input)
    }

    fn initialize(&mut self, _config: &LayerConfig) -> Result<(), LayerProcessError> {
        self.ready = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), LayerProcessError> {
        self.ready = false;
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = LayerRegistry::new();
        let stats = registry.stats();
        assert_eq!(stats.total_layers, 8);
        assert_eq!(stats.enabled_layers, 8); // All enabled by default
        assert_eq!(stats.layers_with_handlers, 0);
    }

    #[test]
    fn test_layer_enable_disable() {
        let mut registry = LayerRegistry::new();

        assert!(registry.is_enabled(Layer::GaiaConsciousness));
        registry.disable(Layer::GaiaConsciousness);
        assert!(!registry.is_enabled(Layer::GaiaConsciousness));
        registry.enable(Layer::GaiaConsciousness);
        assert!(registry.is_enabled(Layer::GaiaConsciousness));
    }

    #[test]
    fn test_handler_registration() {
        let mut registry = LayerRegistry::new();
        let handler = Arc::new(RwLock::new(PassthroughHandler::new(Layer::BasePhysics)));

        assert!(!registry.has_handler(Layer::BasePhysics));
        registry.register_handler(handler).unwrap();
        assert!(registry.has_handler(Layer::BasePhysics));
    }

    #[test]
    fn test_process_with_handler() {
        let mut registry = LayerRegistry::new();
        let handler = Arc::new(RwLock::new(PassthroughHandler::new(Layer::CrossDomain)));
        registry.register_handler(handler).unwrap();

        let input = LayerState::new(Layer::CrossDomain, "test".to_string());
        let output = registry.process(input).unwrap();
        assert_eq!(output.layer, Layer::CrossDomain);
    }

    #[test]
    fn test_process_without_handler() {
        let registry = LayerRegistry::new();
        let input = LayerState::new(Layer::ExternalApis, "test".to_string());
        let result = registry.process(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_enabled_layers() {
        let mut registry = LayerRegistry::new();
        registry.disable(Layer::ExternalApis);
        registry.disable(Layer::CollaborativeLearning);

        let enabled = registry.enabled_layers();
        assert_eq!(enabled.len(), 6);
        assert!(!enabled.contains(&Layer::ExternalApis));
        assert!(!enabled.contains(&Layer::CollaborativeLearning));
    }

    #[test]
    fn test_configuration() {
        let mut registry = LayerRegistry::new();

        let config = LayerConfig::new(Layer::GaiaConsciousness)
            .with_amplification_factor(1.5)
            .with_max_iterations(20);

        registry.configure(Layer::GaiaConsciousness, config);

        let retrieved = registry.get_config(Layer::GaiaConsciousness).unwrap();
        assert_eq!(retrieved.amplification_factor, 1.5);
        assert_eq!(retrieved.max_amplification_iterations, 20);
    }
}
