//! End-to-End Integration Tests for GUI Agent
//!
//! These tests verify the complete GUI agent pipeline from device connection
//! to action execution. Most tests are skipped by default unless real devices
//! or emulators are available.
//!
//! Run with:
//!   cargo test --features gui --test gui_e2e -- --ignored
//!
//! Environment variables:
//!   GUI_TEST_DEVICE_ID - Device ID to use (e.g., "emulator-5554")
//!   GUI_TEST_ENDPOINT - AgentCPM-GUI endpoint (default: http://localhost:8000)
//!   GUI_RUN_E2E - Set to "1" to run E2E tests

#![cfg(feature = "gui")]

use rustyworm::mimicry::{
    gui_agent::{GuiAction, GuiError, NormalizedPoint, Platform, SwipeDirection, TaskStatus},
    gui_bridge::{GuiBridge, GuiBridgeConfig, GuiConversation, Language},
    platforms::{DeviceManager, PlatformFactory},
};
use std::env;

/// Check if E2E tests should run
fn should_run_e2e() -> bool {
    env::var("GUI_RUN_E2E").map(|v| v == "1").unwrap_or(false)
}

/// Get test device ID from environment
fn get_test_device_id() -> Option<String> {
    env::var("GUI_TEST_DEVICE_ID").ok()
}

/// Get AgentCPM-GUI endpoint from environment
fn get_gui_endpoint() -> String {
    env::var("GUI_TEST_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:8000/v1/chat/completions".to_string())
}

// ============================================================================
// Unit Tests (always run)
// ============================================================================

mod unit_tests {
    use super::*;

    #[test]
    fn test_gui_action_creation() {
        let click = GuiAction::click(500, 300);
        assert_eq!(click.action_type(), "click");
        assert!(click.is_primitive());

        let long_press = GuiAction::long_press(500, 300, 1000);
        assert_eq!(long_press.action_type(), "long_press");
        assert_eq!(long_press.duration, Some(1000));

        let swipe = GuiAction::swipe_direction(500, 500, SwipeDirection::Down);
        assert_eq!(swipe.action_type(), "swipe");

        let type_text = GuiAction::type_text("Hello");
        assert_eq!(type_text.action_type(), "type");
        assert_eq!(type_text.type_text.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_normalized_coordinates() {
        // Test normalization
        let point = NormalizedPoint::from_absolute(540, 960, 1080, 1920);
        assert_eq!(point.x, 500);
        assert_eq!(point.y, 500);

        // Test denormalization
        let (x, y) = point.to_absolute(1080, 1920);
        assert_eq!(x, 540);
        assert_eq!(y, 960);

        // Test clamping
        let clamped = NormalizedPoint::new(1500, 2000);
        assert_eq!(clamped.x, 1000);
        assert_eq!(clamped.y, 1000);
    }

    #[test]
    fn test_action_serialization() {
        let action =
            GuiAction::click_with_thought(500, 300, "Test click").with_status(TaskStatus::Continue);

        let json = action.to_compact_json();
        assert!(json.contains("\"POINT\""));
        assert!(json.contains("\"thought\""));
        assert!(json.contains("Test click"));

        // Deserialize
        let parsed: GuiAction = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.point.unwrap().x, 500);
        assert_eq!(parsed.thought.as_deref(), Some("Test click"));
    }

    #[test]
    fn test_gui_bridge_config() {
        let config = GuiBridgeConfig::default();
        assert_eq!(config.model_name, "AgentCPM-GUI");
        assert_eq!(config.temperature, 0.1);
        assert!(config.enable_thought);
    }

    #[test]
    fn test_gui_conversation() {
        let mut conv = GuiConversation::new(5);

        conv.add_turn("Click login", GuiAction::click(500, 300));
        conv.add_turn("Type username", GuiAction::type_text("user"));

        assert_eq!(conv.len(), 2);
        assert!(!conv.is_empty());

        let context = conv.get_context();
        assert!(context.contains("Turn 1"));
        assert!(context.contains("Click login"));
    }

    #[test]
    fn test_platform_factory() {
        let android = PlatformFactory::create(Platform::Android);
        assert_eq!(android.platform(), Platform::Android);

        let ios = PlatformFactory::create(Platform::Ios);
        assert_eq!(ios.platform(), Platform::Ios);

        let macos = PlatformFactory::create(Platform::MacOs);
        assert_eq!(macos.platform(), Platform::MacOs);
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert!(!manager.is_connected());
        assert!(manager.active_device_id().is_none());
    }

    #[test]
    fn test_swipe_direction_calculations() {
        let start = NormalizedPoint::new(500, 500);

        let up = SwipeDirection::Up.end_point(&start, 300);
        assert_eq!(up.x, 500);
        assert_eq!(up.y, 200);

        let down = SwipeDirection::Down.end_point(&start, 300);
        assert_eq!(down.y, 800);

        let left = SwipeDirection::Left.end_point(&start, 300);
        assert_eq!(left.x, 200);

        let right = SwipeDirection::Right.end_point(&start, 300);
        assert_eq!(right.x, 800);
    }
}

// ============================================================================
// Integration Tests (require device/emulator)
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1 and a connected device"]
    fn test_device_list() {
        if !should_run_e2e() {
            return;
        }

        let manager = DeviceManager::new();
        let devices = manager.list_devices();

        println!("Found {} devices:", devices.len());
        for device in &devices {
            println!("  {} ({:?})", device.name, device.platform);
        }

        // Should at least find local desktop
        assert!(!devices.is_empty());
    }

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1 and GUI_TEST_DEVICE_ID"]
    fn test_device_connection() {
        if !should_run_e2e() {
            return;
        }

        let device_id = match get_test_device_id() {
            Some(id) => id,
            None => {
                println!("Skipping: GUI_TEST_DEVICE_ID not set");
                return;
            }
        };

        let mut manager = DeviceManager::new();

        // Connect
        match manager.connect(&device_id) {
            Ok(()) => {
                assert!(manager.is_connected());
                assert_eq!(manager.active_device_id(), Some(device_id.as_str()));

                // Disconnect
                manager.disconnect().expect("Disconnect failed");
                assert!(!manager.is_connected());
            }
            Err(e) => {
                println!(
                    "Connection failed (expected if device not available): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1 and GUI_TEST_DEVICE_ID"]
    fn test_screenshot_capture() {
        if !should_run_e2e() {
            return;
        }

        let device_id = match get_test_device_id() {
            Some(id) => id,
            None => return,
        };

        let mut manager = DeviceManager::new();

        if manager.connect(&device_id).is_err() {
            return;
        }

        match manager.capture_screenshot() {
            Ok(screenshot) => {
                println!("Screenshot captured:");
                println!("  Size: {} bytes", screenshot.data.len());
                println!("  Dimensions: {}x{}", screenshot.width, screenshot.height);
                println!("  Format: {:?}", screenshot.format);
                println!("  Platform: {:?}", screenshot.platform);

                assert!(screenshot.data.len() > 0);
                assert!(screenshot.width > 0);
                assert!(screenshot.height > 0);
            }
            Err(e) => {
                println!("Screenshot failed: {}", e);
            }
        }

        manager.disconnect().ok();
    }

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1 and GUI_TEST_DEVICE_ID"]
    fn test_action_execution() {
        if !should_run_e2e() {
            return;
        }

        let device_id = match get_test_device_id() {
            Some(id) => id,
            None => return,
        };

        let mut manager = DeviceManager::new();

        if manager.connect(&device_id).is_err() {
            return;
        }

        // Test a simple click (center of screen)
        let click = GuiAction::click(500, 500);
        match manager.execute_action(&click) {
            Ok(()) => println!("Click executed successfully"),
            Err(e) => println!("Click failed: {}", e),
        }

        // Test a swipe
        let swipe = GuiAction::swipe_direction(500, 700, SwipeDirection::Up);
        match manager.execute_action(&swipe) {
            Ok(()) => println!("Swipe executed successfully"),
            Err(e) => println!("Swipe failed: {}", e),
        }

        manager.disconnect().ok();
    }
}

// ============================================================================
// AgentCPM-GUI Integration Tests (require model server)
// ============================================================================

mod model_integration_tests {
    use super::*;

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1 and running AgentCPM-GUI server"]
    fn test_gui_bridge_health() {
        if !should_run_e2e() {
            return;
        }

        let config = GuiBridgeConfig {
            endpoint: get_gui_endpoint(),
            ..Default::default()
        };

        let bridge = GuiBridge::new(config);

        match bridge.health_check() {
            Ok(healthy) => {
                println!("Health check: {}", if healthy { "OK" } else { "FAIL" });
                assert!(healthy);
            }
            Err(e) => {
                println!(
                    "Health check failed (expected if server not running): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1, device, and AgentCPM-GUI server"]
    fn test_full_prediction_pipeline() {
        if !should_run_e2e() {
            return;
        }

        let device_id = match get_test_device_id() {
            Some(id) => id,
            None => return,
        };

        // Connect to device
        let mut manager = DeviceManager::new();
        if manager.connect(&device_id).is_err() {
            return;
        }

        // Capture screenshot
        let screenshot = match manager.capture_screenshot() {
            Ok(s) => s,
            Err(_) => {
                manager.disconnect().ok();
                return;
            }
        };

        // Create bridge
        let config = GuiBridgeConfig {
            endpoint: get_gui_endpoint(),
            ..Default::default()
        };
        let bridge = GuiBridge::new(config);

        // Predict action
        match bridge.predict(&screenshot, "What app icons do you see?") {
            Ok(result) => {
                println!("Prediction successful:");
                println!("  Action: {:?}", result.action);
                println!("  Confidence: {:.2}", result.confidence);
                println!("  Latency: {}ms", result.latency_ms);
                println!("  Model: {}", result.model);
                if let Some(ref output) = result.raw_output {
                    println!("  Raw output: {}", output);
                }
            }
            Err(e) => {
                println!("Prediction failed: {}", e);
            }
        }

        manager.disconnect().ok();
    }

    #[test]
    #[ignore = "Requires GUI_RUN_E2E=1, device, and AgentCPM-GUI server"]
    fn test_multi_step_task() {
        if !should_run_e2e() {
            return;
        }

        let device_id = match get_test_device_id() {
            Some(id) => id,
            None => return,
        };

        let mut manager = DeviceManager::new();
        if manager.connect(&device_id).is_err() {
            return;
        }

        let config = GuiBridgeConfig {
            endpoint: get_gui_endpoint(),
            ..Default::default()
        };
        let bridge = GuiBridge::new(config);

        let mut conversation = GuiConversation::new(10);

        // Execute up to 5 steps
        for step in 1..=5 {
            println!("\n--- Step {} ---", step);

            let screenshot = match manager.capture_screenshot() {
                Ok(s) => s,
                Err(e) => {
                    println!("Screenshot failed: {}", e);
                    break;
                }
            };

            let instruction = "Navigate to Settings";
            let result = match bridge.predict(&screenshot, instruction) {
                Ok(r) => r,
                Err(e) => {
                    println!("Prediction failed: {}", e);
                    break;
                }
            };

            println!("Action: {:?}", result.action);

            // Check for task completion
            if let Some(status) = result.action.status {
                match status {
                    TaskStatus::Finish | TaskStatus::Satisfied => {
                        println!("Task completed!");
                        break;
                    }
                    TaskStatus::Impossible => {
                        println!("Task impossible!");
                        break;
                    }
                    _ => {}
                }
            }

            // Execute action
            if result.action.is_primitive() {
                if let Err(e) = manager.execute_action(&result.action) {
                    println!("Action execution failed: {}", e);
                    break;
                }
            }

            // Add to conversation
            conversation.add_turn(instruction, result.action);

            // Wait for screen to stabilize
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

        println!("\nConversation history:");
        println!("{}", conversation.get_context());

        manager.disconnect().ok();
    }
}

// ============================================================================
// Benchmark Tests
// ============================================================================

mod benchmark_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_action_serialization() {
        let action = GuiAction::click_with_thought(500, 300, "Benchmark click")
            .with_status(TaskStatus::Continue);

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = action.to_compact_json();
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() / iterations;

        println!(
            "Action serialization: {} iterations in {:?} ({} ns/op)",
            iterations, elapsed, per_op
        );

        assert!(per_op < 10000); // Less than 10µs per operation
    }

    #[test]
    fn bench_coordinate_conversion() {
        let iterations = 100000;
        let start = Instant::now();

        for i in 0..iterations {
            let point = NormalizedPoint::from_absolute(i % 1080, i % 1920, 1080, 1920);
            let _ = point.to_absolute(1080, 1920);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() / iterations as u128;

        println!(
            "Coordinate conversion: {} iterations in {:?} ({} ns/op)",
            iterations, elapsed, per_op
        );

        assert!(per_op < 1000); // Less than 1µs per operation
    }
}
