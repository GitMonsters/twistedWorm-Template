//! GUI Agent Example - Cross-Platform GUI Automation with AgentCPM-GUI
//!
//! This example demonstrates how to use RustyWorm's GUI agent capabilities
//! to automate GUI interactions across Android, iOS, and Desktop platforms.
//!
//! Prerequisites:
//! - AgentCPM-GUI model served via vLLM (http://localhost:8000)
//! - For Android: ADB installed and device connected
//! - For iOS: libimobiledevice + WebDriverAgent
//! - For Desktop: Platform-specific tools (xdotool, cliclick, etc.)
//!
//! Run with:
//!   cargo run --features gui --example gui_agent
//!
//! Or with all features:
//!   cargo run --features full --example gui_agent

use std::time::Duration;

#[cfg(feature = "gui")]
use rustyworm::mimicry::{
    gui_agent::{
        GuiAction, GuiTaskSequence, NormalizedPoint, Platform, PressKey, SwipeDirection, TaskStatus,
    },
    gui_bridge::{GuiBridge, GuiBridgeConfig, GuiConversation, Language},
    platforms::{DeviceManager, PlatformFactory},
};

fn main() {
    println!("=== RustyWorm GUI Agent Example ===\n");

    #[cfg(feature = "gui")]
    {
        // Example 1: Basic GUI Actions
        demo_gui_actions();

        // Example 2: Device Discovery
        demo_device_discovery();

        // Example 3: GUI Bridge Configuration
        demo_gui_bridge();

        // Example 4: Task Sequences
        demo_task_sequences();

        // Example 5: Multi-turn Conversations
        demo_conversations();

        // Example 6: Platform-specific automation
        demo_platform_automation();
    }

    #[cfg(not(feature = "gui"))]
    {
        println!("GUI feature not enabled. Run with:");
        println!("  cargo run --features gui --example gui_agent");
    }

    println!("\n=== Example Complete ===");
}

#[cfg(feature = "gui")]
fn demo_gui_actions() {
    println!("--- Demo 1: GUI Actions ---\n");

    // Click action
    let click = GuiAction::click(500, 300);
    println!("Click action: {:?}", click);
    println!("  Type: {}", click.action_type());
    println!("  JSON: {}", click.to_compact_json());

    // Click with reasoning
    let click_with_thought =
        GuiAction::click_with_thought(750, 200, "Clicking the submit button in the form");
    println!("\nClick with thought: {:?}", click_with_thought);

    // Long press
    let long_press = GuiAction::long_press(500, 500, 1000);
    println!("\nLong press (1s): {:?}", long_press);
    println!("  Type: {}", long_press.action_type());

    // Swipe actions
    let swipe_down = GuiAction::swipe_direction(500, 300, SwipeDirection::Down);
    println!("\nSwipe down: {:?}", swipe_down);

    let swipe_to = GuiAction::swipe_to(100, 500, 900, 500);
    println!("Swipe to point: {:?}", swipe_to);

    // Type text
    let type_action = GuiAction::type_text("Hello, RustyWorm!");
    println!("\nType text: {:?}", type_action);

    // Press hardware key
    let back = GuiAction::press_key(PressKey::Back);
    let home = GuiAction::press_key(PressKey::Home);
    println!("\nPress Back: {:?}", back);
    println!("Press Home: {:?}", home);

    // Wait action
    let wait = GuiAction::wait(2000);
    println!("\nWait 2s: {:?}", wait);

    // Status update
    let finish = GuiAction::status(TaskStatus::Finish);
    println!("Finish status: {:?}", finish);

    // Chained action with thought and status
    let complex_action = GuiAction::click(500, 800)
        .with_thought("Final confirmation button")
        .with_status(TaskStatus::Finish);
    println!("\nComplex action: {}", complex_action.to_compact_json());

    // Coordinate conversion
    let point = NormalizedPoint::from_absolute(540, 960, 1080, 1920);
    println!("\nCoordinate conversion:");
    println!("  Absolute (540, 960) on 1080x1920 -> Normalized {}", point);
    let (abs_x, abs_y) = point.to_absolute(1080, 1920);
    println!(
        "  Normalized {} on 1080x1920 -> Absolute ({}, {})",
        point, abs_x, abs_y
    );

    // Distance calculation
    let p1 = NormalizedPoint::new(0, 0);
    let p2 = NormalizedPoint::new(1000, 1000);
    println!(
        "  Distance from {} to {}: {:.2}",
        p1,
        p2,
        p1.distance_to(&p2)
    );
}

#[cfg(feature = "gui")]
fn demo_device_discovery() {
    println!("\n--- Demo 2: Device Discovery ---\n");

    // Create device manager
    let manager = DeviceManager::new();

    // List available devices
    println!("Detecting devices across all platforms...");
    let devices = manager.list_devices();

    if devices.is_empty() {
        println!("No devices found (expected if no physical devices connected)");
    } else {
        println!("Found {} device(s):", devices.len());
        for device in &devices {
            println!("\n  Device: {}", device.name);
            println!("    ID: {}", device.id);
            println!("    Platform: {:?}", device.platform);
            if let Some(ref version) = device.os_version {
                println!("    OS Version: {}", version);
            }
            if let Some((w, h)) = device.resolution {
                println!("    Resolution: {}x{}", w, h);
            }
            println!("    Connection: {:?}", device.connection);
        }
    }

    // Create platform instances
    println!("\nCreating platform instances:");

    let android = PlatformFactory::create(Platform::Android);
    println!("  Android platform: {:?}", android.platform());
    println!("    Connected: {}", android.is_connected());

    let ios = PlatformFactory::create(Platform::Ios);
    println!("  iOS platform: {:?}", ios.platform());
    println!("    Connected: {}", ios.is_connected());

    let desktop = PlatformFactory::create(Platform::MacOs);
    println!("  Desktop platform: {:?}", desktop.platform());
    println!("    Connected: {}", desktop.is_connected());
}

#[cfg(feature = "gui")]
fn demo_gui_bridge() {
    println!("\n--- Demo 3: GUI Bridge Configuration ---\n");

    // Default configuration
    let default_config = GuiBridgeConfig::default();
    println!("Default configuration:");
    println!("  Endpoint: {}", default_config.endpoint);
    println!("  Model: {}", default_config.model_name);
    println!("  Timeout: {}ms", default_config.timeout_ms);
    println!("  Max tokens: {}", default_config.max_tokens);
    println!("  Temperature: {}", default_config.temperature);
    println!("  Top-p: {}", default_config.top_p);
    println!("  Enable thought: {}", default_config.enable_thought);
    println!("  Language: {:?}", default_config.language);

    // Custom configuration
    let custom_config = GuiBridgeConfig {
        endpoint: "http://gpu-server:8000/v1/chat/completions".to_string(),
        model_name: "AgentCPM-GUI-4B".to_string(),
        timeout_ms: 60000,
        max_tokens: 4096,
        temperature: 0.0, // Deterministic
        top_p: 0.1,
        enable_thought: true,
        language: Language::English,
    };
    println!("\nCustom configuration:");
    println!("  Endpoint: {}", custom_config.endpoint);
    println!("  Model: {}", custom_config.model_name);

    // Create bridge
    let bridge = GuiBridge::new(default_config);
    println!("\nBridge created:");
    println!("  Endpoint: {}", bridge.config().endpoint);

    // Health check (will fail without server)
    match bridge.health_check() {
        Ok(healthy) => println!("  Health check: {}", if healthy { "OK" } else { "FAIL" }),
        Err(e) => println!("  Health check: Not available ({})", e),
    }
}

#[cfg(feature = "gui")]
fn demo_task_sequences() {
    println!("\n--- Demo 4: Task Sequences ---\n");

    // Create a login task sequence
    let mut login_sequence = GuiTaskSequence::new("Log into the application", Platform::Android);
    login_sequence.app_context = Some("com.example.app".to_string());

    // Add actions
    login_sequence.add_action(GuiAction::click_with_thought(
        500,
        300,
        "Click username field",
    ));
    login_sequence.add_action(GuiAction::type_text("user@example.com"));
    login_sequence.add_action(GuiAction::click_with_thought(
        500,
        400,
        "Click password field",
    ));
    login_sequence.add_action(GuiAction::type_text("securepassword123"));
    login_sequence.add_action(
        GuiAction::click_with_thought(500, 550, "Click login button")
            .with_status(TaskStatus::Finish),
    );

    // Complete the sequence
    login_sequence.complete(3500);

    println!("Login sequence:");
    println!("  Instruction: {}", login_sequence.instruction);
    println!("  Platform: {:?}", login_sequence.platform);
    println!("  App: {:?}", login_sequence.app_context);
    println!("  Actions: {}", login_sequence.len());
    println!("  Completed: {}", login_sequence.completed);
    println!("  Total time: {}ms", login_sequence.total_time_ms);

    println!("\nAction summary:");
    for (action_type, count) in login_sequence.action_summary() {
        println!("  {}: {}", action_type, count);
    }

    // Create a scroll-and-find sequence
    let mut scroll_sequence = GuiTaskSequence::new("Find and tap Settings", Platform::Ios);

    scroll_sequence.add_action(GuiAction::swipe_direction(500, 700, SwipeDirection::Up));
    scroll_sequence.add_action(GuiAction::wait(500));
    scroll_sequence.add_action(GuiAction::swipe_direction(500, 700, SwipeDirection::Up));
    scroll_sequence.add_action(GuiAction::wait(500));
    scroll_sequence.add_action(
        GuiAction::click_with_thought(500, 450, "Found Settings icon")
            .with_status(TaskStatus::Finish),
    );

    scroll_sequence.complete(2500);

    println!("\nScroll sequence:");
    println!("  Actions: {}", scroll_sequence.len());
    println!("  Summary: {:?}", scroll_sequence.action_summary());
}

#[cfg(feature = "gui")]
fn demo_conversations() {
    println!("\n--- Demo 5: Multi-turn Conversations ---\n");

    // Create conversation with max 10 turns
    let mut conversation = GuiConversation::new(10);
    println!("Created conversation (max 10 turns)");
    println!("  Length: {}", conversation.len());
    println!("  Empty: {}", conversation.is_empty());

    // Simulate a multi-turn interaction
    conversation.add_turn("Open Settings", GuiAction::click(500, 300));
    conversation.add_turn(
        "Find Wi-Fi option",
        GuiAction::swipe_direction(500, 500, SwipeDirection::Down),
    );
    conversation.add_turn("Tap Wi-Fi", GuiAction::click(500, 200));
    conversation.add_turn("Toggle Wi-Fi on", GuiAction::click(900, 150));

    println!("\nAfter 4 turns:");
    println!("  Length: {}", conversation.len());

    println!("\nConversation context:");
    println!("{}", conversation.get_context());

    // Test max history limit
    let mut limited_conv = GuiConversation::new(3);
    for i in 0..5 {
        limited_conv.add_turn(
            &format!("Action {}", i),
            GuiAction::click((i * 100) as u16, (i * 100) as u16),
        );
    }

    println!("\nLimited conversation (max 3, added 5):");
    println!("  Length: {}", limited_conv.len());
    println!("  Context:\n{}", limited_conv.get_context());

    // Clear conversation
    conversation.clear();
    println!("\nAfter clear: empty = {}", conversation.is_empty());
}

#[cfg(feature = "gui")]
fn demo_platform_automation() {
    println!("\n--- Demo 6: Platform-Specific Automation ---\n");

    // Note: These won't execute without actual devices/tools
    println!("Platform capabilities (simulation only):\n");

    // Android via ADB
    println!("Android (ADB):");
    println!("  Screenshot: adb exec-out screencap -p");
    println!("  Tap: adb shell input tap <x> <y>");
    println!("  Swipe: adb shell input swipe <x1> <y1> <x2> <y2> <duration>");
    println!("  Text: adb shell input text '<text>'");
    println!("  Keyevent: adb shell input keyevent <code>");

    println!("\niOS (libimobiledevice + WDA):");
    println!("  Screenshot: idevicescreenshot");
    println!("  Tap: POST /session/:id/wda/tap");
    println!("  Swipe: POST /session/:id/wda/dragfromtoforduration");
    println!("  Text: POST /session/:id/wda/keys");
    println!("  Home: POST /session/:id/wda/homescreen");

    println!("\nmacOS (osascript + cliclick):");
    println!("  Screenshot: screencapture -x");
    println!("  Click: cliclick c:<x>,<y>");
    println!("  Type: osascript -e 'tell app \"System Events\" to keystroke \"<text>\"'");

    println!("\nLinux (xdotool):");
    println!("  Screenshot: gnome-screenshot or scrot");
    println!("  Click: xdotool mousemove <x> <y> click 1");
    println!("  Type: xdotool type '<text>'");

    println!("\nWindows (PowerShell):");
    println!("  Screenshot: [System.Windows.Forms.Screen]::PrimaryScreen");
    println!("  Click: [System.Windows.Forms.Cursor]::Position + SendInput");
    println!("  Type: [System.Windows.Forms.SendKeys]::SendWait('<text>')");

    // Device manager workflow
    println!("\n--- Device Manager Workflow ---\n");

    let manager = DeviceManager::new();
    println!("1. Create DeviceManager");
    println!("   Connected: {}", manager.is_connected());

    println!("\n2. List devices:");
    let devices = manager.list_devices();
    println!("   Found: {} device(s)", devices.len());

    println!("\n3. Connect to device (simulated):");
    println!("   manager.connect(\"emulator-5554\")");
    // Would call: manager.connect("emulator-5554")?;

    println!("\n4. Capture screenshot:");
    println!("   let screenshot = manager.capture_screenshot()?;");

    println!("\n5. Execute action:");
    println!("   manager.execute_action(&GuiAction::click(500, 300))?;");

    println!("\n6. Get app context:");
    println!("   let context = manager.get_context()?;");

    println!("\n7. Disconnect:");
    println!("   manager.disconnect()?;");
}
