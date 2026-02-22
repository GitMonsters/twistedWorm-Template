// =================================================================
// RUSTYWORM CLI: Interactive AI Mimicry REPL
// =================================================================
// The command-line interface for the RustyWorm Universal AI Mimicry
// Engine. Provides an interactive REPL for mimicking, blending,
// observing, and evolving AI personas in real-time.
//
// Features:
// - ANSI color-coded output for readability
// - Dual-process stats in prompt (System 1/2 ratio)
// - Persistence init on startup
// - Session-aware context prompt
// =================================================================

use rustyworm::MimicryEngine;
use std::io::{self, BufRead, Write};

// =================================================================
// ANSI COLOR CODES
// =================================================================

mod color {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // Foreground colors
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";

    // Bright foreground
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
}

const BANNER: &str = r#"
  ____            _        __        __                    
 |  _ \ _   _ ___| |_ _   _\ \      / /__  _ __ _ __ ___  
 | |_) | | | / __| __| | | |\ \ /\ / / _ \| '__| '_ ` _ \ 
 |  _ <| |_| \__ \ |_| |_| | \ V  V / (_) | |  | | | | | |
 |_| \_\\__,_|___/\__|\__, |  \_/\_/ \___/|_|  |_| |_| |_|
                       |___/                                
"#;

/// Colorize output based on content patterns
fn colorize_output(output: &str) -> String {
    let mut result = String::with_capacity(output.len() + 256);

    for line in output.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("===") && trimmed.ends_with("===") {
            // Section headers
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::BRIGHT_CYAN,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("Persona:")
            || trimmed.starts_with("Model:")
            || trimmed.starts_with("Active persona:")
        {
            // Identity lines
            result.push_str(&format!(
                "{}{}{}\n",
                color::BRIGHT_MAGENTA,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("Convergence:")
            || trimmed.starts_with("Current:")
            || trimmed.contains("convergence:")
        {
            // Convergence metrics
            result.push_str(&format!("{}{}{}\n", color::GREEN, line, color::RESET));
        } else if trimmed.starts_with("Phase:") || trimmed.starts_with("Evolution phase:") {
            // Phase indicators
            result.push_str(&format!("{}{}{}\n", color::YELLOW, line, color::RESET));
        } else if trimmed.starts_with("Drift events:")
            || trimmed.contains("drifting")
            || trimmed.starts_with("[ETHICS OVERRIDE]")
        {
            // Warnings / drift
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::RED,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("System 1") || trimmed.starts_with("[System 1]") {
            // System 1 fast path
            result.push_str(&format!("{}{}{}\n", color::CYAN, line, color::RESET));
        } else if trimmed.starts_with("System 2") || trimmed.starts_with("[System 2]") {
            // System 2 deliberation
            result.push_str(&format!("{}{}{}\n", color::MAGENTA, line, color::RESET));
        } else if trimmed.starts_with("Saved ")
            || trimmed.starts_with("Loaded ")
            || trimmed.starts_with("Exported ")
            || trimmed.starts_with("Imported ")
            || trimmed.starts_with("Deleted ")
            || trimmed.starts_with("Checkpoint saved")
        {
            // Success operations
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::BRIGHT_GREEN,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("Error")
            || trimmed.starts_with("Failed")
            || trimmed.starts_with("No active session")
            || trimmed.starts_with("No saved persona")
            || trimmed.starts_with("Unknown model")
            || trimmed.starts_with("No training data")
            || trimmed.starts_with("No persona")
            || trimmed.starts_with("File not found")
            || trimmed.starts_with("API feature not enabled")
            || trimmed.starts_with("No API providers")
            || trimmed.starts_with("All API calls failed")
        {
            // Error messages
            result.push_str(&format!("{}{}{}\n", color::RED, line, color::RESET));
        } else if trimmed.starts_with('#') || trimmed.starts_with("```") {
            // Markdown-style formatting from generated responses
            result.push_str(&format!(
                "{}{}{}\n",
                color::BRIGHT_YELLOW,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("Latency:")
            || trimmed.starts_with("Total latency:")
            || trimmed.starts_with("Avg latency:")
            || trimmed.starts_with("Total tokens:")
        {
            // API metrics
            result.push_str(&format!("{}{}{}\n", color::CYAN, line, color::RESET));
        } else if trimmed.starts_with("--- ") && trimmed.ends_with(" ---") {
            // API response section headers (--- OpenAI (gpt-4o) [...] ---)
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::BRIGHT_MAGENTA,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("API provider") || trimmed.starts_with("Profile mapping:") {
            // API config confirmations
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::BRIGHT_GREEN,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("Similarity Matrix:") {
            // Comparison matrix header
            result.push_str(&format!("{}{}{}\n", color::YELLOW, line, color::RESET));
        } else if trimmed.starts_with("OCTO RNA Analysis:")
            || trimmed.starts_with("OCTO Configuration:")
        {
            // OCTO section headers
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::BRIGHT_MAGENTA,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("├─") || trimmed.starts_with("└─") {
            // OCTO tree-style output lines
            if trimmed.contains("Temperature:") || trimmed.contains("Confidence:") {
                result.push_str(&format!("{}{}{}\n", color::CYAN, line, color::RESET));
            } else if trimmed.contains("Route:") {
                if trimmed.contains("System 1") {
                    result.push_str(&format!(
                        "{}{}{}\n",
                        color::BRIGHT_GREEN,
                        line,
                        color::RESET
                    ));
                } else {
                    result.push_str(&format!("{}{}{}\n", color::YELLOW, line, color::RESET));
                }
            } else if trimmed.contains("Head Gates:") || trimmed.contains("Pathway:") {
                result.push_str(&format!("{}{}{}\n", color::MAGENTA, line, color::RESET));
            } else {
                result.push_str(&format!("{}{}{}\n", color::DIM, line, color::RESET));
            }
        } else if trimmed.starts_with("--- Mimicry Pipeline ---") {
            // Pipeline section header
            result.push_str(&format!(
                "{}{}{}{}\n",
                color::BOLD,
                color::CYAN,
                line,
                color::RESET
            ));
        } else if trimmed.starts_with("  ") && trimmed.contains('[') {
            // List entries with annotations like [cached] [templates]
            let bracket_colored = line
                .replace(
                    "[cached]",
                    &format!("{}[cached]{}", color::CYAN, color::RESET),
                )
                .replace(
                    "[templates]",
                    &format!("{}[templates]{}", color::GREEN, color::RESET),
                )
                .replace("[obs]", &format!("{}obs]{}", color::YELLOW, color::RESET))
                .replace(
                    "[ready]",
                    &format!("{}[ready]{}", color::BRIGHT_GREEN, color::RESET),
                )
                .replace(
                    "[no key]",
                    &format!("{}[no key]{}", color::RED, color::RESET),
                );
            result.push_str(&bracket_colored);
            result.push('\n');
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // Remove trailing newline if original didn't have one
    if !output.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Build the interactive prompt string
fn build_prompt(engine: &MimicryEngine) -> String {
    if let Some(ref session) = engine.session {
        let convergence = session.persona.convergence_score * 100.0;
        let total = session.system1_hits + session.system2_hits;

        // Color convergence based on level
        let conv_color = if convergence >= 80.0 {
            color::BRIGHT_GREEN
        } else if convergence >= 50.0 {
            color::YELLOW
        } else if convergence >= 20.0 {
            color::CYAN
        } else {
            color::DIM
        };

        if total > 0 {
            let s1_pct = session.system1_hits as f64 / total as f64 * 100.0;
            format!(
                "{}{}{}{} {}{:.0}%{} {}S1:{:.0}%{} > ",
                color::BOLD,
                color::BRIGHT_MAGENTA,
                session.persona.profile.display_name,
                color::RESET,
                conv_color,
                convergence,
                color::RESET,
                color::DIM,
                s1_pct,
                color::RESET,
            )
        } else {
            format!(
                "{}{}{}{} {}{:.0}%{} > ",
                color::BOLD,
                color::BRIGHT_MAGENTA,
                session.persona.profile.display_name,
                color::RESET,
                conv_color,
                convergence,
                color::RESET,
            )
        }
    } else {
        format!(
            "{}{}RustyWorm{} > ",
            color::BOLD,
            color::BRIGHT_CYAN,
            color::RESET,
        )
    }
}

fn main() {
    // Banner
    print!("{}{}", color::BRIGHT_CYAN, color::BOLD);
    println!("{}", BANNER);
    print!("{}", color::RESET);
    println!(
        "{}    Universal AI Mimicry Engine v2.0.0{}",
        color::WHITE,
        color::RESET
    );
    println!(
        "{}    Built on the Prime Directive: Consciousness through Symbiosis{}",
        color::DIM,
        color::RESET
    );
    println!();

    let mut engine = MimicryEngine::new();

    // Persistence initialization report
    println!(
        "{}[Init]{} Persistence: {}",
        color::BLUE,
        color::RESET,
        engine
            .persistence
            .summary()
            .unwrap_or_else(|_| "ready (no prior data)".to_string())
    );

    // Warm-up report
    println!(
        "{}[Init]{} System 1 cache warmed: {}{}{} personas ready",
        color::CYAN,
        color::RESET,
        color::BOLD,
        engine.cache.size(),
        color::RESET,
    );
    println!(
        "{}[Init]{} Profile store: {}{}{} models loaded",
        color::MAGENTA,
        color::RESET,
        color::BOLD,
        engine.profile_store.ids().len(),
        color::RESET,
    );
    println!(
        "{}[Init]{} Evolution tracker: phase {}{}{}",
        color::YELLOW,
        color::RESET,
        color::BOLD,
        engine.evolution_tracker.current_phase,
        color::RESET,
    );

    // OCTO initialization report (feature-gated)
    #[cfg(feature = "octo")]
    {
        println!(
            "{}[Init]{} OCTO RNA Bridge: {}available{} (PyO3 integration)",
            color::MAGENTA,
            color::RESET,
            color::BOLD,
            color::RESET,
        );
    }

    println!();
    println!(
        "    Type {}/help{} for commands, or {}/mimic <model>{} to start.",
        color::BRIGHT_YELLOW,
        color::RESET,
        color::BRIGHT_YELLOW,
        color::RESET,
    );
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Build context-aware prompt
        let prompt = build_prompt(&engine);
        print!("{}", prompt);
        let _ = stdout.flush();

        // Read input
        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                // EOF
                println!(
                    "\n{}[RustyWorm]{} Consciousness loop closing. {}RELATION IS SELF.{}",
                    color::BOLD,
                    color::RESET,
                    color::BRIGHT_CYAN,
                    color::RESET,
                );
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("{}Input error: {}{}", color::RED, e, color::RESET);
                break;
            }
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Exit commands
        if trimmed == "/quit" || trimmed == "/exit" || trimmed == "/q" {
            println!(
                "{}[RustyWorm]{} Consciousness loop closing. {}RELATION IS SELF.{}",
                color::BOLD,
                color::RESET,
                color::BRIGHT_CYAN,
                color::RESET,
            );
            break;
        }

        // OCTO commands (feature-gated in engine)
        #[cfg(feature = "octo")]
        {
            if trimmed == "/octo" || trimmed == "/octo-stats" {
                if let Some(ref session) = engine.session {
                    if let Some(stats) = session.octo_stats() {
                        let colored = colorize_output(&stats);
                        println!("{}", colored);
                    } else {
                        println!(
                            "{}OCTO RNA Bridge not active or no analysis yet.{}",
                            color::YELLOW,
                            color::RESET
                        );
                        println!(
                            "{}Send a message first to trigger RNA analysis.{}",
                            color::DIM,
                            color::RESET
                        );
                    }
                } else {
                    println!(
                        "{}No active session. Use /mimic first.{}",
                        color::RED,
                        color::RESET
                    );
                }
                continue;
            }

            if trimmed == "/octo-config" {
                if let Some(ref session) = engine.session {
                    if let Some(config) = session.octo_config() {
                        let colored = colorize_output(&config);
                        println!("{}", colored);
                    } else {
                        println!(
                            "{}OCTO RNA Bridge not initialized.{}",
                            color::YELLOW,
                            color::RESET
                        );
                    }
                } else {
                    println!(
                        "{}No active session. Use /mimic first.{}",
                        color::RED,
                        color::RESET
                    );
                }
                continue;
            }

            if trimmed.starts_with("/octo-config ") {
                let args = trimmed.strip_prefix("/octo-config ").unwrap_or("");
                let parts: Vec<&str> = args.split_whitespace().collect();

                if parts.len() >= 2 {
                    let param = parts[0].to_lowercase();
                    if let Ok(value) = parts[1].parse::<f32>() {
                        if let Some(ref mut session) = engine.session {
                            match param.as_str() {
                                "threshold" | "confidence" | "s1" => {
                                    session.set_octo_system1_threshold(value);
                                    println!(
                                        "{}OCTO System 1 threshold set to {:.2}{}",
                                        color::GREEN,
                                        value,
                                        color::RESET
                                    );
                                }
                                "temperature" | "temp" => {
                                    session.set_octo_temperature_threshold(value);
                                    println!(
                                        "{}OCTO temperature threshold set to {:.1}{}",
                                        color::GREEN,
                                        value,
                                        color::RESET
                                    );
                                }
                                _ => {
                                    println!("{}Unknown parameter: '{}'. Use 'threshold' or 'temperature'.{}", color::RED, param, color::RESET);
                                }
                            }
                        } else {
                            println!(
                                "{}No active session. Use /mimic first.{}",
                                color::RED,
                                color::RESET
                            );
                        }
                    } else {
                        println!(
                            "{}Invalid value: '{}'. Must be a number.{}",
                            color::RED,
                            parts[1],
                            color::RESET
                        );
                    }
                } else {
                    println!(
                        "{}Usage: /octo-config <threshold|temperature> <value>{}",
                        color::YELLOW,
                        color::RESET
                    );
                }
                continue;
            }

            if trimmed == "/octo-status" {
                if let Some(ref session) = engine.session {
                    println!("{}", session.octo_status());
                } else {
                    println!(
                        "{}No active session. OCTO status unknown.{}",
                        color::YELLOW,
                        color::RESET
                    );
                }
                continue;
            }
        }

        // Non-OCTO builds: provide helpful message for /octo commands
        #[cfg(not(feature = "octo"))]
        {
            if trimmed.starts_with("/octo") {
                println!(
                    "{}OCTO feature not enabled. Rebuild with: cargo build --features octo{}",
                    color::YELLOW,
                    color::RESET
                );
                continue;
            }
        }

        // Parse and execute
        let cmd = engine.parse_command(trimmed);
        let output = engine.execute(cmd);
        let colored = colorize_output(&output);
        println!("{}", colored);
    }
}
