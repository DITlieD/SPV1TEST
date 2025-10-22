// src/protocol.rs
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// Define IEL Modes with parameters
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum IelMode {
    Aggressive,
    Passive,
    Twap { duration_secs: u64 },
    Vwap, // Placeholder
    Adaptive,
}

// Commands sent from Python (Cortex) to Rust (Cerebellum)
#[derive(Serialize, Deserialize, Debug)]
pub enum CortexCommand {
    Initialize {
        assets: HashSet<String>,
    },
    ExecuteOrder {
        symbol: String,
        side: String, // "Buy" or "Sell"
        quantity: f64,
        strategy_id: String,
        iel_mode: IelMode,
        timestamp_sent: f64,
    },
    Halt, // Failsafe command
}

// Reports sent from Rust (Cerebellum) to Python (Cortex)
#[derive(Serialize, Deserialize, Debug)]
pub enum CerebellumReport {
    OrderUpdate {
        symbol: String,
        strategy_id: String,
        status: String, // e.g., "NEW", "FILLED", "FAILED"
        avg_price: f64,
        executed_qty: f64,
        latency_ms: u64,
        error_message: Option<String>,
        timestamp_sent: f64,
    },
    // NEW: Heartbeat message
    Heartbeat {
        timestamp: u64,
        status: String,
    }
}