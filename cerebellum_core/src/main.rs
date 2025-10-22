// src/main.rs
mod protocol;
mod lob;
mod exchange;
// NEW Modules
mod matching_engine;
mod microstructure;
mod storage;

use protocol::{CortexCommand, CerebellumReport};
use lob::{GlobalBooks, market_data_handler};
use exchange::ExchangeClient;
// NEW Imports
use microstructure::{GlobalMicrostructure, compute_and_update_microstructure};
use storage::DataStorage;

use log::{info, error, warn};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use std::collections::HashMap;
use std::time::Duration as StdDuration;

// ZMQ Endpoints
const ZMQ_COMMAND_ENDPOINT: &str = "tcp://127.0.0.1:5555";
const ZMQ_REPORT_ENDPOINT: &str = "tcp://127.0.0.1:5556";

// (start_ipc_threads function remains the same as the previous implementation)
/// Handles the ZMQ communication in dedicated blocking threads.
fn start_ipc_threads(
    cmd_tx: tokio::sync::mpsc::Sender<CortexCommand>,
    report_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<CerebellumReport>>>,
) {
    // Thread 1: Command Receiver (PULL)
    std::thread::spawn(move || {
        let context = zmq::Context::new();
        let receiver = context.socket(zmq::PULL).unwrap();
        receiver.bind(ZMQ_COMMAND_ENDPOINT).expect("Failed to bind PULL socket");
        info!("[IPC] Listening for commands on {}", ZMQ_COMMAND_ENDPOINT);

        loop {
            match receiver.recv_string(0) {
                Ok(Ok(msg)) => {
                    match serde_json::from_str::<CortexCommand>(&msg) {
                        Ok(cmd) => {
                            // Forward command to the main async loop
                            if cmd_tx.blocking_send(cmd).is_err() {
                                error!("[IPC] Failed to forward command to processing channel.");
                            }
                        },
                        Err(e) => error!("[IPC] Deserialization error: {}", e),
                    }
                },
                Err(e) => error!("[IPC] ZMQ Receive error: {}", e),
                _ => {}
            }
        }
    });

    // Thread 2: Report Sender (PUSH) with Heartbeat
    std::thread::spawn(move || {
        let context = zmq::Context::new();
        let sender = context.socket(zmq::PUSH).unwrap();
        sender.connect(ZMQ_REPORT_ENDPOINT).expect("Failed to connect PUSH socket");
        info!("[IPC] Reporting to {}", ZMQ_REPORT_ENDPOINT);

        let mut last_heartbeat = std::time::Instant::now();
        const HEARTBEAT_INTERVAL: StdDuration = StdDuration::from_secs(5);

        // We need a runtime to handle the async lock inside the sync thread
        let runtime = tokio::runtime::Runtime::new().unwrap();

        loop {
            // Sleep briefly
            std::thread::sleep(StdDuration::from_millis(50));

            runtime.block_on(async {
                let mut report_rx_guard = report_rx.lock().await;

                // 1. Process pending reports
                while let Ok(report) = report_rx_guard.try_recv() {
                     match serde_json::to_string(&report) {
                        Ok(msg) => {
                            if sender.send(msg.as_bytes(), zmq::DONTWAIT).is_err() {
                                error!("[IPC] Failed to send report.");
                            }
                        },
                        Err(e) => error!("[IPC] Serialization error: {}", e),
                    }
                }

                // 2. Send Heartbeat
                if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                    let heartbeat = CerebellumReport::Heartbeat {
                        timestamp: chrono::Utc::now().timestamp_millis() as u64,
                        status: "Healthy".to_string(), // Should reflect actual system health
                    };
                    if let Ok(msg) = serde_json::to_string(&heartbeat) {
                        if sender.send(msg.as_bytes(), zmq::DONTWAIT).is_err() {
                            warn!("[IPC] Failed to send heartbeat.");
                        }
                    }
                    last_heartbeat = std::time::Instant::now();
                }
            });
        }
    });
}


#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    env_logger::init();
    info!("[Cerebellum] Initializing...");

    // Initialize shared state and components
    let global_books: GlobalBooks = Arc::new(RwLock::new(HashMap::new()));
    let global_micro: GlobalMicrostructure = Arc::new(RwLock::new(HashMap::new()));
    // NOTE: DataStorage initialized for future tick data persistence features
    let _data_storage = Arc::new(DataStorage::new());

    // Initialize Exchange Client (Load keys from env vars)
    let api_key = std::env::var("BYBIT_API_KEY").expect("BYBIT_API_KEY not set");
    let api_secret = std::env::var("BYBIT_API_SECRET").expect("BYBIT_API_SECRET not set");
    let client = ExchangeClient::new(api_key, api_secret, false);

    // IPC Channels and Threads Setup
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel::<CortexCommand>(100);
    let (report_tx, report_rx) = tokio::sync::mpsc::channel::<CerebellumReport>(100);
    let report_rx_arc = Arc::new(Mutex::new(report_rx));

    start_ipc_threads(cmd_tx, report_rx_arc);

    let mut mdh_handle: Option<tokio::task::JoinHandle<()>> = None;

    // Main application loop
    info!("[Cerebellum] Ready. Waiting for commands...");
    loop {
        tokio::select! {
            Some(cmd) = cmd_rx.recv() => {
                 match cmd {
                    CortexCommand::Initialize { assets } => {
                        info!("[Main] Initialize command received.");
                        if let Some(handle) = mdh_handle.take() {
                            handle.abort(); // Stop existing MDH
                        }
                        let books_clone = global_books.clone();
                        let assets_vec: Vec<String> = assets.into_iter().collect();

                        mdh_handle = Some(tokio::spawn(async move {
                            market_data_handler(books_clone, assets_vec).await;
                        }));
                    },
                    CortexCommand::ExecuteOrder { .. } => {
                        // Spawn task to handle execution via IEL
                        let client_clone = client.clone();
                        let books_clone = global_books.clone();
                        let report_tx_clone = report_tx.clone();

                        tokio::spawn(async move {
                            match client_clone.execute_iel(cmd, books_clone).await {
                                Ok(report) => {
                                    if report_tx_clone.send(report).await.is_err() {
                                        error!("[Main] Failed to send report to IPC thread.");
                                    }
                                },
                                Err(e) => {
                                    error!("[Execution] Failed: {}", e);
                                }
                            }
                        });
                    },
                    CortexCommand::Halt => {
                        warn!("[FAILSAFE] HALT command received. Shutting down.");
                        // TODO: Implement graceful shutdown (cancel all open orders)
                        break;
                    }
                }
            }
            // Periodic Microstructure Computation (e.g., every 500ms)
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(500)) => {
                compute_and_update_microstructure(&global_books, &global_micro).await;
            }
        }
    }
}