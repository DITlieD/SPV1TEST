// src/exchange.rs
use reqwest::Client;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use anyhow::{Result, anyhow};
use log::{info, warn, error};
use crate::protocol::{CortexCommand, CerebellumReport, IelMode};
use crate::lob::GlobalBooks;
use tokio::time::{sleep, Duration};
use rand::Rng;

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
pub struct ExchangeClient {
    #[allow(dead_code)]
    client: Client,
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    api_secret: String,
    #[allow(dead_code)]
    base_url: String,
}

impl ExchangeClient {
    // (new() and sign() methods remain the same as the previous implementation)
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let base_url = if testnet {
            "https://api-testnet.bybit.com".to_string()
        } else {
            "https://api.bybit.com".to_string()
        };
        Self { client: Client::new(), api_key, api_secret, base_url }
    }

    #[allow(dead_code)]
    fn sign(&self, timestamp: &str, payload: &str) -> String {
        let recv_window = "5000";
        let param_str = format!("{}{}{}{}", timestamp, self.api_key, recv_window, payload);
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes()).unwrap();
        mac.update(param_str.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }


    // Helper to get BBO safely
    async fn get_bbo(&self, symbol: &str, books: &GlobalBooks) -> Result<(f64, f64)> {
        let books_read = books.read().await;
        let book_lock = books_read.get(symbol).ok_or_else(|| anyhow!("LOB data unavailable for {}", symbol))?;
        let book = book_lock.read().await;
        let (best_bid, best_ask) = book.get_bbo();

        let bid_p = best_bid.ok_or_else(|| anyhow!("Best bid missing"))?.0;
        let ask_p = best_ask.ok_or_else(|| anyhow!("Best ask missing"))?.0;
        Ok((bid_p, ask_p))
    }

    // Intelligent Execution Layer (IEL) Logic
    pub async fn execute_iel(&self, cmd: CortexCommand, books: GlobalBooks) -> Result<CerebellumReport> {
        let start_time = std::time::Instant::now();

        if let CortexCommand::ExecuteOrder { symbol, side, quantity, strategy_id, iel_mode, timestamp_sent } = cmd {

            // --- Smart Order Router (SOR) / Strategy Dispatcher ---
            let result = match iel_mode {
                IelMode::Aggressive => self.execute_aggressive(&symbol, &side, quantity, &books).await,
                IelMode::Twap { duration_secs } => self.execute_twap(&symbol, &side, quantity, duration_secs).await,
                IelMode::Adaptive => self.execute_adaptive(&symbol, &side, quantity, &books).await,
                _ => {
                    warn!("[IEL] Mode {:?} not fully implemented. Falling back to Aggressive.", iel_mode);
                    self.execute_aggressive(&symbol, &side, quantity, &books).await
                }
            };

            let execution_time = start_time.elapsed().as_millis();

            // Standardize report generation
            match result {
                Ok(mut report) => {
                    if let CerebellumReport::OrderUpdate { strategy_id: ref mut sid, latency_ms: ref mut lat, timestamp_sent: ref mut ts, .. } = report {
                        *sid = strategy_id;
                        *lat = execution_time as u64;
                        *ts = timestamp_sent;
                    }
                    Ok(report)
                },
                Err(e) => {
                    error!("[IEL] Execution failed: {}", e);
                    // Return failure report
                    Ok(CerebellumReport::OrderUpdate {
                        symbol, strategy_id, status: "FAILED".to_string(), avg_price: 0.0, executed_qty: 0.0,
                        latency_ms: execution_time as u64, error_message: Some(e.to_string()), timestamp_sent,
                    })
                }
            }
        } else {
            Err(anyhow!("Invalid command type received by IEL"))
        }
    }

    async fn execute_aggressive(&self, symbol: &str, side: &str, quantity: f64, books: &GlobalBooks) -> Result<CerebellumReport> {
        let (best_bid, best_ask) = self.get_bbo(symbol, books).await?;

        // Price aggressively (5 bps through the book)
        let price = if side == "Buy" { best_ask * 1.0005 } else { best_bid * 0.9995 };

        // TODO: Implement actual API call (Limit IOC or Market)
        info!("[IEL Aggressive] Executing {} @ {:.4}", side, price);
        sleep(Duration::from_millis(50)).await; // Simulate latency

        // Placeholder success response
        Ok(CerebellumReport::OrderUpdate {
            symbol: symbol.to_string(), strategy_id: String::new(), status: "FILLED".to_string(),
            avg_price: price, executed_qty: quantity, latency_ms: 0, error_message: None, timestamp_sent: 0.0,
        })
    }

    // TWAP Strategy
    async fn execute_twap(&self, symbol: &str, _side: &str, quantity: f64, duration_secs: u64) -> Result<CerebellumReport> {
        let num_slices = (duration_secs / 15).max(1); // Aim for ~15s slices
        let slice_qty = quantity / num_slices as f64;
        let base_interval = Duration::from_secs(duration_secs / num_slices);

        let mut total_executed_qty = 0.0;
        let mut total_cost = 0.0;

        for i in 0..num_slices {
            // TODO: Execute slice using actual API call with side parameter
            let execution_price = 1000.0; // Placeholder price
            info!("[IEL TWAP] Executing slice {}/{} for {}", i + 1, num_slices, symbol);

            total_executed_qty += slice_qty;
            total_cost += slice_qty * execution_price;

            if i < num_slices - 1 {
                // Randomize interval (+/- 20%)
                let randomization = rand::thread_rng().gen_range(0.8..1.2);
                sleep(base_interval.mul_f64(randomization)).await;
            }
        }

        let avg_price = if total_executed_qty > 0.0 { total_cost / total_executed_qty } else { 0.0 };

        Ok(CerebellumReport::OrderUpdate {
            symbol: symbol.to_string(), strategy_id: String::new(), status: "FILLED".to_string(),
            avg_price, executed_qty: total_executed_qty, latency_ms: 0, error_message: None, timestamp_sent: 0.0,
        })
    }

    // Adaptive (SOR) Strategy: Switches based on spread
    async fn execute_adaptive(&self, symbol: &str, side: &str, quantity: f64, books: &GlobalBooks) -> Result<CerebellumReport> {
        let (best_bid, best_ask) = self.get_bbo(symbol, books).await?;
        let spread = best_ask - best_bid;
        let mid_price = (best_ask + best_bid) / 2.0;
        let spread_pct = spread / mid_price;

        // Threshold: 0.05% spread
        if spread_pct < 0.0005 {
            info!("[IEL Adaptive] Spread tight ({:.4}%). Executing Aggressively.", spread_pct * 100.0);
            self.execute_aggressive(symbol, side, quantity, books).await
        } else {
            info!("[IEL Adaptive] Spread wide ({:.4}%). Executing Passively.", spread_pct * 100.0);
            // Placeholder for Passive (Post-Only) execution
            let price = if side == "Buy" { best_bid } else { best_ask };
             Ok(CerebellumReport::OrderUpdate {
                symbol: symbol.to_string(), strategy_id: String::new(), status: "NEW".to_string(),
                avg_price: price, executed_qty: 0.0, latency_ms: 0, error_message: None, timestamp_sent: 0.0,
            })
        }
    }
}