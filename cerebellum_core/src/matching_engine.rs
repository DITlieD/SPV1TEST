// src/matching_engine.rs
// NOTE: This module provides infrastructure for future slippage simulation features
#![allow(dead_code)]

use crate::lob::{GlobalBooks, OrderBook};
use anyhow::{Result, anyhow};

// This module simulates execution against the LOB for slippage/VWAP estimation.

#[derive(Debug, Clone, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

pub struct ExecutionSimulationReport {
    pub filled_quantity: f64,
    pub avg_price: f64, // VWAP
    pub slippage_bps: f64,
}

pub struct MatchingEngineSimulator;

impl MatchingEngineSimulator {
    pub async fn simulate_execution(
        symbol: &str,
        side: Side,
        quantity: f64,
        books: GlobalBooks
    ) -> Result<ExecutionSimulationReport> {
        let books_read = books.read().await;
        if let Some(book_lock) = books_read.get(symbol) {
            let book = book_lock.read().await;

            // Calculate mid-price
            let (bbo_bid, bbo_ask) = book.get_bbo();
            let mid_price = if let (Some((bid_p, _)), Some((ask_p, _))) = (bbo_bid, bbo_ask) {
                (bid_p + ask_p) / 2.0
            } else {
                return Err(anyhow!("Insufficient liquidity"));
            };

            let (filled_quantity, weighted_price_sum) = match side {
                Side::Buy => Self::match_buy(quantity, &book),
                Side::Sell => Self::match_sell(quantity, &book),
            };

            if filled_quantity > 0.0 {
                let avg_price = weighted_price_sum / filled_quantity;
                let price_diff = (avg_price - mid_price).abs();
                let slippage_bps = (price_diff / mid_price) * 10000.0;

                Ok(ExecutionSimulationReport {
                    filled_quantity,
                    avg_price,
                    slippage_bps,
                })
            } else {
                Err(anyhow!("Order could not be filled"))
            }

        } else {
            Err(anyhow!("Symbol {} not found", symbol))
        }
    }

    // Match buy against asks (lowest first)
    fn match_buy(mut quantity_remaining: f64, book: &OrderBook) -> (f64, f64) {
        let mut filled_quantity = 0.0;
        let mut weighted_price_sum = 0.0;

        for (price_of, qty_at_level) in book.asks.iter() {
            if quantity_remaining <= 0.0 { break; }
            let fill_at_level = quantity_remaining.min(*qty_at_level);
            filled_quantity += fill_at_level;
            weighted_price_sum += fill_at_level * price_of.0;
            quantity_remaining -= fill_at_level;
        }
        (filled_quantity, weighted_price_sum)
    }

    // Match sell against bids (highest first - using rev())
    fn match_sell(mut quantity_remaining: f64, book: &OrderBook) -> (f64, f64) {
        let mut filled_quantity = 0.0;
        let mut weighted_price_sum = 0.0;

        for (price_of, qty_at_level) in book.bids.iter().rev() {
            if quantity_remaining <= 0.0 { break; }
            let fill_at_level = quantity_remaining.min(*qty_at_level);
            filled_quantity += fill_at_level;
            weighted_price_sum += fill_at_level * price_of.0;
            quantity_remaining -= fill_at_level;
        }
        (filled_quantity, weighted_price_sum)
    }
}
