// src/microstructure.rs
use crate::lob::{OrderBook, GlobalBooks};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MicrostructureFeatures {
    pub symbol: String,
    pub voi_depth: f64, // Volume Order Imbalance over depth
    pub book_pressure_l1: f64, // Imbalance at L1
    pub spread: f64,
    pub mid_price: f64,
}

// Global structure to hold the latest features
pub type GlobalMicrostructure = Arc<RwLock<HashMap<String, MicrostructureFeatures>>>;

// Function to calculate features for a single OrderBook
pub fn calculate_features(book: &OrderBook) -> Option<MicrostructureFeatures> {
    let (best_bid_data, best_ask_data) = book.get_bbo();

    let (bid_p, bid_q_l1, ask_p, ask_q_l1) = match (best_bid_data, best_ask_data) {
        (Some((bp, bq)), Some((ap, aq))) => (bp, bq, ap, aq),
        _ => return None, // Insufficient data
    };

    let mid_price = (bid_p + ask_p) / 2.0;
    let spread = ask_p - bid_p;

    // 1. Book Pressure (L1 Imbalance)
    let book_pressure_l1 = if bid_q_l1 + ask_q_l1 > 0.0 {
        (bid_q_l1 - ask_q_l1) / (bid_q_l1 + ask_q_l1)
    } else { 0.0 };

    // 2. Volume Order Imbalance (VOI over depth)
    let depth = 10;
    // Use rev() for bids to iterate from highest price downwards
    let bid_volume: f64 = book.bids.iter().rev().take(depth).map(|(_, size)| size).sum();
    let ask_volume: f64 = book.asks.iter().take(depth).map(|(_, size)| size).sum();

    let voi_depth = if bid_volume + ask_volume > 0.0 {
        (bid_volume - ask_volume) / (bid_volume + ask_volume)
    } else {
        0.0
    };

    Some(MicrostructureFeatures {
        symbol: book.symbol.clone(),
        voi_depth,
        book_pressure_l1,
        spread,
        mid_price,
    })
}

// Helper function to update the global state (to be called periodically)
pub async fn compute_and_update_microstructure(books: &GlobalBooks, global_micro: &GlobalMicrostructure) {
    let books_read = books.read().await;
    let mut features_map = HashMap::new();

    // Iterate over the Arcs holding the RwLocks for each book
    for (symbol, book_lock) in books_read.iter() {
        let book = book_lock.read().await; // Acquire read lock on the specific book
        if let Some(feats) = calculate_features(&book) {
             features_map.insert(symbol.clone(), feats);
        }
    }

    // Update the global state in one write operation
    let mut micro_write = global_micro.write().await;
    *micro_write = features_map;
}
