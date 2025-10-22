// src/lob.rs
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use futures_util::{StreamExt, SinkExt};
// SECURITY FIX: Use secure connect_async (no certificate bypass)
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;
use log::{info, error, warn, debug};
use std::collections::{HashMap, BTreeMap};
use ordered_float::OrderedFloat;
use anyhow;

// Fine-grained locking: HashMap holds Arcs to individual book RwLocks.
pub type GlobalBooks = Arc<RwLock<HashMap<String, Arc<RwLock<OrderBook>>>>>;
type PriceLevel = OrderedFloat<f64>;

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    // BTreeMap sorts ascending. Use last_key_value() for best bid, first_key_value() for best ask.
    pub bids: BTreeMap<PriceLevel, f64>,
    pub asks: BTreeMap<PriceLevel, f64>,
    pub last_update_id: u64,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
        }
    }

    fn apply_snapshot(&mut self, snapshot: &L2Data) {
        self.bids.clear();
        self.asks.clear();
        self.last_update_id = snapshot.update_id;
        self.update_levels(&snapshot.bids, true);
        self.update_levels(&snapshot.asks, false);
        debug!("[LOB] Snapshot applied for {}. UpdateID: {}", self.symbol, self.last_update_id);
    }

    fn apply_delta(&mut self, delta: &L2Data) {
        if delta.update_id <= self.last_update_id {
            warn!("[LOB] Outdated delta for {}. Current: {}, Received: {}", self.symbol, self.last_update_id, delta.update_id);
            return;
        }
        self.last_update_id = delta.update_id;
        self.update_levels(&delta.bids, true);
        self.update_levels(&delta.asks, false);
    }

    fn update_levels(&mut self, levels: &[[String; 2]], is_bid: bool) {
        let book = if is_bid { &mut self.bids } else { &mut self.asks };
        for level in levels {
            if let (Ok(price_f64), Ok(size)) = (level[0].parse::<f64>(), level[1].parse::<f64>()) {
                let price = OrderedFloat(price_f64);
                if size == 0.0 {
                    book.remove(&price);
                } else {
                    book.insert(price, size);
                }
            }
        }
    }

    pub fn get_bbo(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        let best_bid = self.bids.last_key_value().map(|(p, s)| (p.0, *s));
        let best_ask = self.asks.first_key_value().map(|(p, s)| (p.0, *s));
        (best_bid, best_ask)
    }
}

// Structures for deserializing Bybit V5 L2 WebSocket messages
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct WsMessage {
    #[allow(dead_code)]
    topic: Option<String>,  // CRITICAL FIX: Optional for subscription responses
    #[serde(rename = "type")]
    msg_type: Option<String>,  // CRITICAL FIX: Optional for subscription responses
    data: Option<L2Data>,  // CRITICAL FIX: Optional (not present in subscription confirmations)
}

#[derive(Deserialize, Debug)]
struct L2Data {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "u")]
    update_id: u64,
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,
}

// CRITICAL FIX: Use linear (perpetual futures) WebSocket, not spot
const BYBIT_WS_URL: &str = "wss://stream.bybit.com/v5/public/linear";

pub async fn market_data_handler(books: GlobalBooks, assets: Vec<String>) {
    if assets.is_empty() { return; }
    let url = Url::parse(BYBIT_WS_URL).expect("Invalid WebSocket URL");

    // SECURITY FIX: Connect securely (no certificate bypass)
    let (mut ws_stream, _) = connect_async(url).await.expect("Failed to connect to WebSocket");
    info!("[MDH] WebSocket connected securely.");

    // Initialize local book structures within the global state
    {
        let mut books_write = books.write().await;
        for asset in &assets {
             books_write.entry(asset.clone()).or_insert_with(|| Arc::new(RwLock::new(OrderBook::new(asset.clone()))));
        }
    }

    // Subscribe to L2 streams (orderbook.50)
    let subscriptions: Vec<String> = assets.iter().map(|s| format!("orderbook.50.{}", s)).collect();
    let subscribe_msg = serde_json::json!({
        "op": "subscribe",
        "args": subscriptions
    });

    ws_stream.send(Message::Text(subscribe_msg.to_string())).await.expect("Failed to subscribe");
    info!("[MDH] Subscribed to L2 feeds (depth 50).");

    while let Some(msg) = ws_stream.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Err(e) = process_message(text, &books).await {
                    error!("[MDH] Error processing message: {}", e);
                }
            },
            Ok(Message::Ping(data)) => { ws_stream.send(Message::Pong(data)).await.ok(); },
            Err(e) => { error!("[MDH] WebSocket error: {}", e); break; }
            _ => {}
        }
    }
}

async fn process_message(text: String, books: &GlobalBooks) -> Result<(), anyhow::Error> {
    // CRITICAL FIX: Skip non-orderbook messages (subscription confirmations, pings, etc.)
    if !text.contains("orderbook.50.") {
        debug!("[MDH] Skipping non-orderbook message");
        return Ok(());
    }

    let msg: WsMessage = serde_json::from_str(&text)?;

    // CRITICAL FIX: Only process if we have data (skip subscription responses)
    if let (Some(msg_type), Some(data)) = (msg.msg_type, msg.data) {
        let symbol = data.symbol.clone();

        // Access the specific book lock efficiently
        let books_read = books.read().await;
        if let Some(book_lock) = books_read.get(&symbol) {
            let mut book = book_lock.write().await; // Acquire write lock for the specific book
            match msg_type.as_str() {
                "snapshot" => book.apply_snapshot(&data),
                "delta" => book.apply_delta(&data),
                _ => (),
            }
        }
    }
    Ok(())
}
