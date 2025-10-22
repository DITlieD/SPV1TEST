// src/storage.rs
// NOTE: This module provides infrastructure for future tick data persistence features
#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use arrow::array::{Float64Array, Int64Array, StringArray, RecordBatch};
use arrow::datatypes::{Schema, Field, DataType};
use chrono::Utc;
use anyhow::Result;
use log::info;

// Simplified structure for storing ticks/executions
#[derive(Debug, Clone)]
pub struct StoredData {
    pub timestamp: i64,
    pub symbol: String,
    pub event_type: String, // e.g., "trade", "execution", "l2_update"
    pub price: f64,
    pub quantity: f64,
}

pub struct DataStorage {
    schema: Arc<Schema>,
}

impl DataStorage {
    pub fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("event_type", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
            Field::new("quantity", DataType::Float64, false),
        ]));
        // Ensure the directory exists
        std::fs::create_dir_all("data/cerebellum_store").ok();
        Self { schema }
    }

    // Writes a batch of data. This should be called by a dedicated background task.
    pub fn write_batch(&self, data_batch: Vec<StoredData>) -> Result<()> {
        if data_batch.is_empty() {
            return Ok(());
        }

        // Prepare Arrow arrays
        let timestamp_array = Int64Array::from(data_batch.iter().map(|t| t.timestamp).collect::<Vec<_>>());
        let symbol_array = StringArray::from(data_batch.iter().map(|t| t.symbol.clone()).collect::<Vec<_>>());
        let event_type_array = StringArray::from(data_batch.iter().map(|t| t.event_type.clone()).collect::<Vec<_>>());
        let price_array = Float64Array::from(data_batch.iter().map(|t| t.price).collect::<Vec<_>>());
        let quantity_array = Float64Array::from(data_batch.iter().map(|t| t.quantity).collect::<Vec<_>>());

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(timestamp_array),
                Arc::new(symbol_array),
                Arc::new(event_type_array),
                Arc::new(price_array),
                Arc::new(quantity_array),
            ],
        )?;

        // Generate filename (partitioned by hour for efficiency)
        let datetime_str = Utc::now().format("%Y%m%d_%H").to_string();
        let batch_id = Utc::now().timestamp_subsec_millis();
        let filename = format!("data/cerebellum_store/data_{}_{}.parquet", datetime_str, batch_id);

        let file = File::create(&filename)?;

        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, self.schema.clone(), Some(props))?;

        writer.write(&batch)?;
        writer.close()?;

        info!("[Storage] Wrote batch of {} records to {}", data_batch.len(), filename);
        Ok(())
    }
}
