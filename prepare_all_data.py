import os
import ccxt.pro as ccxt # Use the async-compatible version of ccxt
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import time
from config import ASSET_UNIVERSE, TIMEFRAMES, BYBIT_USE_TESTNET

DATA_DIR = 'data'
DAYS_TO_FETCH = 90
DATA_MAX_AGE_HOURS = 168  # 7 days - skip re-fetch during testing
# Use a semaphore to limit concurrent requests to avoid rate limiting
MAX_CONCURRENT_DOWNLOADS = 10

async def download_one_symbol(symbol: str, timeframe: str, exchange, semaphore):
    """
    Asynchronously downloads and saves data for a single symbol/timeframe pair
    if it's missing or outdated.
    """
    async with semaphore: # Acquire a spot in the semaphore
        sanitized_symbol = symbol.split(':')[0].replace('/', '')
        filename = f"{sanitized_symbol}_{timeframe}_raw.csv"
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath):
            file_mod_time = os.path.getmtime(filepath)
            hours_since_mod = (time.time() - file_mod_time) / 3600
            if hours_since_mod < DATA_MAX_AGE_HOURS:
                print(f"[DataPrep] -> SKIPPING {symbol} {timeframe} (recent, {hours_since_mod:.1f}h old)")
                return

        try:
            print(f"[DataPrep] -> FETCHING {symbol} {timeframe}...")
            since = exchange.parse8601((datetime.utcnow() - timedelta(days=DAYS_TO_FETCH)).isoformat())
            
            all_ohlcv = []
            last_since = None
            while True:
                if last_since == since:
                    print(f"💥 ERROR fetching {symbol} {timeframe}: Timestamp not advancing, breaking loop.")
                    break
                last_since = since

                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
            
            if not all_ohlcv:
                print(f"[DataPrep] -> No data returned for {symbol} {timeframe}.")
                return

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.to_csv(filepath)
            print(f"[DataPrep] -> SAVED {len(df)} bars for {symbol} {timeframe}")

        except Exception as e:
            print(f"💥 ERROR fetching {symbol} {timeframe}: {e}")

async def main():
    """Main async function to orchestrate concurrent data downloads."""
    print(f"\n[DataPrep] --- Starting Concurrent Data Preparation (Last {DAYS_TO_FETCH} Days) ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    exchange = ccxt.bybit({
        'options': {
            'defaultType': 'swap',
        },
    })
    exchange.set_sandbox_mode(BYBIT_USE_TESTNET)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    # Create a list of all download tasks
    tasks = []
    for asset in ASSET_UNIVERSE:
        for tf_name, tf_value in TIMEFRAMES.items():
            tasks.append(download_one_symbol(asset, tf_value, exchange, semaphore))
            
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    
    await exchange.close()
    print("\n[DataPrep] --- Data Preparation Finished ---")

if __name__ == "__main__":
    # Use asyncio.run() to execute the main async function
    asyncio.run(main())