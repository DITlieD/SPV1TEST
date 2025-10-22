# data_fetcher.py
import ccxt.pro as ccxt
import pandas as pd
from datetime import datetime, timedelta
import config
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

def load_historical_data(symbol, timeframe, limit=10000):
    """Loads historical OHLCV data from CSV."""
    try:
        base_symbol = symbol.split(':')[0].replace('/', '')
        file_path = os.path.join(config.DATA_DIR, f"{base_symbol}_{timeframe}_raw.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        return df.tail(limit)
    except Exception as e:
        logger.error(f"Error loading historical data for {symbol}: {e}")
        return pd.DataFrame()

async def get_market_data(symbol: str, timeframe: str = '15m', limit: int = 1000, exchange_instance=None):
    """Fetches OHLCV data for a given symbol and timeframe."""

    # FIX: For Bybit perpetuals, use the FULL symbol format (e.g., 'BTC/USDT:USDT')
    # Only clean for spot trading (no ':' in symbol)
    if ':' in symbol:
        # Perpetual contract - use full symbol
        query_symbol = symbol
    else:
        # Spot - use as is
        query_symbol = symbol

    print(f"[Data Fetcher] Fetching {limit} candles for {query_symbol} from Bybit...")

    # If no exchange instance is passed, create a new one for Bybit Sandbox
    exchange = None
    if exchange_instance:
        exchange = exchange_instance
    else:
        exchange = ccxt.bybit({
            'options': {
                'defaultType': 'swap', # Use swap for perpetual contracts
            },
        })
        exchange.set_sandbox_mode(config.BYBIT_USE_TESTNET)

    try:
        # --- ROBUSTNESS FIX: Add retry logic ---
        for attempt in range(3):
            try:
                ohlcv = await exchange.fetch_ohlcv(query_symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df # Success
            except Exception as e:
                if attempt < 2: # If not the last attempt
                    print(f"--- WARNING: Attempt {attempt + 1} failed for {query_symbol}. Retrying in 2s... Error: {e}")
                    await asyncio.sleep(2)
                else:
                    # This was the last attempt, handle final failure
                    print(f"--- ERROR fetching market data for {query_symbol} after 3 attempts: {e}")
                    return pd.DataFrame()
        # -----------------------------------------
    finally:
        if not exchange_instance and hasattr(exchange, 'close'):
            await exchange.close()

async def get_l2_order_book(symbol: str, limit: int = 100, exchange_instance=None):
    """Fetches the Level 2 order book for a given symbol."""

    # FIX: For Bybit perpetuals, use the FULL symbol format
    if ':' in symbol:
        query_symbol = symbol  # Perpetual contract - use full symbol
    else:
        query_symbol = symbol  # Spot - use as is

    print(f"[Data Fetcher] Fetching L2 order book for {query_symbol}...")

    exchange = None
    if exchange_instance:
        exchange = exchange_instance
    else:
        exchange = ccxt.bybit({
            'options': {
                'defaultType': 'swap',
            },
        })
        exchange.set_sandbox_mode(config.BYBIT_USE_TESTNET)

    try:
        order_book = await exchange.fetch_l2_order_book(query_symbol, limit=limit)
        return order_book
    except Exception as e:
        print(f"--- ERROR fetching L2 order book for {query_symbol}: {e}")
        return None
    finally:
        if not exchange_instance and hasattr(exchange, 'close'):
            await exchange.close()
