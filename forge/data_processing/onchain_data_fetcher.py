# forge/data_processing/onchain_data_fetcher.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
from functools import lru_cache

CACHE_DURATION = 3600 # Cache for 1 hour

def _get_time_based_key():
    # Creates a key that invalidates automatically after CACHE_DURATION
    return time.time() // CACHE_DURATION

@lru_cache(maxsize=32)
def fetch_onchain_data_cached(start_date, end_date, freq, time_key=None):
    """
    Fetches real on-chain data from the DeFiLlama API.
    - Fetches historical Total Value Locked (TVL) of all DeFi protocols.
    - Fetches historical market cap of all stablecoins.
    """
    print("[On-Chain] Cache miss. Fetching real on-chain data from DeFiLlama...")
    try:
        # 1. Fetch Total DeFi TVL
        tvl_url = "https://api.llama.fi/charts"
        response_tvl = requests.get(tvl_url)
        response_tvl.raise_for_status()
        data_tvl = response_tvl.json()
        
        df_tvl = pd.DataFrame(data_tvl)
        df_tvl['date'] = pd.to_datetime(df_tvl['date'], unit='s')
        df_tvl.set_index('date', inplace=True)
        df_tvl.rename(columns={'totalLiquidityUSD': 'total_tvl_usd'}, inplace=True)
        
        # 2. Fetch Stablecoin Market Cap
        stables_url = "https://stablecoins.llama.fi/stablecoins?includePrices=false"
        response_stables = requests.get(stables_url)
        response_stables.raise_for_status()
        data_stables = response_stables.json().get('peggedAssets', [])
        
        stable_history = {}
        for stable in data_stables:
            if 'chainCirculating' in stable and 'peggedUSD' in stable['chainCirculating']:
                 for timestamp, circulating in stable['chainCirculating']['peggedUSD'].items():
                    dt = datetime.fromtimestamp(int(timestamp)).date() # Group by day
                    if dt not in stable_history: stable_history[dt] = 0
                    stable_history[dt] += circulating

        df_stables = pd.DataFrame.from_dict(stable_history, orient='index', columns=['stablecoin_supply'])
        df_stables.index = pd.to_datetime(df_stables.index)
        df_stables.sort_index(inplace=True)

        # 3. Combine and Resample
        df_onchain = df_tvl.join(df_stables, how='outer')
        df_onchain.index.rename('timestamp', inplace=True)
        
        df_onchain.ffill(inplace=True)
        
        df_onchain = df_onchain[(df_onchain.index >= start_date) & (df_onchain.index <= end_date)]
        if not df_onchain.empty:
            df_onchain = df_onchain.resample(freq).last().ffill()

        print("[On-Chain] Successfully fetched and processed DeFiLlama data.")
        return df_onchain

    except requests.exceptions.RequestException as e:
        print(f"[On-Chain] ERROR: Failed to fetch data from DeFiLlama: {e}")
        return pd.DataFrame(columns=['total_tvl_usd', 'stablecoin_supply'])

def fetch_onchain_data(start_date, end_date, freq='4H'):
    time_key = _get_time_based_key()
    # Call the internal cached function
    return fetch_onchain_data_cached(start_date, end_date, freq, time_key=time_key)