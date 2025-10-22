# data_processing_v2.py (V5.3 - Final)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pandas_ta')

import pandas as pd
import numpy as np
import asyncio
import config
import ccxt.pro as ccxt # Add missing import
from data_fetcher import get_market_data # <-- FIX: ADDED MISSING IMPORT
from forge.data_processing.onchain_data_fetcher import fetch_onchain_data
from forge.data_processing.microstructure_features import get_trade_flow_imbalance
import pandas_ta as ta

from forge.data_processing.labeling import create_categorical_labels
import torch
from forge.data_processing.graph_builder import GraphBuilder
from forge.models.gnn_intermarket import GAT_Intermarket

def add_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches, processes, and merges on-chain data features.
    If it fails, it logs a warning and returns the original dataframe.
    """
    if df.empty:
        return df

    try:
        # --- FIX for GroupBy index issue ---
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().set_index('timestamp')

        start_date = df.index.min()
        end_date = df.index.max()

        df_onchain = fetch_onchain_data(start_date, end_date, freq=pd.infer_freq(df.index))
        
        if df_onchain.empty:
            warnings.warn("On-chain data fetch returned an empty dataframe. Skipping feature merge.")
            return df

        # --- Feature Engineering for On-Chain Data ---
        df_onchain['tvl_momentum'] = df_onchain['total_tvl_usd'].rolling(window=7).mean().pct_change()
        df_onchain['stablecoin_supply_momentum'] = df_onchain['stablecoin_supply'].rolling(window=7).mean().pct_change()
        df_onchain['tvl_zscore'] = (df_onchain['total_tvl_usd'] - df_onchain['total_tvl_usd'].rolling(window=30).mean()) / df_onchain['total_tvl_usd'].rolling(window=30).std()
        df_onchain['stablecoin_supply_zscore'] = (df_onchain['stablecoin_supply'] - df_onchain['stablecoin_supply'].rolling(window=30).mean()) / df_onchain['stablecoin_supply'].rolling(window=30).std()

        # FIX: Use merge_asof to prevent data leakage (lookahead bias)
        # direction='backward' ensures we only use on-chain data available AT or BEFORE the timestamp
        df.sort_index(inplace=True)
        df_onchain.sort_index(inplace=True)

        df = pd.merge_asof(
            df,
            df_onchain,
            left_index=True,
            right_index=True,
            direction='backward',  # CRITICAL: Ensures no lookahead bias
            allow_exact_matches=True
        )

        # --- DEFINITIVE FIX for data sanitization ---
        # Back-fill and then forward-fill to propagate the first valid on-chain data backward, then fill any gaps.
        onchain_cols = df_onchain.columns
        df[onchain_cols] = df[onchain_cols].bfill().ffill()
        # ---------------------------------------------

    except Exception as e:
        warnings.warn(f"Failed to add on-chain features. Proceeding without them. Error: {e}")
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)

    # Create ATRr (ATR ratio = ATR / close) for compatibility with old models
    if 'ATR_14' in df.columns:
        df['ATRr_14'] = df['ATR_14'] / (df['close'] + 1e-9)

    return df

def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
    df['momentum_20'] = df['close'].pct_change(periods=20)
    df['skewness_20'] = df['close'].pct_change().rolling(window=20).skew()
    return df

def add_all_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    if symbol is None and hasattr(df, 'name'):
        symbol = df.name

    df = add_technical_indicators(df)
    df = add_market_structure_features(df)
    # df = add_onchain_features(df)
    return df

def process_and_align_mtfa(data_dict: dict, app_config, exchange=None) -> pd.DataFrame:
    """
    MTFA (Multi-Timeframe Analysis) Processing Pipeline

    Processes features for multiple timeframes independently and aligns HTF features onto the LTF.
    This provides noise reduction and trend context for 1m execution.

    Args:
        data_dict: Dictionary containing raw dataframes keyed by timeframe name (e.g., {'1m': df1, '15m': df15, '1h': df60})
        app_config: Configuration object with TIMEFRAMES
        exchange: Exchange object passed to FeatureFactory (None in Forge worker context)

    Returns:
        DataFrame with 1m bars enriched with 15m and 1h features (e.g., RSI_14_15m, RSI_14_1h)

    Philosophy:
        - 1m provides execution speed
        - 15m provides tactical context
        - 1h provides strategic trend
        - GP autonomously learns which combinations create edge
    """
    import logging
    from forge.data_processing.feature_factory import FeatureFactory

    logger = logging.getLogger("MTFA")
    processed_tfs = {}

    # 1. Identify LTF (Lowest Timeframe) robustly
    timeframes = list(data_dict.keys())
    try:
        # Identify the lowest timeframe (e.g., '1m') based on time delta
        ltf_name = min(timeframes, key=lambda tf: pd.to_timedelta(tf).total_seconds())
    except ValueError as e:
        logger.error(f"[MTFA] ERROR: Could not parse timeframes {timeframes}: {e}. Aborting MTFA.")
        return pd.DataFrame()

    logger.info(f"[MTFA] LTF identified: {ltf_name}")

    # 2. Process each timeframe independently
    for tf_name, df_raw in data_dict.items():
        if df_raw.empty:
            logger.warning(f"[MTFA] Skipping empty dataframe for {tf_name}")
            continue

        logger.info(f"[MTFA] Processing features for timeframe: {tf_name} ({len(df_raw)} bars)")

        # Initialize FeatureFactory (Requires dictionary structure for input)
        # We use a temporary key 'MTFA_TEMP' for processing.
        ff = FeatureFactory({'MTFA_TEMP': df_raw.copy()}, logger=logger)
        # exchange=None is passed if we are in the Forge worker context (determined by the caller)
        # VELOCITY FIX: Skip Feature Synthesizer during MTFA intermediate timeframe processing
        # Evolved features should only be created on the final merged data, not intermediate timeframes
        processed_data = ff.create_all_features(app_config, exchange, skip_feature_evolution=True)
        processed_df = processed_data.get('MTFA_TEMP')

        if processed_df is None or processed_df.empty:
            logger.warning(f"[MTFA] FeatureFactory returned empty for {tf_name}")
            continue

        # 3. Rename features and handle OHLCV
        if tf_name != ltf_name:
            # Identify features (excluding OHLCV)
            feature_cols = [col for col in processed_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

            # Suffix HTF features (e.g., 'RSI_14' -> 'RSI_14_15m')
            rename_map = {col: f"{col}_{tf_name}" for col in feature_cols}

            # Keep only the features, discard the HTF OHLCV
            processed_tfs[tf_name] = processed_df[feature_cols].rename(columns=rename_map)
            logger.info(f"[MTFA] Renamed {len(feature_cols)} features for {tf_name}")
        else:
            # Keep the LTF dataframe as is (including OHLCV)
            processed_tfs[tf_name] = processed_df
            logger.info(f"[MTFA] Kept LTF {tf_name} dataframe intact")

    if ltf_name not in processed_tfs:
        logger.error(f"[MTFA] LTF data {ltf_name} missing after processing.")
        return pd.DataFrame()

    # 4. Alignment and Injection using merge_asof (causality-preserving)
    ltf_df = processed_tfs[ltf_name].copy()
    ltf_df.sort_index(inplace=True)  # Required for merge_asof

    for tf_name, htf_df in processed_tfs.items():
        if tf_name == ltf_name:
            continue

        htf_df.sort_index(inplace=True)
        logger.info(f"[MTFA] Aligning {tf_name} features onto {ltf_name}...")

        # CRITICAL: Use merge_asof for causality-preserving alignment
        # direction='backward' ensures we only use data available AT or BEFORE the LTF timestamp.
        # This prevents lookahead bias!
        ltf_df = pd.merge_asof(
            ltf_df,
            htf_df,
            left_index=True,
            right_index=True,
            direction='backward',
            allow_exact_matches=True
        )
        logger.info(f"[MTFA] Successfully aligned {len(htf_df.columns)} features from {tf_name}")

    # 5. Final Sanitization
    ltf_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop initial rows where HTF data wasn't available yet.
    rows_before = len(ltf_df)
    ltf_df.dropna(inplace=True)
    rows_after = len(ltf_df)

    logger.info(f"[MTFA] Dropped {rows_before - rows_after} rows with incomplete HTF data")
    logger.info(f"[MTFA] Final dataset: {rows_after} bars with {len(ltf_df.columns)} features")

    return ltf_df

async def run_graph_feature_pipeline(app_config, exchange):
    """The new master pipeline for creating graph-based features for all assets."""
    print("[GNN Pipeline] Starting feature generation...")
    limit = 500
    
    # Create ONE exchange instance for all concurrent calls to prevent session errors
    gnn_exchange = None
    try:
        gnn_exchange = ccxt.bybit({
            'options': {
                'defaultType': 'swap',
            },
        })
        gnn_exchange.set_sandbox_mode(config.BYBIT_USE_TESTNET)
        
        tasks = {
            asset: get_market_data(asset, app_config.TIMEFRAMES['tactical'], limit=limit, exchange_instance=gnn_exchange)
            for asset in app_config.ASSET_UNIVERSE
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    finally:
        # Ensure the single exchange instance is always closed
        if gnn_exchange:
            await gnn_exchange.close()

    # --- DEFINITIVE FIX for data fetching failures ---
    # Run all tasks concurrently and continue even if some fail
    # results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # Filter out both exceptions and empty dataframes
    all_dfs = {}
    for asset, result in zip(tasks.keys(), results):
        if isinstance(result, pd.DataFrame) and not result.empty:
            all_dfs[asset] = result
        else:
            print(f"[GNN Pipeline] WARNING: Failed to fetch or process data for {asset}. Excluding from this graph build.")

    if not all_dfs:
        print("[GNN Pipeline] ERROR: Failed to fetch data for ALL assets. Skipping GNN.")
        return pd.DataFrame()

    # Proceed with the assets we successfully have data for
    successful_assets = list(all_dfs.keys())
    combined_df = pd.concat(all_dfs.values(), keys=successful_assets, names=['symbol', 'timestamp'])
    combined_features = combined_df.groupby(level='symbol', group_keys=False).apply(add_all_features)

    # FIX: Be less aggressive with dropna. Forward fill features across time for each symbol.
    combined_features = combined_features.groupby(level='symbol', group_keys=False).ffill()
    combined_features.dropna(inplace=True)

    # Check if dataframe is empty after processing
    if combined_features.empty:
        print("[GNN Pipeline] ERROR: Dataframe empty after feature processing and cleaning. Skipping GNN.")
        return pd.DataFrame()

    latest_features = combined_features.groupby(level='symbol').tail(1)
    builder = GraphBuilder(assets=successful_assets, all_historical_dfs=all_dfs) # Pass historical data
    market_graph = builder.create_graph_from_features(latest_features)

    if market_graph.num_nodes == 0:
        print("[GNN Pipeline] No nodes in graph. Skipping GNN.")
        return pd.DataFrame()

    in_channels = market_graph.num_node_features
    gat_model = GAT_Intermarket(in_channels=in_channels, hidden_channels=32, out_channels=8)
    gat_model.eval()
    with torch.no_grad():
        graph_embeddings = gat_model(market_graph)

    embedding_df = pd.DataFrame(graph_embeddings.numpy(), index=successful_assets, columns=[f'gat_{i}' for i in range(8)])

    # --- DEFINITIVE FIX V3: Avoid _x suffix collision ---
    if 'symbol' in latest_features.columns:
        latest_features = latest_features.drop(columns=['symbol'])

    latest_features_reset = latest_features.reset_index()
    embedding_df_reset = embedding_df.reset_index().rename(columns={'index': 'symbol'})

    # CRITICAL FIX: Use left merge and specify suffixes to avoid _x/_y
    # Only keep columns from embedding_df (gat_*), not any duplicates
    final_df = pd.merge(
        latest_features_reset,
        embedding_df_reset,
        on='symbol',
        how='left',
        suffixes=('', '_gnn')  # Keep original names, add _gnn only to conflicts
    ).set_index(['symbol', 'timestamp'])
    
    print("[GNN Pipeline] GNN feature generation complete.")
    return final_df

def get_specialist_labels(df, atr_col, module_type='reversion', window=20, std_dev=2):
    if module_type == 'reversion':
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)

        conditions = [
            (df['close'] < lower_band),
            (df['close'] > upper_band)
        ]
        choices = [1, -1]
        
        labels = np.select(conditions, choices, default=0)
        return pd.Series(labels, index=df.index)
    else:
        return pd.Series(0, index=df.index)

async def get_aligned_mtf_data(symbol: str, app_config, exchange):
    limit_tactical = 2000
    limit_strategic = limit_tactical // 16
    limit_micro = limit_tactical * 15

    tactical_task = get_market_data(symbol, app_config.TIMEFRAMES['tactical'], limit_tactical, exchange)
    strategic_task = get_market_data(symbol, app_config.TIMEFRAMES['strategic'], limit_strategic, exchange)
    micro_task = get_market_data(symbol, app_config.TIMEFRAMES['microstructure'], limit_micro, exchange)

    df_tactical_raw, df_strategic_raw, df_micro_raw = await asyncio.gather(tactical_task, strategic_task, micro_task)

    if df_tactical_raw.empty or df_strategic_raw.empty or df_micro_raw.empty:
        return pd.DataFrame()

    df_tactical_features = add_all_features(df_tactical_raw.copy()).add_prefix('tactical_')
    df_strategic_features = add_all_features(df_strategic_raw.copy()).add_prefix('strategic_')
    tfi_series = get_trade_flow_imbalance(df_micro_raw)

    df_tactical_features.reset_index(inplace=True)
    df_strategic_features.reset_index(inplace=True)
    tfi_df = tfi_series.reset_index()

    aligned_df = pd.merge_asof(df_tactical_features, df_strategic_features, on='timestamp', direction='backward')
    aligned_df = pd.merge_asof(aligned_df, tfi_df, on='timestamp', direction='backward')
    
    aligned_df.set_index('timestamp', inplace=True)
    aligned_df.attrs['symbol'] = symbol
    return aligned_df
