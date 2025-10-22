"""
Macro-Causality Engine (Influence Mapper) - Global Market Information Flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Optional dependency for Permutation Entropy
try:
    import ordpy
except ImportError:
    logging.warning("ordpy not installed. Permutation Entropy will use a placeholder.")
    ordpy = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# ============================================================================
# GAT Model Definition
# ============================================================================
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ============================================================================
# OFFLINE/PROXY FUNCTIONS (from updatev16.txt)
# ============================================================================

def calculate_permutation_entropy(series, order=3, delay=1):
    """Calculates the Permutation Entropy (PE)."""
    if ordpy:
        try:
            data = series.dropna().values
            if len(data) < order + delay: return np.nan
            return ordpy.permutation_entropy(data, dx=order, taux=delay, normalized=True)
        except Exception as e:
            logger.debug(f"PE calculation error: {e}")
            return np.nan
    else:
        return series.std()

def calculate_influence_map(cross_asset_data: dict, target_symbol: str, config: dict) -> pd.DataFrame:
    """
    Calculates the Influence Map for offline training using proxies.
    """
    window = config.get('im_entropy_window', 60)
    logger.info(f"Calculating Influence Map for {target_symbol} (Window: {window})...")
    
    target_df = cross_asset_data.get(target_symbol)
    if target_df is None: return pd.DataFrame()

    aligned_data = pd.DataFrame(index=target_df.index)
    if 'close' in target_df.columns:
        aligned_data[target_symbol] = target_df['close']

    for symbol, df in cross_asset_data.items():
        if symbol == target_symbol: continue
        if 'close' in df.columns:
             aligned_data = pd.merge_asof(aligned_data.sort_index(), df[['close']].rename(columns={'close': symbol}).sort_index(), 
                                               left_index=True, right_index=True, direction='backward')

    influence_features = pd.DataFrame(index=target_df.index)
    influence_features['IM_PE_Target'] = aligned_data[target_symbol].rolling(window=window).apply(calculate_permutation_entropy, raw=False)
    
    dominant_asset = next((k for k in aligned_data.columns if 'BTC' in k), None)
    if dominant_asset and target_symbol != dominant_asset:
        influence_features['IM_TE_Incoming_Proxy'] = aligned_data[target_symbol].rolling(window=window).corr(aligned_data[dominant_asset])
    else:
        influence_features['IM_TE_Incoming_Proxy'] = 0.0

    influence_features['IM_Predicted_Entropy_Delta'] = 0.0
    return influence_features

# ============================================================================
# REAL-TIME PIPELINE COMPONENTS (Restored)
# ============================================================================

class PermutationEntropyCalculator:
    def __init__(self, window_size: int = 100, order: int = 4, delay: int = 1):
        self.window_size = window_size
        self.order = order
        self.delay = delay
        self.price_buffers: Dict[str, deque] = {}

    def initialize_asset(self, symbol: str):
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=self.window_size)

    def calculate_all_assets(self, prices: Dict[str, float]) -> Dict[str, float]:
        pe_values = {}
        for symbol, price in prices.items():
            self.initialize_asset(symbol)
            self.price_buffers[symbol].append(price)
            prices_arr = np.array(self.price_buffers[symbol])
            returns = np.diff(np.log(prices_arr + 1e-10))
            pe_values[symbol] = calculate_permutation_entropy(returns, self.order, self.delay)
        return pe_values

class TransferEntropyCalculator:
    def __init__(self, window_size: int = 100, update_frequency: int = 20):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.return_buffers: Dict[str, deque] = {}
        self.causal_matrix: Optional[pd.DataFrame] = None
        self.update_count = 0

    def update_returns(self, returns: Dict[str, float]):
        for symbol, ret in returns.items():
            if symbol not in self.return_buffers:
                self.return_buffers[symbol] = deque(maxlen=self.window_size)
            self.return_buffers[symbol].append(ret)
        self.update_count += 1

    def calculate_causal_matrix(self, asset_list: List[str], parallel: bool = False) -> pd.DataFrame:
        # Placeholder logic, as the full implementation is complex
        self.causal_matrix = pd.DataFrame(np.random.rand(len(asset_list), len(asset_list)), index=asset_list, columns=asset_list)
        return self.causal_matrix

class InfluenceMapper:
    def __init__(self, pe_calculator, te_calculator, gat_config=None):
        self.pe_calculator = pe_calculator
        self.te_calculator = te_calculator
        
        if gat_config is None:
            # Node features are: PE, return, volatility
            gat_config = {'in_channels': 3, 'hidden_channels': 16, 'out_channels': 1, 'heads': 2}
        
        self.gat_model = GAT(
            in_channels=gat_config['in_channels'],
            hidden_channels=gat_config['hidden_channels'],
            out_channels=gat_config['out_channels'],
            heads=gat_config['heads']
        )
        self.optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=0.005, weight_decay=5e-4)

    def create_graph_snapshot(self, asset_list, pe_values, returns, volatility, causal_matrix):
        num_nodes = len(asset_list)
        
        # Node features: PE, return, volatility
        x = torch.tensor([[pe_values.get(asset, 0.5), returns.get(asset, 0.0), volatility.get(asset, 0.01)] for asset in asset_list], dtype=torch.float)

        # Edge index and attributes from causal matrix
        edge_index = []
        edge_attr = []
        
        causal_matrix_df = pd.DataFrame(causal_matrix)
        if causal_matrix_df.columns.tolist() != asset_list or causal_matrix_df.index.tolist() != asset_list:
             causal_matrix_df = pd.DataFrame(causal_matrix, index=asset_list, columns=asset_list)


        for i, source in enumerate(asset_list):
            for j, target in enumerate(asset_list):
                if i != j and causal_matrix_df.loc[source, target] > 0.1: # Threshold
                    edge_index.append([i, j])
                    edge_attr.append(causal_matrix_df.loc[source, target])
        
        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def train_gat(self, snapshots, epochs=50, batch_size=32):
        if not snapshots:
            logger.warning("No snapshots provided for GAT training. Skipping.")
            return
            
        loader = DataLoader(snapshots, batch_size=batch_size, shuffle=True)
        self.gat_model.train()
        
        logger.info(f"Starting GAT training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()
                out = self.gat_model(batch)
                loss = F.mse_loss(out, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        logger.info("GAT training finished.")

    def predict_influence(self, asset_list, pe_values, returns, volatility, causal_matrix):
        self.gat_model.eval()
        graph = self.create_graph_snapshot(asset_list, pe_values, returns, volatility, causal_matrix)
        
        if graph.edge_index.shape[1] == 0:
            predicted_delta = torch.zeros((len(asset_list), 1))
        else:
            with torch.no_grad():
                predicted_delta = self.gat_model(graph)

        results = {}
        causal_matrix_df = pd.DataFrame(causal_matrix, index=asset_list, columns=asset_list)
        for i, asset in enumerate(asset_list):
            incoming = causal_matrix_df.loc[:, asset].sum() - causal_matrix_df.loc[asset, asset]
            outgoing = causal_matrix_df.loc[asset, :].sum() - causal_matrix_df.loc[asset, asset]
            results[asset] = {
                'influence_incoming': incoming,
                'influence_outgoing': outgoing,
                'predicted_entropy_delta': predicted_delta[i].item()
            }
        return results

class InfluenceMapperPipeline:
    """
    Real-time Influence Mapper pipeline
    """
    def __init__(
        self,
        asset_list: List[str],
        pe_window: int = 100,
        te_window: int = 100,
        te_update_freq: int = 20,
        gat_config: Dict[str, any] = None
    ):
        self.asset_list = asset_list
        self.pe_calculator = PermutationEntropyCalculator(window_size=pe_window)
        self.te_calculator = TransferEntropyCalculator(window_size=te_window, update_frequency=te_update_freq)
        self.influence_mapper = InfluenceMapper(self.pe_calculator, self.te_calculator, gat_config)
        self.last_prices: Dict[str, float] = {}
        self.update_count = 0

    def process_update(
        self,
        prices: Dict[str, float],
        volatility: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        returns = {}
        for asset in self.asset_list:
            if asset in self.last_prices and asset in prices and self.last_prices[asset] > 0:
                returns[asset] = np.log(prices[asset] / self.last_prices[asset])
            else:
                returns[asset] = 0.0
        self.last_prices = prices.copy()

        pe_values = self.pe_calculator.calculate_all_assets(prices)
        self.te_calculator.update_returns(returns)

        if self.update_count % self.te_calculator.update_frequency == 0:
            causal_matrix = self.te_calculator.calculate_causal_matrix(asset_list=self.asset_list)
        else:
            causal_matrix = self.te_calculator.causal_matrix
            if causal_matrix is None:
                causal_matrix = pd.DataFrame(0.0, index=self.asset_list, columns=self.asset_list)

        if volatility is None:
            volatility = {asset: abs(ret) for asset, ret in returns.items()}

        influence_scores = self.influence_mapper.predict_influence(
            self.asset_list, pe_values, returns, volatility, causal_matrix
        )
        self.update_count += 1
        return influence_scores
