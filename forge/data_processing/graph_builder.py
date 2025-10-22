import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

class GraphBuilder:
    """
    Constructs a dynamic graph representation of the market, where assets are nodes
    and the relationships between them are edges.
    """
    def __init__(self, assets: list, all_historical_dfs: dict, correlation_threshold=0.1, corr_window=30):
        self.assets = assets
        self.asset_map = {asset: i for i, asset in enumerate(assets)}
        self.all_historical_dfs = all_historical_dfs
        self.correlation_threshold = correlation_threshold
        self.corr_window = corr_window

    def _calculate_real_correlation_matrix(self) -> np.ndarray:
        """Calculates the rolling correlation matrix from historical close prices."""
        # Combine close prices from all assets into a single DataFrame
        close_prices = pd.concat(
            {asset: df['close'] for asset, df in self.all_historical_dfs.items()},
            axis=1
        )
        # Forward-fill missing values and then drop any remaining NaNs
        close_prices.ffill(inplace=True)
        close_prices.dropna(inplace=True)

        if len(close_prices) < self.corr_window:
            print(f"[GraphBuilder] Warning: Not enough data ({len(close_prices)} points) for correlation window ({self.corr_window}). Using available data.")
        
        # Calculate percentage returns
        returns = close_prices.pct_change()
        
        # Calculate rolling correlation. The result for each window is a MultiIndex DataFrame.
        # We take the last valid entry, which corresponds to the correlation of the most recent window.
        rolling_corr = returns.rolling(window=self.corr_window).corr()
        
        # Get the correlation matrix for the last timestamp by selecting the last valid index
        last_timestamp = rolling_corr.index.get_level_values(0)[-1]
        latest_corr_matrix = rolling_corr.loc[last_timestamp]
        
        # Reorder columns and index to match the asset_map for consistency
        ordered_assets = list(self.asset_map.keys())
        latest_corr_matrix = latest_corr_matrix.reindex(index=ordered_assets, columns=ordered_assets)
        
        return latest_corr_matrix.values

    def create_graph_from_features(self, all_features_df: pd.DataFrame) -> Data:
        """
        Builds a graph for a single time step.
        """
        num_nodes = len(self.assets)
        
        numeric_features_df = all_features_df.select_dtypes(include=np.number)
        
        node_features = torch.zeros((num_nodes, len(numeric_features_df.columns)))
        for asset, i in self.asset_map.items():
            if asset in numeric_features_df.index.get_level_values('symbol'):
                asset_features = numeric_features_df.xs(asset, level='symbol')
                node_features[i] = torch.tensor(asset_features.values, dtype=torch.float32)

        # Use the real correlation matrix
        corr_matrix = self._calculate_real_correlation_matrix()
        
        edge_list = []
        edge_weights = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                correlation = corr_matrix[i, j]
                if not np.isnan(correlation) and abs(correlation) > self.correlation_threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_weights.extend([correlation, correlation])

        if not edge_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)