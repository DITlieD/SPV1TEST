# forge/portfolio/optimizer.py
import pandas as pd
import riskfolio as rp
from data_processing_v2 import get_market_data

class PortfolioOptimizer:
    def __init__(self, asset_universe, returns_window=90):
        self.asset_universe = asset_universe
        self.returns_window = returns_window
        self.weights = self._get_initial_weights()

    def _get_initial_weights(self):
        """Returns equal weights for all assets."""
        n_assets = len(self.asset_universe)
        return {asset: 1.0 / n_assets for asset in self.asset_universe}

    def calculate_hrp_weights(self):
        """
        Calculates portfolio weights using Hierarchical Risk Parity.
        """
        print("[Genesis] Calculating HRP weights...")
        
        # 1. Get historical data
        returns = self._get_returns()
        if returns.empty:
            print("[Genesis] Could not get returns. Using initial weights.")
            return self.weights

        # 2. Calculate HRP weights
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
        w = port.optimization(model='HRP', codependence='pearson', rm='MV', rf=0, linkage='single', leaf_order=True)
        
        if w is not None and not w.empty:
            self.weights = w.to_dict()['weights']
            print("[Genesis] HRP weights calculated successfully.")
        else:
            print("[Genesis] HRP calculation failed. Using initial weights.")
            
        return self.weights

    def _get_returns(self):
        """
        Fetches historical data and calculates returns for all assets.
        """
        all_returns = {}
        for asset in self.asset_universe:
            df = get_market_data(asset, '1d', limit=self.returns_window)
            if not df.empty:
                all_returns[asset] = df['close'].pct_change().dropna()
        
        return pd.DataFrame(all_returns)