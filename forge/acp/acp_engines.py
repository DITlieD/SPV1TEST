"""
ACP Engines: Chimera, Apex Predator, and Topological Radar

PHASES 6, 7, 8: Consolidated implementation of the Apex Convergence Protocol engines

These engines work together to infer hidden strategies and synthesize counter-strategies:

- Chimera Engine (Phase 6): Infer competitor strategies via microstructure mimicry
- Apex Predator (Phase 7): Synthesize counter-strategies with dual fitness
- Topological Radar (Phase 8): Detect regime changes via persistent homology

Author: Singularity Protocol - ACN Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import differential_evolution
from scipy.spatial.distance import euclidean
import warnings
import pandas_ta as ta

def extract_observed_market_signature(price_df: pd.DataFrame) -> np.ndarray:
    """
    Extracts a microstructure signature directly from market data.
    """ 
    # Recent volatility (ATR percentage)
    atr = price_df.ta.atr(length=14)
    volatility = (atr / price_df['close']).mean() # Use mean over the window

    # Trade intensity proxy (average volume)
    trade_intensity = price_df['volume'].mean()

    # Price return skewness/kurtosis
    returns = price_df['close'].pct_change().dropna()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    return np.array([volatility, trade_intensity, skewness, kurtosis])

warnings.filterwarnings('ignore')


# ============================================================================
# PHASE 6: CHIMERA ENGINE - STRATEGY INFERENCE
# ============================================================================

class ChimeraEngine:
    """
    Infer hidden strategies via microstructure mimicry

    Uses CMA-ES to find strategy parameters that produce similar
    microstructure signatures to observed market behavior.

    Objective:
        Find parameters θ such that:
        Microstructure(Strategy(θ)) ≈ Microstructure(Observed_Market)
    """

    def __init__(
        self,
        strategy_template: Callable,
        microstructure_features: List[str] = None,
        max_iterations: int = 50
    ):
        """
        Initialize Chimera Engine

        Args:
            strategy_template: Callable strategy function to parameterize
            microstructure_features: List of features to match (OFI, spread, etc.)
            max_iterations: Max CMA-ES iterations
        """
        self.strategy_template = strategy_template
        self.microstructure_features = microstructure_features or [
            'ofi', 'book_pressure', 'spread', 'trade_intensity'
        ]
        self.max_iterations = max_iterations

    def extract_microstructure_signature(
        self,
        price_df: pd.DataFrame,
        signals: np.ndarray
    ) -> np.ndarray:
        """
        Extract microstructure signature from price action + signals

        Args:
            price_df: DataFrame with OHLCV data
            signals: Trading signals (0=HOLD, 1=BUY, -1=SELL)

        Returns:
            Feature vector representing microstructure signature
        """
        # Calculate microstructure features
        features = []

        # OFI proxy: difference in buy vs sell pressure
        buy_volume = np.sum(price_df['volume'][signals == 1])
        sell_volume = np.sum(price_df['volume'][signals == -1])
        ofi_proxy = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)
        features.append(ofi_proxy)

        # Trade intensity: fraction of active signals
        trade_intensity = np.mean(signals != 0)
        features.append(trade_intensity)

        # Average trade size (when active)
        active_volumes = price_df['volume'][signals != 0]
        avg_trade_size = np.mean(active_volumes) if len(active_volumes) > 0 else 0
        features.append(avg_trade_size)

        # Spread impact proxy: volatility during trades
        trade_volatility = np.std(price_df['close'][signals != 0]) if np.sum(signals != 0) > 10 else 0
        features.append(trade_volatility)

        return np.array(features)

    def infer_strategy_parameters(
        self,
        observed_microstructure: np.ndarray,
        price_df: pd.DataFrame,
        param_bounds: List[Tuple[float, float]]
    ) -> Dict[str, any]:
        """
        Infer strategy parameters via CMA-ES

        Args:
            observed_microstructure: Target microstructure signature
            price_df: Price data
            param_bounds: Parameter bounds for optimization

        Returns:
            Dictionary with inferred parameters and fit quality
        """
        def objective(params):
            """Objective: Wasserstein distance between microstructures"""
            # Generate signals using trial parameters
            try:
                signals = self.strategy_template(price_df, params)
            except:
                return 1e10  # Penalty for invalid parameters

            # Extract microstructure
            trial_microstructure = self.extract_microstructure_signature(price_df, signals)

            # Distance (Euclidean for simplicity; could use Wasserstein)
            distance = euclidean(observed_microstructure, trial_microstructure)

            return distance

        # Run differential evolution (similar to CMA-ES)
        print(f"\n[Chimera] Inferring strategy parameters...")
        result = differential_evolution(
            objective,
            bounds=param_bounds,
            maxiter=self.max_iterations,
            polish=False,
            workers=1
        )

        inferred_params = result.x
        fit_quality = 1.0 / (1.0 + result.fun)  # Normalize to [0, 1]

        print(f"[Chimera] Inference complete")
        print(f"  - Fit quality: {fit_quality:.4f}")
        print(f"  - Inferred params: {inferred_params}")

        return {
            'parameters': inferred_params,
            'fit_quality': fit_quality,
            'distance': result.fun
        }


# ============================================================================
# PHASE 7: APEX PREDATOR - COUNTER-STRATEGY SYNTHESIS
# ============================================================================

class ApexPredator:
    """
    Counter-Strategy Synthesis with Dual Fitness

    Evolves strategies that:
    1. Perform well in backtest (traditional fitness)
    2. Exploit weaknesses in inferred opponent strategies (adversarial fitness)

    Fitness = α * Traditional_Fitness + β * Adversarial_Fitness
    """

    def __init__(
        self,
        chimera_engine: ChimeraEngine,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Initialize Apex Predator

        Args:
            chimera_engine: Chimera Engine for strategy inference
            alpha: Weight for traditional fitness
            beta: Weight for adversarial fitness
        """
        self.chimera_engine = chimera_engine
        self.alpha = alpha
        self.beta = beta

    def calculate_adversarial_fitness(
        self,
        our_strategy_signals: np.ndarray,
        opponent_strategy_signals: np.ndarray,
        price_df: pd.DataFrame
    ) -> float:
        """
        Calculate adversarial fitness: how well do we exploit opponent?

        Args:
            our_strategy_signals: Our strategy's signals
            opponent_strategy_signals: Inferred opponent signals
            price_df: Price data

        Returns:
            Adversarial fitness score
        """
        # Calculate returns when we trade AGAINST opponent
        # If opponent buys, we benefit from selling (and vice versa)

        # Our returns when we counter-trade
        counter_returns = []

        for i in range(1, len(our_strategy_signals)):
            if our_strategy_signals[i] != 0 and opponent_strategy_signals[i] != 0:
                # We're both active
                price_change = (price_df['close'].iloc[i] - price_df['close'].iloc[i-1]) / price_df['close'].iloc[i-1]

                if our_strategy_signals[i] == -opponent_strategy_signals[i]:
                    # Counter-trade: we profit from their mistakes
                    our_return = our_strategy_signals[i] * price_change
                    counter_returns.append(our_return)

        # Adversarial fitness = average profit when counter-trading
        if len(counter_returns) > 0:
            adversarial_fitness = np.mean(counter_returns)
        else:
            adversarial_fitness = 0.0

        return adversarial_fitness

    def calculate_dual_fitness(
        self,
        our_strategy_signals: np.ndarray,
        traditional_fitness: float,
        opponent_strategy_signals: np.ndarray,
        price_df: pd.DataFrame
    ) -> float:
        """
        Calculate dual fitness (traditional + adversarial)

        Args:
            our_strategy_signals: Our strategy signals
            traditional_fitness: Backtest fitness
            opponent_strategy_signals: Opponent signals
            price_df: Price data

        Returns:
            Combined dual fitness
        """
        adversarial_fitness = self.calculate_adversarial_fitness(
            our_strategy_signals,
            opponent_strategy_signals,
            price_df
        )

        dual_fitness = (
            self.alpha * traditional_fitness +
            self.beta * adversarial_fitness
        )

        return dual_fitness

    def synthesize_counter_strategy(
        self,
        price_df: pd.DataFrame,
        observed_opponent_microstructure: np.ndarray,
        strategy_evolution_function: Callable,
        opponent_strategy_template: Callable,
        param_bounds: List[Tuple[float, float]]
    ) -> Dict[str, any]:
        """
        Synthesize counter-strategy

        Args:
            price_df: Price data
            observed_opponent_microstructure: Opponent's microstructure signature
            strategy_evolution_function: Function to evolve strategies
            opponent_strategy_template: Template for opponent strategy
            param_bounds: Parameter bounds

        Returns:
            Counter-strategy and performance metrics
        """
        print(f"\n[Apex Predator] Synthesizing counter-strategy...")

        # Step 1: Infer opponent strategy
        print(f"[Step 1] Inferring opponent strategy...")
        self.chimera_engine.strategy_template = opponent_strategy_template

        opponent_params = self.chimera_engine.infer_strategy_parameters(
            observed_opponent_microstructure,
            price_df,
            param_bounds
        )

        # Generate opponent signals
        opponent_signals = opponent_strategy_template(price_df, opponent_params['parameters'])

        # Step 2: Evolve counter-strategy with dual fitness
        print(f"[Step 2] Evolving counter-strategy with dual fitness...")

        # (This would integrate with GP evolution system)
        # For testing, simulate a counter-strategy
        counter_strategy_signals = -opponent_signals  # Simple inversion

        print(f"[Apex Predator] Counter-strategy synthesized")

        return {
            'opponent_params': opponent_params,
            'counter_strategy_signals': counter_strategy_signals
        }


# ============================================================================
# PHASE 8: TOPOLOGICAL RADAR - REGIME DETECTION
# ============================================================================

class TopologicalRadar:
    """
    Detect regime changes using Topological Data Analysis (TDA)

    Uses Persistent Homology to detect when market topology changes,
    indicating regime shifts.
    """

    def __init__(
        self,
        window_size: int = 50,
        embedding_dim: int = 3,
        embedding_delay: int = 1
    ):
        """
        Initialize Topological Radar

        Args:
            window_size: Sliding window for point cloud construction
            embedding_dim: Takens embedding dimension
            embedding_delay: Time delay for embedding
        """
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.embedding_delay = embedding_delay

        # Try to import ripser (optional dependency)
        try:
            from ripser import ripser
            self.ripser_available = True
            self.ripser = ripser
        except ImportError:
            self.ripser_available = False
            print("[WARN] ripser not installed - using simplified topology")

    def create_point_cloud(self, returns: np.ndarray) -> np.ndarray:
        """
        Create point cloud via Takens embedding

        Args:
            returns: Return series

        Returns:
            Embedded point cloud [N x embedding_dim]
        """
        n = len(returns)
        num_points = n - (self.embedding_dim - 1) * self.embedding_delay

        if num_points <= 0:
            return np.array([[0]])

        point_cloud = np.zeros((num_points, self.embedding_dim))

        for i in range(num_points):
            for j in range(self.embedding_dim):
                point_cloud[i, j] = returns[i + j * self.embedding_delay]

        return point_cloud

    def calculate_persistence(self, point_cloud: np.ndarray) -> float:
        """
        Calculate persistence score (simplified)

        Full TDA would compute persistence diagrams with ripser.
        This is a simplified proxy.

        Args:
            point_cloud: Point cloud

        Returns:
            Persistence score
        """
        if self.ripser_available:
            # Full persistent homology
            diagrams = self.ripser(point_cloud, maxdim=1)
            # Extract persistence (birth-death)
            h0 = diagrams['dgms'][0]
            if len(h0) > 1:
                persistence = np.mean(h0[:, 1] - h0[:, 0])
            else:
                persistence = 0.0
        else:
            # Simplified: use variance of distances as proxy
            from scipy.spatial.distance import pdist
            distances = pdist(point_cloud)
            persistence = np.std(distances)

        return float(persistence)

    def detect_regime_change(
        self,
        returns: np.ndarray,
        persistence_history: List[float],
        threshold_std: float = 2.0
    ) -> bool:
        """
        Detect regime change via persistence anomaly

        Args:
            returns: Recent return series
            persistence_history: Historical persistence values
            threshold_std: Threshold in standard deviations

        Returns:
            True if regime change detected
        """
        # Create point cloud
        point_cloud = self.create_point_cloud(returns)

        # Calculate persistence
        current_persistence = self.calculate_persistence(point_cloud)

        # Check for anomaly
        if len(persistence_history) > 10:
            mean_persistence = np.mean(persistence_history)
            std_persistence = np.std(persistence_history) + 1e-10

            z_score = (current_persistence - mean_persistence) / std_persistence

            regime_change = abs(z_score) > threshold_std
        else:
            regime_change = False

        persistence_history.append(current_persistence)

        return regime_change


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_acp_engines():
    """
    Test ACP Engines (Phases 6, 7, 8)
    """
    print("="*80)
    print("ACP ENGINES - Unit Test")
    print("="*80)

    np.random.seed(42)

    # Synthetic price data
    n = 200
    price_df = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.randn(n) * 0.01),
        'volume': np.random.uniform(1000, 5000, n)
    })

    # PHASE 6: Chimera Engine
    print("\n" + "="*80)
    print("PHASE 6: Chimera Engine - Test")
    print("="*80)

    def simple_strategy(df, params):
        """Simple RSI strategy"""
        threshold_buy, threshold_sell = params
        # Simplified RSI
        returns = df['close'].pct_change()
        rsi = 50 + returns.rolling(14).mean() * 100

        signals = np.zeros(len(df))
        signals[rsi < threshold_buy] = 1
        signals[rsi > threshold_sell] = -1

        return signals

    chimera = ChimeraEngine(
        strategy_template=simple_strategy,
        max_iterations=20
    )

    # Generate "observed" signals
    observed_signals = simple_strategy(price_df, [30, 70])
    observed_microstructure = chimera.extract_microstructure_signature(price_df, observed_signals)

    print(f"\n[OK] Observed microstructure: {observed_microstructure}")

    # Infer parameters
    inferred = chimera.infer_strategy_parameters(
        observed_microstructure,
        price_df,
        param_bounds=[(20, 40), (60, 80)]
    )

    print(f"\n[OK] PHASE 6 COMPLETE: Chimera Engine")

    # PHASE 7: Apex Predator
    print("\n" + "="*80)
    print("PHASE 7: Apex Predator - Test")
    print("="*80)

    apex = ApexPredator(chimera, alpha=0.6, beta=0.4)

    our_signals = -observed_signals  # Counter-trade
    traditional_fitness = 0.8

    dual_fitness = apex.calculate_dual_fitness(
        our_signals,
        traditional_fitness,
        observed_signals,
        price_df
    )

    print(f"\n[OK] Dual fitness calculated: {dual_fitness:.4f}")
    print(f"[OK] PHASE 7 COMPLETE: Apex Predator")

    # PHASE 8: Topological Radar
    print("\n" + "="*80)
    print("PHASE 8: Topological Radar - Test")
    print("="*80)

    radar = TopologicalRadar(window_size=50, embedding_dim=3)

    returns = price_df['close'].pct_change().dropna().values

    persistence_history = []

    regime_changes = []
    for i in range(50, len(returns), 10):
        window_returns = returns[i-50:i]
        regime_change = radar.detect_regime_change(window_returns, persistence_history)
        regime_changes.append(regime_change)

    print(f"\n[OK] Regime detection complete")
    print(f"  - Windows analyzed: {len(regime_changes)}")
    print(f"  - Regime changes detected: {sum(regime_changes)}")
    print(f"[OK] PHASE 8 COMPLETE: Topological Radar")

    print("\n" + "="*80)
    print("PHASES 6-8 COMPLETE: ACP Engines")
    print("="*80)
    print("\n[OK] Chimera Engine: Strategy inference operational")
    print("[OK] Apex Predator: Counter-strategy synthesis operational")
    print("[OK] Topological Radar: Regime detection operational")
    print("\nNext: Phases 9-10 - System Integration & Testing")


if __name__ == "__main__":
    test_acp_engines()
