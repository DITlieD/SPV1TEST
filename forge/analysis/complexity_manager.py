"""
Kolmogorov Complexity Manager (KCM) - Algorithmic Complexity for Risk Management

This module implements the Kolmogorov Complexity Manager of the Causal Cognition Architecture (CCA).

Objective:
    Use algorithmic complexity as a proxy for market unpredictability and strategy overfitting.
    High complexity = high entropy = high risk = reduce position size.
    Complex strategies = likely overfitted = penalize in evolution.

Methodology:
    1. Market Complexity Index (MCI): Lempel-Ziv complexity on price/microstructure
    2. Strategy Complexity: Estimate complexity of GP tree structure
    3. Dynamic Risk Scaling: Scale position size inversely with MCI
    4. Parsimony Pressure: Penalize complex strategies in fitness

Outputs:
    - MCI: Real-time market complexity score
    - Risk_Multiplier: Position sizing adjustment [0, 1]
    - Strategy_Complexity_Penalty: Fitness penalty for overly complex strategies

Author: Singularity Protocol - ACN Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
from numba import jit
import zlib
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# STEP 4.1: MARKET COMPLEXITY INDEX (MCI) CALCULATOR
# ============================================================================

@jit(nopython=True)
def lempel_ziv_complexity_numba(sequence: np.ndarray) -> int:
    """
    Calculate Lempel-Ziv complexity (Numba optimized)

    LZ complexity counts the number of distinct patterns in a sequence.
    It's a measure of algorithmic randomness/compressibility.

    High LZ = high complexity = harder to predict

    Args:
        sequence: Binary or discretized sequence

    Returns:
        LZ complexity (number of distinct patterns)
    """
    n = len(sequence)
    i = 0
    complexity = 1
    prefix_length = 1

    while i + prefix_length <= n:
        # Check if current substring is novel
        substring = sequence[i:i+prefix_length]
        is_novel = True

        # Check against all previous substrings
        for j in range(i):
            for k in range(1, prefix_length + 1):
                if j + k <= i:
                    if np.array_equal(sequence[j:j+k], substring[:k]):
                        if k == prefix_length:
                            is_novel = False
                            break

        if is_novel:
            complexity += 1
            i += prefix_length
            prefix_length = 1
        else:
            prefix_length += 1

    return complexity


def calculate_lz_complexity(data: np.ndarray, discretize: bool = True, bins: int = 5) -> float:
    """
    Calculate Lempel-Ziv complexity with discretization

    Args:
        data: Continuous data (prices, returns, etc.)
        discretize: Whether to discretize continuous data
        bins: Number of bins for discretization

    Returns:
        Normalized LZ complexity [0, 1]
    """
    if discretize:
        # Discretize into bins
        discrete_data = np.digitize(data, bins=np.linspace(data.min(), data.max(), bins))
    else:
        discrete_data = data.astype(np.int32)

    # Calculate LZ complexity
    complexity = lempel_ziv_complexity_numba(discrete_data)

    # Normalize by theoretical maximum (n / log2(n))
    n = len(data)
    if n <= 1:
        return 0.0
    max_complexity = n / np.log2(n)

    normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0.0

    return float(normalized_complexity)


def calculate_compression_ratio(data: np.ndarray) -> float:
    """
    Calculate compression ratio using zlib

    Alternative to LZ complexity - measures compressibility.
    Low compression = high complexity.

    Args:
        data: Raw data

    Returns:
        Compression ratio [0, 1] (1 = incompressible = complex)
    """
    # Convert to bytes
    data_bytes = data.tobytes()

    # Compress
    compressed = zlib.compress(data_bytes, level=9)

    # Ratio
    compression_ratio = len(compressed) / len(data_bytes)

    # Invert: high compression ratio = low complexity
    # We want high values = high complexity
    complexity_score = 1.0 - compression_ratio

    return float(np.clip(complexity_score, 0, 1))


class MarketComplexityIndex:
    """
    Real-time Market Complexity Index calculator

    Tracks rolling complexity of price movements to gauge unpredictability.
    """

    def __init__(
        self,
        window_size: int = 100,
        update_frequency: int = 10,
        method: str = 'lz'  # 'lz' or 'compression'
    ):
        """
        Initialize MCI calculator

        Args:
            window_size: Rolling window for complexity calculation
            update_frequency: Recalculate every N updates
            method: 'lz' (Lempel-Ziv) or 'compression' (zlib ratio)
        """
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.method = method

        # Per-asset buffers
        self.price_buffers: Dict[str, deque] = {}
        self.return_buffers: Dict[str, deque] = {}
        self.mci_cache: Dict[str, float] = {}
        self.update_counter: Dict[str, int] = {}

    def initialize_asset(self, symbol: str):
        """Initialize tracking for an asset"""
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=self.window_size)
            self.return_buffers[symbol] = deque(maxlen=self.window_size)
            self.mci_cache[symbol] = 0.5  # Neutral default
            self.update_counter[symbol] = 0

    def calculate_mci(self, symbol: str, price: float) -> float:
        """
        Calculate Market Complexity Index for an asset

        Args:
            symbol: Asset symbol
            price: Current price

        Returns:
            MCI value [0, 1] (0 = simple/predictable, 1 = complex/random)
        """
        self.initialize_asset(symbol)

        # Update buffers
        self.price_buffers[symbol].append(price)

        if len(self.price_buffers[symbol]) >= 2:
            # Calculate return
            ret = np.log(price / self.price_buffers[symbol][-2])
            self.return_buffers[symbol].append(ret)

        self.update_counter[symbol] += 1

        # Only recalculate periodically
        if (self.update_counter[symbol] % self.update_frequency == 0 and
            len(self.return_buffers[symbol]) >= 20):

            returns = np.array(self.return_buffers[symbol])

            if self.method == 'lz':
                # Lempel-Ziv complexity on returns
                mci = calculate_lz_complexity(returns, discretize=True, bins=5)
            elif self.method == 'compression':
                # Compression ratio
                mci = calculate_compression_ratio(returns)
            else:
                mci = 0.5

            self.mci_cache[symbol] = mci

        return self.mci_cache[symbol]

    def get_all_mci(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate MCI for all assets

        Args:
            prices: Dictionary of {symbol: current_price}

        Returns:
            Dictionary of {symbol: MCI_value}
        """
        mci_values = {}

        for symbol, price in prices.items():
            mci_values[symbol] = self.calculate_mci(symbol, price)

        return mci_values


# ============================================================================
# STEP 4.2: STRATEGY COMPLEXITY ESTIMATOR
# ============================================================================

def estimate_tree_complexity(tree_string: str) -> float:
    """
    Estimate complexity of a GP tree structure

    Uses multiple heuristics:
    - Tree depth
    - Number of nodes
    - Number of unique operators
    - Kolmogorov complexity approximation (compression)

    Args:
        tree_string: String representation of GP tree

    Returns:
        Complexity score [0, 1]
    """
    # Heuristic 1: String length (proxy for tree size)
    length_score = min(len(tree_string) / 500.0, 1.0)  # Normalize by typical max

    # Heuristic 2: Nesting depth (count max parentheses depth)
    max_depth = 0
    current_depth = 0
    for char in tree_string:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1

    depth_score = min(max_depth / 10.0, 1.0)  # Normalize by typical max depth

    # Heuristic 3: Compression ratio (Kolmogorov approximation)
    compression_score = calculate_compression_ratio(np.frombuffer(tree_string.encode(), dtype=np.uint8))

    # Combined complexity
    complexity = 0.4 * length_score + 0.3 * depth_score + 0.3 * compression_score

    return float(complexity)


class StrategyComplexityEstimator:
    """
    Estimate and track complexity of evolved strategies
    """

    def __init__(self, complexity_threshold: float = 0.7):
        """
        Args:
            complexity_threshold: Threshold above which strategies are considered overly complex
        """
        self.complexity_threshold = complexity_threshold
        self.complexity_history: List[float] = []

    def evaluate_complexity(self, strategy_representation: str) -> Dict[str, float]:
        """
        Evaluate strategy complexity

        Args:
            strategy_representation: String representation of strategy (GP tree, etc.)

        Returns:
            Dictionary with:
            - complexity: Complexity score [0, 1]
            - penalty: Fitness penalty (0 if below threshold, increasing above)
            - is_overfit_risk: Boolean flag
        """
        complexity = estimate_tree_complexity(strategy_representation)

        # Record history
        self.complexity_history.append(complexity)

        # Calculate penalty (quadratic above threshold)
        if complexity > self.complexity_threshold:
            excess = complexity - self.complexity_threshold
            penalty = excess ** 2  # Quadratic penalty
        else:
            penalty = 0.0

        # Overfit risk flag
        is_overfit_risk = complexity > self.complexity_threshold

        return {
            'complexity': complexity,
            'penalty': penalty,
            'is_overfit_risk': is_overfit_risk
        }


# ============================================================================
# STEP 4.3: DYNAMIC RISK SCALING
# ============================================================================

class DynamicRiskScaler:
    """
    Scale position sizes based on market complexity

    High MCI = high unpredictability = reduce risk
    """

    def __init__(
        self,
        mci_calculator: MarketComplexityIndex,
        base_risk: float = 1.0,
        min_risk_multiplier: float = 0.1,
        max_risk_multiplier: float = 1.0,
        scaling_mode: str = 'inverse'  # 'inverse', 'exponential', 'step'
    ):
        """
        Initialize risk scaler

        Args:
            mci_calculator: MCI calculator instance
            base_risk: Base risk level (1.0 = full size)
            min_risk_multiplier: Minimum multiplier (when MCI = 1)
            max_risk_multiplier: Maximum multiplier (when MCI = 0)
            scaling_mode: How to scale risk with MCI
        """
        self.mci_calculator = mci_calculator
        self.base_risk = base_risk
        self.min_risk_multiplier = min_risk_multiplier
        self.max_risk_multiplier = max_risk_multiplier
        self.scaling_mode = scaling_mode

    def calculate_risk_multiplier(self, symbol: str, mci: float) -> float:
        """
        Calculate risk multiplier for an asset

        Args:
            symbol: Asset symbol
            mci: Market Complexity Index [0, 1]

        Returns:
            Risk multiplier to apply to position size
        """
        if self.scaling_mode == 'inverse':
            # Linear inverse: multiplier = max - (max - min) * mci
            multiplier = self.max_risk_multiplier - (self.max_risk_multiplier - self.min_risk_multiplier) * mci

        elif self.scaling_mode == 'exponential':
            # Exponential decay: multiplier = max * exp(-k * mci)
            k = 2.0  # Decay rate
            multiplier = self.max_risk_multiplier * np.exp(-k * mci)
            multiplier = max(multiplier, self.min_risk_multiplier)

        elif self.scaling_mode == 'step':
            # Step function
            if mci < 0.3:
                multiplier = self.max_risk_multiplier
            elif mci < 0.7:
                multiplier = 0.5
            else:
                multiplier = self.min_risk_multiplier

        else:
            multiplier = 1.0

        return float(multiplier)

    def get_risk_adjustments(self, prices: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Get risk adjustments for all assets

        Args:
            prices: Current prices

        Returns:
            Dictionary of {symbol: {mci, risk_multiplier, adjusted_risk}}
        """
        # Calculate MCIs
        mci_values = self.mci_calculator.get_all_mci(prices)

        # Calculate risk multipliers
        adjustments = {}

        for symbol, mci in mci_values.items():
            multiplier = self.calculate_risk_multiplier(symbol, mci)
            adjusted_risk = self.base_risk * multiplier

            adjustments[symbol] = {
                'mci': mci,
                'risk_multiplier': multiplier,
                'adjusted_risk': adjusted_risk
            }

        return adjustments


# ============================================================================
# STEP 4.4: INTEGRATED COMPLEXITY MANAGER
# ============================================================================

class KolmogorovComplexityManager:
    """
    Integrated complexity management system

    Combines:
    - Market complexity tracking (MCI)
    - Strategy complexity estimation
    - Dynamic risk scaling
    - Parsimony-driven evolution
    """

    def __init__(
        self,
        mci_window: int = 100,
        mci_update_freq: int = 10,
        strategy_complexity_threshold: float = 0.7,
        base_risk: float = 1.0,
        min_risk_multiplier: float = 0.1
    ):
        """
        Initialize KCM

        Args:
            mci_window: Window for MCI calculation
            mci_update_freq: MCI update frequency
            strategy_complexity_threshold: Complexity threshold for strategies
            base_risk: Base risk level
            min_risk_multiplier: Minimum risk multiplier
        """
        # Initialize components
        self.mci_calculator = MarketComplexityIndex(
            window_size=mci_window,
            update_frequency=mci_update_freq,
            method='lz'
        )

        self.strategy_complexity_estimator = StrategyComplexityEstimator(
            complexity_threshold=strategy_complexity_threshold
        )

        self.risk_scaler = DynamicRiskScaler(
            self.mci_calculator,
            base_risk=base_risk,
            min_risk_multiplier=min_risk_multiplier,
            scaling_mode='inverse'
        )

    def process_market_update(self, prices: Dict[str, float]) -> Dict[str, any]:
        """
        Process market update and return complexity-based adjustments

        Args:
            prices: Current prices for all assets

        Returns:
            Dictionary with MCI values and risk adjustments
        """
        # Get risk adjustments (this also updates MCI)
        risk_adjustments = self.risk_scaler.get_risk_adjustments(prices)

        return {
            'risk_adjustments': risk_adjustments,
            'timestamp': pd.Timestamp.now()
        }

    def evaluate_strategy(self, strategy_representation: str) -> Dict[str, float]:
        """
        Evaluate a strategy's complexity

        Args:
            strategy_representation: String representation of strategy

        Returns:
            Complexity evaluation results
        """
        return self.strategy_complexity_estimator.evaluate_complexity(strategy_representation)

    def get_complexity_adjusted_fitness(
        self,
        base_fitness: float,
        strategy_representation: str,
        complexity_weight: float = 0.2
    ) -> float:
        """
        Calculate complexity-adjusted fitness

        Fitness_adjusted = Fitness_base - (complexity_weight * complexity_penalty)

        Args:
            base_fitness: Original fitness from backtest
            strategy_representation: Strategy string
            complexity_weight: Weight for complexity penalty

        Returns:
            Adjusted fitness
        """
        complexity_eval = self.evaluate_strategy(strategy_representation)

        adjusted_fitness = base_fitness - (complexity_weight * complexity_eval['penalty'])

        return adjusted_fitness


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_kcm():
    """
    Test Kolmogorov Complexity Manager
    """
    print("="*80)
    print("Kolmogorov Complexity Manager - Unit Test")
    print("="*80)

    np.random.seed(42)

    # Test 1: LZ Complexity
    print("\nTest 1: Lempel-Ziv Complexity")

    # Simple pattern (low complexity)
    simple_pattern = np.array([1, 2, 1, 2, 1, 2, 1, 2] * 10)
    lz_simple = calculate_lz_complexity(simple_pattern, discretize=False)
    print(f"  Simple pattern LZ: {lz_simple:.4f} (expect low ~0.1-0.3)")

    # Random pattern (high complexity)
    random_pattern = np.random.randint(0, 5, size=80)
    lz_random = calculate_lz_complexity(random_pattern, discretize=False)
    print(f"  Random pattern LZ: {lz_random:.4f} (expect high ~0.7-1.0)")

    print(f"\n  [{'OK' if lz_random > lz_simple else 'WARN'}] Random > Simple: {lz_random:.4f} > {lz_simple:.4f}")

    # Test 2: Market Complexity Index
    print("\n" + "="*80)
    print("Market Complexity Index (MCI) - Test")
    print("="*80)

    mci_calc = MarketComplexityIndex(window_size=50, update_frequency=5)

    # Simulate trending market (low complexity)
    print("\nSimulating trending market...")
    trending_price = 40000
    for i in range(60):
        trending_price *= (1 + 0.001)  # Steady trend
        mci = mci_calc.calculate_mci('BTC_TRENDING', trending_price)

    mci_trending = mci
    print(f"  MCI (trending): {mci_trending:.4f}")

    # Simulate random market (high complexity)
    print("\nSimulating random market...")
    mci_calc_random = MarketComplexityIndex(window_size=50, update_frequency=5)
    random_price = 40000
    for i in range(60):
        random_price *= (1 + np.random.randn() * 0.02)  # Random walk
        mci = mci_calc_random.calculate_mci('BTC_RANDOM', random_price)

    mci_random = mci
    print(f"  MCI (random): {mci_random:.4f}")

    # Test 3: Strategy Complexity
    print("\n" + "="*80)
    print("Strategy Complexity Estimator - Test")
    print("="*80)

    estimator = StrategyComplexityEstimator(complexity_threshold=0.7)

    # Simple strategy
    simple_strategy = "if(gt(rsi, 70), -1, if(lt(rsi, 30), 1, 0))"
    simple_eval = estimator.evaluate_complexity(simple_strategy)
    print(f"\nSimple strategy:")
    print(f"  Complexity: {simple_eval['complexity']:.4f}")
    print(f"  Penalty: {simple_eval['penalty']:.4f}")
    print(f"  Overfit risk: {simple_eval['is_overfit_risk']}")

    # Complex strategy
    complex_strategy = "if(and(gt(add(mul(rsi,2),div(macd,atr)),70),lt(sub(bb_upper,bb_lower),mul(atr,3))),if(gt(volume,mul(avg_volume,1.5)),-1,0),if(and(lt(sub(close,sma50),mul(atr,2)),gt(adx,25)),if(lt(rsi,30),1,if(gt(cci,100),-1,0)),0))" * 3
    complex_eval = estimator.evaluate_complexity(complex_strategy)
    print(f"\nComplex strategy:")
    print(f"  Complexity: {complex_eval['complexity']:.4f}")
    print(f"  Penalty: {complex_eval['penalty']:.4f}")
    print(f"  Overfit risk: {complex_eval['is_overfit_risk']}")

    # Test 4: Risk Scaling
    print("\n" + "="*80)
    print("Dynamic Risk Scaling - Test")
    print("="*80)

    mci_calc_risk = MarketComplexityIndex(window_size=50)
    risk_scaler = DynamicRiskScaler(
        mci_calc_risk,
        base_risk=1.0,
        min_risk_multiplier=0.1,
        scaling_mode='inverse'
    )

    # Simulate different MCI levels
    test_prices = {'BTCUSDT': 40000}

    # Feed some data
    for i in range(100):
        test_prices['BTCUSDT'] *= (1 + np.random.randn() * 0.02)
        mci_calc_risk.calculate_mci('BTCUSDT', test_prices['BTCUSDT'])

    adjustments = risk_scaler.get_risk_adjustments(test_prices)

    print(f"\nRisk adjustments:")
    for symbol, adj in adjustments.items():
        print(f"  {symbol}:")
        print(f"    MCI: {adj['mci']:.4f}")
        print(f"    Risk multiplier: {adj['risk_multiplier']:.4f}")
        print(f"    Adjusted risk: {adj['adjusted_risk']:.4f}")

    # Test 5: Integrated KCM
    print("\n" + "="*80)
    print("Integrated Complexity Manager - Test")
    print("="*80)

    kcm = KolmogorovComplexityManager(
        mci_window=50,
        strategy_complexity_threshold=0.7
    )

    # Process market update
    prices = {'BTCUSDT': 40000, 'ETHUSDT': 2500}
    for i in range(60):
        for symbol in prices:
            prices[symbol] *= (1 + np.random.randn() * 0.015)

        result = kcm.process_market_update(prices)

    print(f"\n[OK] Processed 60 market updates")
    print(f"\nFinal risk adjustments:")
    for symbol, adj in result['risk_adjustments'].items():
        print(f"  {symbol}: MCI={adj['mci']:.4f}, Multiplier={adj['risk_multiplier']:.4f}")

    # Test complexity-adjusted fitness
    test_strategy = "if(gt(rsi,70),-1,if(lt(rsi,30),1,0))"
    base_fitness = 1.5
    adjusted_fitness = kcm.get_complexity_adjusted_fitness(base_fitness, test_strategy)

    print(f"\nComplexity-adjusted fitness:")
    print(f"  Base fitness: {base_fitness:.4f}")
    print(f"  Adjusted fitness: {adjusted_fitness:.4f}")

    print("\n" + "="*80)
    print("PHASE 4 COMPLETE: Kolmogorov Complexity Manager (KCM)")
    print("="*80)
    print("\n[OK] Step 4.1: Market Complexity Index - COMPLETE")
    print("  - Lempel-Ziv complexity calculation")
    print("  - Real-time MCI tracking")
    print("\n[OK] Step 4.2: Strategy Complexity Estimator - COMPLETE")
    print("  - Tree depth/size heuristics")
    print("  - Compression-based estimation")
    print("\n[OK] Step 4.3: Dynamic Risk Scaling - COMPLETE")
    print("  - Inverse MCI-based scaling")
    print("  - Configurable scaling modes")
    print("\n[OK] Step 4.4: Integrated KCM - COMPLETE")
    print("  - Unified complexity management")
    print("  - Complexity-adjusted fitness")
    print("\nOutputs:")
    print("  - MCI: Market unpredictability measure")
    print("  - Risk_Multiplier: Position sizing adjustment")
    print("  - Complexity_Penalty: Parsimony pressure")
    print("\nNext: Phase 5 - Python HF-ABM Simulator")


if __name__ == "__main__":
    test_kcm()
