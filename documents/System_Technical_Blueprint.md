# The Archangel Protocol: A Technical Blueprint

This document provides a detailed technical explanation of the Archangel Protocol, an automated, hyper-adaptive framework for discovering, validating, and deploying algorithmic trading strategies.

## I. System Philosophy: Autonomous Discovery & Robustness

The system is designed to automate the entire lifecycle of quantitative strategy R&D. It operates on two core principles:

1.  **Autonomous Discovery:** The system should not rely on pre-defined strategy templates. Instead, it must possess the capability to discover novel, complex trading logic from scratch by observing market data.
2.  **Extreme Robustness:** Every discovered strategy must be subjected to a multi-stage gauntlet of rigorous validation tests that go far beyond simple backtesting. The goal is to eliminate strategies based on spurious correlations and ensure they are adaptable to changing market conditions.

---

## II. The Orchestration Engine: `task_scheduler.py`

The entire workflow is orchestrated by the `run_single_forge_cycle` function within the `forge/overlord/task_scheduler.py` module. This function acts as the master conductor, executing each of the following steps in a precise sequence.

---

## III. Phase 1: Multi-Scale Data Processing & Feature Engineering

The foundation of any trading system is the quality of its data and features. This system employs a multi-scale approach to build a rich, comprehensive view of the market.

#### **A. Data Acquisition & Alignment**

*   **Multi-Timeframe Analysis:** The system fetches data on three distinct timeframes:
    *   **Strategic (4-Hour):** For identifying long-term trends and market regimes.
    *   **Tactical (15-Minute):** The primary timeframe for generating trading signals.
    *   **Microstructure (1-Minute):** For calculating high-frequency features related to trade flow and market impact.
*   **Data Alignment:** The data from these different timeframes is carefully aligned using a `merge_asof` operation to ensure that at any given point on the tactical timeframe, the model has access to the most recent information from the other scales without lookahead bias.

#### **B. The Feature Factory (`feature_factory.py`)**

This module is responsible for creating a diverse and powerful set of features.

*   **1. Standard Technical Indicators:** A baseline of well-understood indicators (RSI, MACD, Bollinger Bands, ATR, etc.) is calculated using the `pandas-ta` library.

*   **2. Probabilistic Features (Particle Filter):**
    *   **Algorithm:** A **Particle Filter** (`MarketStateTracker`) is implemented to model the market as a noisy, dynamic system.
    *   **Mechanism:** It maintains a cloud of "particles," each representing a possible true state of the market's return. At each timestep, it uses the observed return and a measure of market uncertainty (proxied by ATR) to update the particle cloud.
    *   **Output:** This process generates two powerful features:
        *   `denoised_close`: A smoothed version of the price, representing the filter's best estimate of the true underlying trend, stripped of noise.
        *   `state_uncertainty`: A measure of the dispersion of the particles, quantifying the system's confidence in its estimate. High uncertainty can signal chaotic or unpredictable conditions.

*   **3. On-Chain Intelligence (Simulated):**
    *   **Concept:** This component is designed to integrate data directly from the blockchain, providing insight into capital flows and investor sentiment.
    *   **Implementation:** Currently simulated via `onchain_data_fetcher.py`, it generates realistic data for:
        *   **Net Exchange Flow:** The net amount of an asset moving to/from exchanges (high inflows suggest selling pressure).
        *   **Stablecoin Supply:** The total supply of stablecoins (an increase suggests more "dry powder" available to enter the market).
    *   **Features:** Momentum and Z-scores are calculated from this data to create actionable features.

*   **4. Relational Dynamics (Graph Neural Network):**
    *   **Concept:** This is the most advanced feature type. It treats the entire crypto market as an interconnected graph, where assets are nodes and the relationships between them (e.g., rolling correlations) are edges.
    *   **Algorithm:** A **Graph Attention Network (GAT)**, a type of GNN, is used. The GAT is powerful because it learns to assign different levels of importance ("attention") to different connections, allowing it to dynamically understand which inter-market relationships are most influential at any given time.
    *   **Process (`run_graph_feature_pipeline`):**
        1.  The feature sets for all assets in the universe are combined.
        2.  A graph is constructed from the latest data point.
        3.  The GAT model processes this graph and outputs an `embedding` for each asset—a dense vector that numerically represents that asset's position and context within the wider market network.
        4.  These embedding values are merged back into the feature set for each asset.

---

## IV. Phase 2: Environment-Aware Strategy Synthesis

Once the feature set is built, the system begins the process of creating a new trading strategy.

#### **A. Market Regime Classification (`EnvironmentClassifier`)**

*   **Algorithm:** The unsupervised clustering algorithm **HDBSCAN** is used.
*   **Mechanism:** It analyzes the high-dimensional feature space of the market data and identifies distinct clusters, or "regimes." These regimes represent different market personalities (e.g., "Low-Volatility Bull Trend," "High-Volatility Sideways Chop").
*   **Purpose:** The identified regime is passed to the synthesis engine. This ensures that the strategy evolved is specifically adapted to the *current* market environment, rather than being a generic, one-size-fits-all model.

#### **B. Autonomous Strategy Synthesis (`StrategySynthesizer`)**

This is the creative core of the system.

*   **Algorithm:** **Genetic Programming (GP)**, implemented using the `deap` library.
*   **Mechanism:**
    1.  **Initialization:** A population of random trading strategies is created. Each strategy is represented as a tree structure.
    2.  **Primitives & Terminals:** The building blocks of these trees are:
        *   **Terminals:** The inputs, which are the entire feature set generated in Phase 1.
        *   **Primitives:** The functions, which include logical operators (`AND`, `OR`, `NOT`), relational operators (`>`, `<`), and arithmetic operators (`+`, `-`, `*`, `/`).
    3.  **Fitness Evaluation:** Each strategy tree is compiled into executable code. This code is then used to generate trading signals on historical data, which are evaluated by the **High-Fidelity Backtester**. The "fitness" of a strategy is its final **Log Wealth**.
    4.  **Evolution:** The best-performing strategies are selected. They "reproduce" through:
        *   **Crossover:** Two parent strategies swap branches of their logic trees to create new offspring.
        *   **Mutation:** A random part of a strategy's logic tree is altered.
*   **Outcome:** This evolutionary process runs for many generations, with the population of strategies continuously improving. The final output is the single best-performing strategy tree—the "champion."

---

## V. Phase 3: Multi-Layered Validation Gauntlet

Before the champion strategy is accepted, it must survive a brutal validation process designed to destroy weak or overfitted models.

#### **A. High-Fidelity Backtesting (`VectorizedBacktester`)**

*   **Engine:** `vectorbt`, a high-performance backtesting library.
*   **Realism:** The backtester is configured to model real-world market frictions:
    *   **Latency:** Trades are executed at the *next bar's open*, simulating the delay between signal generation and order execution.
    *   **Dynamic Slippage:** The cost of crossing the bid-ask spread is modeled as a fraction of the Average True Range (ATR), making trades in volatile conditions more expensive.
    *   **Liquidity Constraints:** A `max_participation_rate` ensures that the simulation cannot assume it can execute a trade that is larger than a certain percentage of the bar's actual traded volume.
    *   **Commissions:** A standard percentage-based fee is applied to all trades.

#### **B. The Causal Inference Gauntlet (`CausalValidator`)**

*   **Problem:** A strategy might be profitable in a backtest simply because a feature was *correlated* with future returns, not because it *caused* them. These spurious correlations are notoriously unreliable and break down in live trading.
*   **Solution:** The system uses the `dowhy` library to build a formal causal model.
    1.  **Modeling:** For each core feature in the champion strategy, it defines a causal graph where the feature is the "treatment," future returns are the "outcome," and variables like volume and volatility are "common causes" (confounders).
    2.  **Estimation:** It estimates the causal effect of the feature on returns.
    3.  **Refutation (Critical Step):** It actively tries to disprove the causal link using tests like the "Placebo Treatment" (shuffling the feature to see if the effect disappears) and "Random Common Cause" (adding a random variable to see if the effect changes).
*   **Outcome:** Only if a feature's causal effect is statistically significant AND it survives the refutation tests is it considered causally robust. If any core feature fails, the entire strategy is rejected.

---

## VI. Phase 4: Finalization & Monitoring

#### **A. The AI Analyst (Interpretability)**

*   **Problem:** The evolved strategy trees are machine code and difficult for humans to understand.
*   **Solution:** The `AIAnalyst` module recursively parses the champion strategy tree. It maps the primitives and terminals to a formatted string, generating a human-readable explanation of the strategy's logic. This provides crucial transparency and fulfills the "Human-in-the-Loop" requirement.

#### **B. The Immune System (Resilience)**

*   **Algorithm:** A deep learning **Autoencoder**.
*   **Mechanism:** The model is trained on a vast dataset of "normal" market and system behavior. In real-time, it takes the current state and attempts to reconstruct it.
*   **Anomaly Detection:** A large "reconstruction error" means the model is seeing something it has never seen before—a potential anomaly. This could be a market flash crash, a faulty data feed, or a system bug.
*   **Action:** A high error can trigger a defensive protocol, such as halting all trading or reducing position sizes, acting as an automated safety net.

#### **C. The Model Registry**

If a strategy successfully passes every single stage of this pipeline, it is saved to the `ModelRegistry`. This registry stores the executable model, its detailed performance metrics from the gauntlet, and the human-readable explanation from the AI Analyst, making it the new, active strategy for its designated asset.
