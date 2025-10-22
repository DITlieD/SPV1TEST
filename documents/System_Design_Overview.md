# The Singularity Protocol: System Workflow Overview

This document provides a high-level overview of the automated trading system's design, explaining the journey from strategy creation to final validation.

The system is designed as a fully autonomous pipeline that continuously discovers, refines, and validates new trading strategies tailored to the current market environment.

---

### The Core Workflow

The system's process can be visualized as a series of increasingly rigorous filters. A cycle begins with thousands of potential strategies, and only the single best, most robust candidate survives to the end.

#### **Step 1: Autonomous Strategy Synthesis (The Genetic Engine)**

The process starts here. The system uses a powerful AI technique called **Genetic Programming (GP 2.0)**.

*   **What it does:** Instead of just optimizing parameters, this engine literally writes and evolves entire trading strategies as code. It combines raw market features (like RSI, volume, etc.) with logical operators (`IF`, `AND`, `GREATER THAN`) to create a massive population of unique strategies. Each strategy is a logical tree that outputs a clear "BUY" or "HOLD" signal.

#### **Step 2: High-Fidelity Backtesting (The Arena)**

The thousands of strategies created in Step 1 are immediately thrown into a realistic market simulator to test their performance.

*   **What it does:** Each strategy is evaluated against historical data. This isn't a simple backtest; it's a high-fidelity simulation that models real-world trading frictions like network latency, trading fees, slippage (price changes on execution), and liquidity constraints (the inability to trade unlimited size). Only strategies that are profitable in this harsh environment survive.

#### **Step 3: Causal Inference (The Gauntlet)**

This is the system's most critical intellectual filter. Surviving strategies are tested to ensure their logic is sound and not just based on random correlations.

*   **What it does:** The system analyzes the core features used by the strategy (e.g., "RSI > 50"). It uses **Causal Inference** models to determine if the feature has a genuine, statistically significant, cause-and-effect relationship with future price movements. It actively tries to prove the relationship is fake using refutation tests. Any strategy built on a spurious correlation is immediately eliminated.

#### **Step 4: Final Validation (The Gauntlet)**

The few strategies that have proven to be profitable and causally robust undergo a final battery of stress tests.

*   **What it does:** The strategy is subjected to a **Walk-Forward Analysis**, where it's tested on multiple periods of out-of-sample data to ensure it adapts to changing market conditions and isn't just curve-fit to one specific historical period.

#### **Step 5: Registration & Analysis (The Archives)**

If a single champion strategy emerges from all previous steps, it is considered "forged" and ready for deployment.

*   **What it does:**
    1.  **AI Analyst:** The system translates the complex, machine-generated strategy tree into a human-readable explanation (e.g., "BUY when RSI is greater than 50 AND market volatility is low").
    2.  **Model Registry:** The validated strategy, along with its performance metrics and explanation, is saved to the Model Registry, becoming the new active strategy for the asset.

This entire cycle runs automatically, ensuring the system constantly adapts and evolves its logic to stay ahead of the ever-changing market.
