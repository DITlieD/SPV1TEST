# The Singularity Protocol: System Blueprint (V5)

## 1. Core Objective & Philosophy

The Singularity Protocol is an autonomous, self-evolving crypto trading system. Its prime directive is to achieve and maintain profitability by continuously adapting to changing market conditions.

The core philosophy is built on two cycles:

1.  **Genesis (Bootstrap):** An offline, one-time process that forges the initial generation of intelligent components (prediction models, risk managers, etc.) from historical data.
2.  **The Live Loop (Autonomous Operation):** A continuous, live-trading cycle that not only executes trades but also constantly monitors its own performance, identifies when its models are becoming obsolete, and automatically triggers the "Forge" to evolve superior replacements without human intervention.

This document outlines the architecture of this system.

---

## 2. Key Components & File Responsibilities

The system is composed of several key scripts and modules, each with a specific responsibility.

### 2.1. The Orchestrator (`app.py`)

This is the master entry point of the application.

*   **What it does:** Launches a Flask web server for the UI and starts the main `ServiceManager`. The `ServiceManager` then initiates the entire live trading system in a separate, non-blocking background thread.
*   **Why it exists:** It provides a user-friendly interface to start, stop, and monitor the system, while decoupling the web server from the core trading logic.

### 2.2. The Live Trading Loop (`trading_manager.py`)

This file contains the `MultiAssetTrader`, the orchestrator for all real-time trading activities.

*   **What it does:** It manages multiple, concurrent asynchronous loops for different tasks:
    *   **Data Ingestion:** Efficiently listens for new OHLCV candles from the exchange.
    *   **Data Validation:** Uses the `DataValidator` to ensure data quality.
    *   **Feature Engineering:** Calculates all necessary features for the models.
    *   **Signal Execution:** Passes the features to the appropriate `V3Agent` for a decision.
    *   **State Reconciliation:** Periodically checks its internal state against the exchange's state to prevent discrepancies.
*   **Why it exists:** To manage the high-frequency, real-time tasks of data collection and trade execution for a universe of assets in a robust, concurrent manner.

### 2.3. The Agent Brain (`forge/core/agent.py`)

This file contains the `V3Agent`, the most critical component of the live system. There is one `V3Agent` instance for each asset.

*   **What it does:**
    1.  **Loads all its models:** The specialist prediction models, the HMM regime detector, and the pre-trained RL Governor.
    2.  **Enriches State:** Determines the current market regime with the HMM and adds this information to the feature set.
    3.  **Generates a Signal:** Gets predictions from all specialist models (the "Hydra" ensemble) and combines them using performance-based weights from the `HedgeEnsembleManager`.
    4.  **Manages Risk:** Feeds the final signal and its own live, tracked state (balance, drawdown) into the **RL Governor**, which decides the final, state-aware position size.
    5.  **Executes & Learns:** Places the trade and, upon closing it, updates its internal state and trains its online models with the outcome.
*   **Why it exists:** To act as the autonomous, decision-making entity for a single asset, integrating prediction, risk management, and continuous learning.

### 2.4. The AI Analyst (`singularity_engine.py`)

This is the strategic, "slow-thinking" brain of the system. It runs in parallel to the `MultiAssetTrader`.

*   **What it does:**
    *   **Monitors Performance:** Continuously reads the master performance log.
    *   **Adapts Weights:** Uses the `HedgeEnsembleManager` to dynamically increase the influence of models that are performing well and decrease the influence of those that are not.
    *   **Detects Drift:** Uses the `DriftDetector` to identify when a model's performance for a specific asset has statistically degraded.
    *   **Triggers Evolution:** When drift is detected, it immediately triggers the **Forge** to evolve a new model for that specific asset.
    *   **Proactive Forging:** It also runs a slow, continuous loop that proactively triggers the Forge for each asset in the universe over time, ensuring models are refreshed even before they decay.
*   **Why it exists:** To provide the crucial meta-cognitive ability for the system to monitor itself, learn from its mistakes, and trigger its own evolution.

### 2.5. The Evolutionary Engine (The Forge)

The Forge is not a single file, but a pipeline of scripts orchestrated by `forge/overlord/task_scheduler.py`.

*   **What it does:** This is the heart of the system's long-term adaptation. When triggered by the `SingularityEngine`, it performs a complete, end-to-end evolutionary cycle:
    1.  **Genetic Algorithm (`forge/blueprint_factory/genetic_algorithm.py`):** Creates a new generation of `ModelBlueprint`s (combinations of model architectures and features), seeded with the previous champion model (**Rebirth**).
    2.  **Bayesian Optimization (`forge/crucible/bayesian_optimizer.py`):** Takes the best blueprints from the genetic algorithm and fine-tunes their hyperparameters.
    3.  **Validation Gauntlet (`validation_gauntlet.py`):** Subjects the new, optimized candidate models to a battery of rigorous tests (like Walk-Forward Analysis) to ensure they are statistically robust and not just overfit.
    4.  **Saving:** Only if a model passes the gauntlet is it saved to the `models` directory, where it can be "hot-swapped" into the live trading loop.
*   **Why it exists:** To provide a robust, automated, and statistically sound process for creating new, superior trading models to replace those that have decayed.

---

## 3. Recent Core System Updates

The system has undergone several critical updates to fix logical flaws and align it with its core philosophy.

*   **Forge Integration:** The `SingularityEngine`'s adaptive loop was completely refactored. The old, simplistic `WFAManager` was removed and replaced with the true **Forge** pipeline. The system now correctly uses genetic algorithms and the validation gauntlet to evolve new models.
*   **RL Governor Statefulness:** A critical flaw was fixed where the RL Governor was operating without memory. The `V3Agent` is now fully stateful, tracking its own balance and drawdown, and feeding this live, accurate information to the Governor for every decision.
*   **HMM / Hydra Integration:** The HMM regime detector has been integrated back into the `V3Agent`. The agent now enriches its feature set with regime probabilities, providing crucial context to both the prediction models and the RL Governor. The training process for the RL Governor was also updated to include this context, eliminating a major training/execution mismatch.
*   **Performance Log Restoration:** A show-stopping bug was fixed where the performance log was never being written. This has been corrected, re-enabling the entire adaptive loop of the `SingularityEngine`.
*   **Efficient Data Ingestion:** The `trading_manager`'s data loop was upgraded from the inefficient `watch_trades` to the correct `watch_ohlcv`, ensuring the decision pipeline only runs once per candle.
*   **Code Pruning:** A comprehensive audit was performed, and numerous obsolete, redundant, and non-functional files (e.g., `hydra_pool.py`, `paper_trader.py`, and several scripts in the `forge/gauntlet` directory) were deleted to simplify the codebase and eliminate confusion.
*   **Champion Rebirth Fix:** A bug was fixed in the genetic algorithm, ensuring that the previous champion model is now correctly "rebirthed" into the new population, allowing for continuous, iterative improvement.
