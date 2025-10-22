# The Archangel Protocol: Definitive Technical Blueprint

This document provides a comprehensive, end-to-end technical explanation of the Archangel Protocol. It covers not only the strategy synthesis pipeline but also the higher-level orchestration engines that manage the lifecycle of live, autonomous trading agents.

## I. System Architecture: A Multi-Agent Ecosystem

The system is not a single monolithic program but a multi-agent ecosystem designed for continuous, autonomous operation. It is composed of several key modules that interact to discover, validate, deploy, and manage trading agents.

*   **The Crucible Engine (`crucible_engine.py`):** The master orchestrator. It initializes all other components and runs the main asynchronous event loops.
*   **The Forge (`task_scheduler.py`):** The R&D lab. Its sole purpose is to run a "forge cycle" to create and validate a single new trading strategy from scratch.
*   **The Arena (`arena_manager.py`):** The live trading floor. It holds the population of active trading agents for each asset.
*   **The Watchtower (`watchtower.py`):** The performance monitor and reaper. It judges the live agents and eliminates underperformers.
*   **The Singularity Engine (`singularity_engine.py`):** The portfolio-level risk and execution manager.

---

## II. The Forge: Autonomous Strategy Synthesis

The Forge is the starting point for any new agent. It is a highly sophisticated pipeline designed to mimic and automate the entire workflow of a quantitative researcher.

*(Note: This section is a condensed version of the detailed `System_Technical_Blueprint.md`. Please refer to that document for deeper detail on the items below.)*

#### **A. The Forge Cycle (`run_single_forge_cycle`)**

This function is the core of the Forge. When called, it executes the following steps:

1.  **Data Ingestion & Feature Engineering:** Gathers and aligns multi-timeframe data and generates a rich feature set, including:
    *   Standard Technical Indicators
    *   Probabilistic Features (Particle Filter)
    *   On-Chain Intelligence (Simulated)
    *   **Relational Dynamics (Graph Neural Network):** This is a key step where a GAT model analyzes the entire market to provide cross-asset context to each individual asset's feature set.

2.  **Environment-Aware Synthesis (GP 2.0):**
    *   It first classifies the current market **regime** (e.g., Bull Volatile).
    *   It then uses **Genetic Programming** to evolve a population of thousands of unique trading strategies (as code) specifically tailored to thrive in that environment.

3.  **Multi-Layered Validation Gauntlet:** The single best-evolved strategy is subjected to a brutal series of tests:
    *   **High-Fidelity Backtesting:** Simulation against realistic market frictions (slippage, fees, liquidity).
    *   **Causal Inference Gauntlet:** A critical filter that uses causal models (`dowhy`) to ensure the strategy's logic is based on genuine cause-and-effect relationships, not spurious correlations.

4.  **Registration & Analysis:**
    *   **AI Analyst:** A simulated LLM translates the machine-generated strategy into a human-readable explanation.
    *   **Model Registry:** If the strategy passes all gauntlets, its core components (the uncompiled strategy tree and its primitive set) are saved to the Model Registry, officially becoming a "forged" model.

---

## III. The Arena: Live Agent Management

The Arena is where forged models become live agents and compete.

#### **A. The `ArenaManager`**

*   **Function:** It maintains a dictionary of all active `CrucibleAgent` objects, segregated by asset symbol.
*   **Agent Lifecycle:** It handles the addition of new agents forged by the Forge and the removal of underperforming agents reaped by the Watchtower.
*   **Data Distribution:** It receives the latest market data (including GNN-enriched features) from the `CrucibleEngine` and distributes it to the relevant agents for signal generation.

#### **B. The `CrucibleAgent`**

*   **Structure:** This is a wrapper class that represents a live, competing agent. It contains:
    *   A unique `agent_id`.
    *   An instance of a `V3Agent`, which is the underlying class that loads a forged model from the registry and generates the raw trade signals.

---

## IV. The Watchtower: The Reaper and Evolver

The Watchtower ensures "survival of the fittest" among the live agents in the Arena.

#### **A. The Judgment Loop (`_watchtower_loop`)**

*   **Frequency:** This loop runs periodically (e.g., every 5 minutes).
*   **Mechanism:**
    1.  **Performance Analysis:** It loads the `performance_log.jsonl`, which contains a record of every trade made by every agent.
    2.  **Judgment:** It uses a set of performance criteria (e.g., Sharpe ratio, drawdown, win rate over a recent window) to identify the worst-performing agent for each asset.
    3.  **Reaping:** If an agent is deemed an underperformer, the Watchtower issues a command to the `ArenaManager` to **reap** (remove) it from the live population.

#### **B. Triggering Evolution**

*   **The Core Loop:** Reaping an agent is not just about removal; it is the trigger for evolution.
*   **Mechanism:** Immediately after reaping an agent, the Watchtower calls the Forge (`_run_forge_sync`) to create a new, hopefully better, agent to take its place.
*   **Genetic Progression:** It increments a `generation_counter` for the asset, ensuring that the new agent has a unique ID that tracks its evolutionary lineage (e.g., `BTCUSDT-gen1-0`, `BTCUSDT-gen2-1`, etc.).

This continuous cycle of **Judge -> Reap -> Evolve** is the engine that drives the system's hyper-adaptation.

---

## V. The Singularity Engine: Portfolio-Level Intelligence

The Singularity Engine sits above the individual agents and makes portfolio-level decisions.

*   **Function:** It receives the trade signals generated by *all* agents in the Arena.
*   **Risk Management:** It analyzes the combined signals. If multiple agents signal a "BUY" on highly correlated assets simultaneously, the Singularity Engine might moderate the total position size to avoid excessive directional risk.
*   **Execution:** It is responsible for the final execution of trades, potentially using sophisticated execution algorithms (e.g., TWAP/VWAP) to minimize market impact.
*   **Systemic Safeguards:** It can receive signals from the **Immune System** (the anomaly detector). If an anomaly is detected, the Singularity Engine can trigger a system-wide risk reduction, overriding the individual agents and potentially closing all open positions.

This complete, closed-loop system automates the entire process from research and discovery to live deployment, performance monitoring, and evolution, creating a truly autonomous and adaptive trading ecosystem.
