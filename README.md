# Singularity Protocol: A Multi-Agent Evolutionary Trading System

This is a multi-agent, evolutionary trading system designed for continuous, autonomous operation in the cryptocurrency markets.

## Core Architecture

The system is composed of several distinct components that work together to evolve, deploy, and manage a population of trading agents.

- **CrucibleEngine (`crucible_engine.py`):** The master orchestrator and the heart of the live system. It runs 24/7, handling data fetching, managing the live trading agents, and executing the main event loops.

- **SingularityEngine (`singularity_engine.py`):** The evolution manager. It reads the `DEPLOYMENT_STRATEGY` from `config.py` (e.g., 4 agents for BTC, 2 for ETH) and proactively commissions the Forge to create new agents to meet this strategy.

- **The Arena:** This is not a single file, but a concept managed by the `CrucibleEngine`. It holds the population of all active, independent `V3Agent` instances that are currently trading for each asset.

- **The Forge (`forge_worker.py`):** A dedicated R&D process that runs in a separate process pool. It uses a Genetic Algorithm (`MODEL_BLUEPRINT_GA`) to evolve, train, and validate new trading model strategies. This heavy computational work is kept separate to avoid impacting live trading.

- **The Watchtower:** A loop within the `CrucibleEngine` that acts as a survival-of-the-fittest mechanism. It constantly monitors the performance of all agents in the Arena and culls underperforming ones. The `SingularityEngine` then commissions the Forge to create replacements.

- **Cerebellum (`cerebellum_core/`):** A high-performance execution core written in Rust. It communicates with the main Python application via ZMQ. All trade execution requests are sent to Cerebellum, which handles the low-level, high-speed interaction with the exchange API. This separation ensures that trade execution is fast, reliable, and not subject to Python's Global Interpreter Lock (GIL).
