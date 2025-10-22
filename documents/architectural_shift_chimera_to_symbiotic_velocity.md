# Architectural Shift: From Adversarial (Chimera) to Symbiotic Velocity (SDE) Evolution

## 1. Executive Summary

This document outlines the recent, fundamental architectural change to the GP 2.0 evolutionary engine, as implemented by the `updatev17.txt` blueprint. The system's core evolutionary objective has shifted from an **Adversarial Fitness** model (the Chimera/Apex Predator engine) to a **Symbiotic Velocity Fitness** model.

This change replaces a complex, game-theory-based evaluation with a more direct and efficient model focused on two primary goals:
1.  **Velocity:** Achieving a predefined profit target as quickly as possible.
2.  **Symbiosis:** Mimicking the behavior of a sophisticated, pre-trained "teacher" model (the SDE Implicit Brain).

This document details both the old and new paradigms, the rationale for the change, and the impact it had on the codebase.

## 2. The Old Paradigm: Adversarial Fitness (Chimera/Apex Predator)

The previous system was designed to evolve strategies that were robust in a competitive environment. It treated the market as a game and rewarded strategies that could not only find profitable opportunities but also actively exploit the weaknesses of other, more common strategies.

### Workflow

1.  **Define an Opponent:** The system used a generic, pre-defined strategy (e.g., `simple_rsi_opponent`) to represent a "typical" or predictable trader.
2.  **Observe the Market:** It analyzed the recent price action to determine the market's statistical "personality" or "signature" (e.g., choppy, trending, volatile).
3.  **Infer Opponent Parameters (Chimera Inference):** This was the core of the Chimera engine. It ran an optimization to find the parameters for the generic opponent that would best explain the observed market signature. For example, it might determine that the market's behavior was best explained by an RSI strategy with a period of 29.
4.  **Calculate Dual Fitness:** When evaluating an evolved strategy, it was judged on two criteria:
    *   **Standard Fitness:** Its raw profitability (PnL, Sharpe Ratio) against the market data.
    *   **Apex Predator Fitness:** Its performance *relative to the inferred opponent*. A strategy scored highly if it successfully traded against the opponent's likely moves.
5.  **Combined Score:** These two scores were combined into a "dual fitness" metric that guided the evolution.

**The `Chimera inference failed` log message was a direct result of this component still being active in `singularity_engine.py` after its core logic was removed from the fitness function.**

## 3. The New Paradigm: Symbiotic Velocity Fitness

The `updatev17.txt` blueprint replaced the adversarial model with a new philosophy focused on speed and guided learning. The fitness of an evolved strategy is now a weighted combination of two new components.

### Fitness Components

1.  **Velocity Fitness (`velocity_fitness`):**
    *   **Objective:** To reach a profit target in the shortest time possible.
    *   **Mechanism:** The new `VectorizedBacktester` simulates trades with a predefined capital and profit goal (e.g., grow $1000 by 2%).
    *   **Scoring:**
        *   If the target is **achieved**, the fitness score is high and inversely proportional to the time it took (faster is better).
        *   If the target is **not achieved**, the fitness score is simply its final PnL percentage, which can be negative.
    *   This creates a strong evolutionary pressure to find strategies that generate profit quickly and efficiently.

2.  **Symbiotic Fitness (`mimicry_score`):**
    *   **Objective:** To behave like a sophisticated, pre-trained "teacher" model.
    *   **Mechanism:** The system loads a complex "Implicit Brain" model (SDE), which has already been trained to understand market dynamics. For each data point, this model provides a probability of whether to buy or sell.
    *   **Scoring:** The evolved GP strategy is rewarded for making decisions that align with the high-probability outputs of the teacher model. The score is calculated as the inverse of the log-loss between the GP's signals and the teacher's probabilities, ensuring that mimicry is rewarded.

### The Combined Symbiotic Velocity Fitness

The final fitness score is a weighted average of these two components:

`Total Fitness = (Velocity Fitness * 0.7) + (Mimicry Score * 0.3) - Parsimony Penalty`

This new objective encourages the GP to find strategies that are not only fast and profitable but also share the sophisticated characteristics of the powerful SDE teacher model.

## 4. Rationale for the Change

*   **Direct Focus:** The new system is more directly aligned with the primary goal of rapid capital growth ("Velocity").
*   **Sophisticated Guidance:** Instead of competing against a *simple* opponent, the GP is now guided by a *complex* teacher, which can steer the evolution towards discovering more nuanced and effective market patterns.
*   **Computational Efficiency:** The new fitness calculation is more direct and less computationally expensive than the multi-stage Chimera inference and dual-fitness simulation.

## 5. Codebase Impact

This architectural shift required significant changes to the core evolutionary files:

*   **`singularity_engine.py`:** The call to the `_chimera_loop` was removed.
*   **`config.py`:** Centralized all key parameters for the new system into the `ACN_CONFIG` dictionary.
*   **`forge/crucible/numba_backtester.py`:** Replaced entirely with the new `VectorizedBacktester` that calculates `velocity_fitness`.
*   **`forge/evolution/strategy_synthesizer.py`:** The core fitness evaluation logic was replaced with the new symbiotic calculation.
*   **`forge/overlord/task_scheduler.py`:** Updated to orchestrate the new workflow, including loading the SDE model and initializing the new backtester and synthesizer.
*   **`forge/evolution/gp_primitives.py`:** A new file created to define the building blocks for the GP.
