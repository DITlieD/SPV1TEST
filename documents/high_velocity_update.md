# High-Velocity Configuration Update

This document details the series of changes implemented to re-configure the Singularity Protocol for a high-velocity, aggressive capital growth strategy. The default parameters were balanced for steady, conservative operation. This update tunes the system to prioritize faster evolution, increased trade frequency, and higher risk per trade to accelerate the growth of the initial capital base.

## Summary of Changes

The update focuses on three key areas:
1.  **Accelerating the Rate of Evolution:** Creating and culling models more frequently.
2.  **Increasing Trading Aggression:** Taking larger positions and aiming for higher profits on each trade.
3.  **Increasing Trade Frequency:** Forcing the system to enter and exit trades more rapidly.

---

## Detailed Implementation

### 1. Accelerated Evolution Cycle

The "Forge & Cull" cycle, which governs how quickly the system can improve its model population, has been significantly accelerated.

#### A. Watchtower Judgment Frequency
- **File Changed:** `crucible_engine.py`
- **Method:** `_watchtower_loop`
- **Change:** The judgment frequency for the Watchtower was changed from every 5 minutes to every **15 minutes**.
- **Original Code:** `await asyncio.sleep(300)`
- **New Code:** `await asyncio.sleep(900)`
- **Reasoning:** While most parameters were tuned for speed, this change was a specific user request to give models a slightly longer evaluation window before being judged, balancing speed with the need for sufficient performance data.

#### B. New Model Forging Speed
- **File Changed:** `singularity_engine.py`
- **Method:** `_proactive_forge_loop`
- **Change:** The delay between submitting forge tasks for each asset in the round-robin cycle was reduced from 60 seconds to **10 seconds**.
- **Original Code:** `await asyncio.sleep(60)`
- **New Code:** `await asyncio.sleep(10)`
- **Reasoning:** This dramatically increases the rate at which new "challenger" models are created, providing more genetic diversity for the evolutionary algorithm and ensuring the system is constantly experimenting.

---

### 2. Increased Trading Aggression

To prioritize capital growth, the agent's risk-taking parameters have been made more aggressive.

#### A. Capital Allocation per Trade (Kelly Criterion)
- **File Changed:** `forge/core/agent.py`
- **Method:** `decide_and_execute`
- **Change:** The `fraction` parameter for the `calculate_fractional_kelly` function was doubled, from `0.25` to **`0.5`**.
- **Original Code:** `kelly_fraction = calculate_fractional_kelly(win_prob, risk_reward, fraction=0.25)`
- **New Code:** `kelly_fraction = calculate_fractional_kelly(win_prob, risk_reward, fraction=0.5)`
- **Reasoning:** This is the most significant change for accelerating profit. The agent will now allocate twice as much capital to every trade it takes. This amplifies both gains and losses, representing a higher-risk, higher-reward strategy.

#### B. Risk/Reward Ratio
- **File Changed:** `forge/core/agent.py`
- **Methods:** `decide_and_execute` and `_execute_trade`
- **Change 1 (Decision):** The target `risk_reward` variable was increased from `2.0` to **`3.0`**.
- **Change 2 (Execution):** The `take_profit` calculation was updated to match the new 3:1 target, changing from `4.0 * atr` to **`6.0 * atr`**.
- **Reasoning:** The system will now target profits that are three times the size of its potential loss (defined by the 2-ATR stop-loss). This aims for larger, more impactful winning trades.

---

### 3. Increased Trade Frequency

To ensure capital is not tied up in stagnant positions, the agent's patience has been reduced.

#### A. Adaptive Timeout Reduction
- **File Changed:** `forge/core/agent.py`
- **Method:** `decide_and_execute`
- **Change:** The adaptive timeout logic, which forces an exit if a trade isn't progressing, was made significantly more aggressive. The time limits were cut in half across the board.
- **Original Range:** 15 to 60 minutes.
- **New Range:** **7 to 30 minutes**.
- **Reasoning:** The agent will now exit trades that are moving sideways much more quickly. This increases capital velocity, freeing up funds to be deployed on new, potentially more profitable opportunities that may arise.
