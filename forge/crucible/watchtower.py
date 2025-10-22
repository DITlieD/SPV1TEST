import numpy as np
import pandas as pd
from scipy.stats import norm
from forge.crucible.elite_preservation import ElitePreservationSystem

class Watchtower:
    """
    The Arbiter. Monitors agent performance using robust statistical metrics,
    terminates underperforming agents, and preserves elite winners.

    ULTRA MFT Philosophy:
    - REAP the weak (balance → $0)
    - PRESERVE the strong (balance > $200)
    - EVOLVE the best (pass down elite DNA)
    """
    def __init__(self, confidence_threshold=0.90, min_trades=20, drawdown_limit=1.0):
        self.confidence_threshold = confidence_threshold
        self.min_trades = min_trades
        self.drawdown_limit = drawdown_limit

        # ELITE PRESERVATION: Track and save successful models
        self.elite_system = ElitePreservationSystem()

        print("[Watchtower] Initialized with Elite Preservation System.")
        elite_stats = self.elite_system.get_elite_stats()
        print(f"[Watchtower] Current Elites: {elite_stats['total_elites']} | Best Profit: +{elite_stats['best_profit_pct']:.2f}%")

    def calculate_psr(self, returns: pd.Series) -> float:
        """
        Calculates the Probabilistic Sharpe Ratio (PSR).
        PSR estimates the probability that the estimated Sharpe Ratio is > 0.
        """
        if len(returns) < self.min_trades:
            return 0.0 # Not enough data for a meaningful PSR

        sharpe_ratio = self.calculate_sharpe(returns)
        
        # Skewness and Kurtosis of returns
        skewness = returns.skew()
        kurtosis = returns.kurtosis() # Excess kurtosis

        # PSR calculation
        n = len(returns)
        psr_numerator = (sharpe_ratio * np.sqrt(n - 1))
        psr_denominator = np.sqrt(
            1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2
        )

        if psr_denominator == 0:
            return 0.0

        probabilistic_sharpe_ratio = norm.cdf(psr_numerator / psr_denominator)
        return probabilistic_sharpe_ratio

    def calculate_sharpe(self, returns: pd.Series, risk_free_rate=0.0) -> float:
        """Calculates the annualized Sharpe Ratio."""
        if returns.std() == 0:
            return 0.0
        
        # Assuming daily returns for annualization
        annualized_return = returns.mean() * 252
        annualized_std = returns.std() * np.sqrt(252)
        
        sharpe = (annualized_return - risk_free_rate) / annualized_std
        return sharpe

    def check_agent_lifecycle(self, agent, trade_logs):
        """
        Checks an agent's performance against the termination rules using real trade data.
        """
        agent_trades = [log for log in trade_logs if log.get("model_id") == agent.agent_id]

        # Rule 1: Death by Drawdown (using the agent's live balance)
        current_balance = agent.agent.virtual_balance
        initial_capital = agent.initial_capital
        if (initial_capital - current_balance) / initial_capital >= self.drawdown_limit:
            agent.is_active = False
            print(f"[Watchtower] REAPER: Agent {agent.agent_id} terminated due to excessive drawdown.")
            return

        # Rule 2: Death by Underperformance (PSR)
        if len(agent_trades) >= 50: # Sustained period of 50 trades
            returns = pd.Series([t['pnl_pct'] for t in agent_trades[-50:]]) # Check last 50 trades
            psr = self.calculate_psr(returns)
            
            if psr < 0.5: # PSR threshold
                agent.is_active = False
                print(f"[Watchtower] REAPER: Agent {agent.agent_id} terminated due to low PSR ({psr:.2f}) over the last 50 trades.")
                return
        elif len(agent_trades) >= self.min_trades:
            returns = pd.Series([t['pnl_pct'] for t in agent_trades])
            psr = self.calculate_psr(returns)
            
            if psr < self.confidence_threshold:
                agent.is_active = False
                print(f"[Watchtower] REAPER: Agent {agent.agent_id} terminated due to low PSR ({psr:.2f}).")
                return

    def judge_agents(self, all_agents: dict, trade_logs: list) -> list:
        """
        Iterates through all active agents, checks their lifecycle status,
        preserves elites, and returns a list of agents that have been terminated.

        ULTRA MFT:
        - Check for elite status (balance > $200)
        - Preserve successful models and their DNA
        - Reap failed models (balance → $0)
        """
        terminated_agents = []
        promoted_elites = []

        for symbol, agent_list in all_agents.items():
            for agent in agent_list:
                if agent.is_active:
                    # ELITE PRESERVATION: Check if model has grown capital
                    try:
                        model_id = getattr(agent, 'current_model_id', None) or getattr(agent, 'agent_id', None)
                        if model_id:
                            was_promoted = self.elite_system.check_and_preserve(
                                agent=agent,
                                model_id=model_id,
                                symbol=symbol,
                                initial_capital=200.0  # ULTRA MFT initial capital
                            )
                            if was_promoted:
                                promoted_elites.append((symbol, model_id))
                    except Exception as e:
                        print(f"[Watchtower] Error checking elite status for {symbol}: {e}")

                    # REAPING: Check termination rules
                    self.check_agent_lifecycle(agent, trade_logs)
                    if not agent.is_active:
                        terminated_agents.append((symbol, agent.agent_id))

        if promoted_elites:
            print(f"[Watchtower] 🏆 {len(promoted_elites)} new elites promoted this cycle!")

        return terminated_agents
    
    def promote_elite(self, agent, trade_logs) -> bool:
        """Checks if an agent's performance qualifies it for the Elite Pool."""
        agent_trades = [log for log in trade_logs if log.get("model_id") == agent.agent_id]

        if len(agent_trades) >= self.min_trades:
            returns = pd.Series([t['pnl_pct'] for t in agent_trades])
            psr = self.calculate_psr(returns)
            if psr > 0.99: # High threshold for promotion
                print(f"[Watchtower] PROMOTION: Agent {agent.agent_id} achieved elite status with PSR ({psr:.2f}).")
                return True
        return False
