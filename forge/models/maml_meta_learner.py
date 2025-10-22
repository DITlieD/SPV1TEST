"""
VELOCITY UPGRADE: MAML Meta-Learning for Rapid Regime Adaptation
=================================================================
Model-Agnostic Meta-Learning (MAML) for trading strategies.

Concept:
- Instead of learning a single strategy, MAML learns *how to learn* quickly
- Model is trained across diverse historical regimes
- Can adapt to new regimes within just a few bars of data
- Eliminates lag when market conditions shift

Implementation:
- Uses learn2learn library for MAML
- PyTorch-based neural network policies
- Inner loop: Fast adaptation to specific regime
- Outer loop: Meta-optimization across all regimes

Result: Near-instant adaptation to regime changes without waiting for full Forge cycle.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
import os

try:
    import learn2learn as l2l
    MAML_AVAILABLE = True
except ImportError:
    MAML_AVAILABLE = False
    logging.warning("learn2learn not available. MAML meta-learning disabled. Install with: pip install learn2learn")


class TradingPolicy(nn.Module):
    """
    Neural network trading policy for MAML.

    Architecture:
    - Input: Market features
    - Hidden: 2 layers with ReLU
    - Output: 3 actions (HOLD, BUY, SELL)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(TradingPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions: HOLD, BUY, SELL
        )

    def forward(self, x):
        """
        Args:
            x: Features tensor (batch_size, input_dim)

        Returns:
            Action logits (batch_size, 3)
        """
        return self.network(x)

    def get_action(self, features: np.ndarray) -> int:
        """
        Get discrete action from features.

        Args:
            features: Feature array

        Returns:
            Action index (0=HOLD, 1=BUY, 2=SELL)
        """
        self.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            logits = self.forward(features_tensor)
            action = torch.argmax(logits, dim=1).item()
        return action


class MAMLMetaLearner:
    """
    Meta-learning manager for rapid regime adaptation.

    Workflow:
    1. Collect historical data across different regimes
    2. Meta-train policy to adapt quickly across regimes
    3. At deployment, adapt policy to current regime in 5-10 bars
    4. Continue trading with adapted policy
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        device: str = 'cuda',
        logger=None
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer size
            meta_lr: Meta-learning rate (outer loop)
            inner_lr: Inner adaptation learning rate
            inner_steps: Number of gradient steps for inner adaptation
            device: 'cuda' or 'cpu'
            logger: Logger instance
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.logger = logger or logging.getLogger(__name__)
        self.input_dim = input_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        if not MAML_AVAILABLE:
            self.logger.error("[MAML] learn2learn not installed. MAML functionality disabled.")
            self.policy = None
            self.maml = None
            return

        # Initialize policy
        self.policy = TradingPolicy(input_dim, hidden_dim).to(self.device)

        # Wrap with MAML
        self.maml = l2l.algorithms.MAML(self.policy, lr=inner_lr, first_order=False)

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.maml.parameters(), lr=meta_lr)

        self.logger.info(f"[MAML] Initialized with input_dim={input_dim}, device={self.device}")

    def create_regime_tasks(
        self,
        historical_data: Dict[str, pd.DataFrame],
        regime_labels: Dict[str, np.ndarray],
        n_tasks: int = 10
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create training tasks from historical regimes.

        Each task represents a different market regime.

        Args:
            historical_data: {symbol: DataFrame with features}
            regime_labels: {symbol: regime labels from clustering}
            n_tasks: Number of tasks to create

        Returns:
            List of (support_set, query_set) tuples
        """
        tasks = []

        # Get unique regimes across all symbols
        all_regimes = []
        for symbol, labels in regime_labels.items():
            unique_regimes = np.unique(labels)
            all_regimes.extend([(symbol, r) for r in unique_regimes if r != -1])

        # Sample n_tasks regimes
        sampled_regimes = np.random.choice(len(all_regimes), min(n_tasks, len(all_regimes)), replace=False)

        for idx in sampled_regimes:
            symbol, regime = all_regimes[idx]

            # Get data for this regime
            df = historical_data[symbol]
            labels = regime_labels[symbol]

            # Filter to this regime
            regime_mask = labels == regime
            regime_data = df[regime_mask]

            if len(regime_data) < 20:
                continue  # Not enough data

            # Split into support (training) and query (testing) sets
            split_idx = len(regime_data) // 2
            support_data = regime_data.iloc[:split_idx]
            query_data = regime_data.iloc[split_idx:]

            # Convert to tensors
            support_features = torch.FloatTensor(support_data.select_dtypes(include=np.number).values).to(self.device)
            query_features = torch.FloatTensor(query_data.select_dtypes(include=np.number).values).to(self.device)

            # Create synthetic labels based on returns (simplified)
            support_returns = support_data['close'].pct_change().fillna(0).values
            query_returns = query_data['close'].pct_change().fillna(0).values

            # Labels: 0=HOLD, 1=BUY (if next return positive), 2=SELL (if negative)
            support_labels = torch.LongTensor(
                [1 if r > 0.001 else (2 if r < -0.001 else 0) for r in support_returns]
            ).to(self.device)
            query_labels = torch.LongTensor(
                [1 if r > 0.001 else (2 if r < -0.001 else 0) for r in query_returns]
            ).to(self.device)

            tasks.append(((support_features, support_labels), (query_features, query_labels)))

        self.logger.info(f"[MAML] Created {len(tasks)} regime adaptation tasks")
        return tasks

    def meta_train(self, tasks: List[Tuple], n_iterations: int = 100):
        """
        Meta-train the policy across different regime tasks.

        Args:
            tasks: List of (support_set, query_set) tuples
            n_iterations: Number of meta-training iterations
        """
        if not MAML_AVAILABLE or self.maml is None:
            self.logger.error("[MAML] Cannot train - MAML not available")
            return

        self.logger.info(f"[MAML] Starting meta-training for {n_iterations} iterations...")

        for iteration in range(n_iterations):
            meta_loss = 0.0

            # Sample batch of tasks
            batch_size = min(4, len(tasks))
            task_batch = np.random.choice(len(tasks), batch_size, replace=False)

            for task_idx in task_batch:
                (support_x, support_y), (query_x, query_y) = tasks[task_idx]

                # Clone model for inner adaptation
                learner = self.maml.clone()

                # Inner loop: Adapt to support set
                for _ in range(self.inner_steps):
                    support_logits = learner(support_x)
                    support_loss = nn.CrossEntropyLoss()(support_logits, support_y)
                    learner.adapt(support_loss)

                # Outer loop: Evaluate on query set
                query_logits = learner(query_x)
                query_loss = nn.CrossEntropyLoss()(query_logits, query_y)
                meta_loss += query_loss

            # Meta-update
            meta_loss /= batch_size
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            if (iteration + 1) % 10 == 0:
                self.logger.info(f"[MAML] Iteration {iteration + 1}/{n_iterations}, Meta-Loss: {meta_loss.item():.4f}")

        self.logger.info("[MAML] Meta-training complete")

    def fast_adapt(self, recent_data: pd.DataFrame, n_steps: int = 5) -> TradingPolicy:
        """
        Rapidly adapt the meta-learned policy to current regime.

        This is the key feature: Adapt in just 5-10 bars of new data.

        Args:
            recent_data: Recent market data (last 10-20 bars)
            n_steps: Number of adaptation steps

        Returns:
            Adapted policy ready for trading
        """
        if not MAML_AVAILABLE or self.maml is None:
            self.logger.error("[MAML] Cannot adapt - MAML not available")
            return self.policy

        self.logger.info(f"[MAML] Fast-adapting to current regime with {len(recent_data)} bars...")

        # Clone the meta-learned policy
        adapted_policy = self.maml.clone()

        # Prepare data
        features = torch.FloatTensor(
            recent_data.select_dtypes(include=np.number).values
        ).to(self.device)

        # Create synthetic labels from returns
        returns = recent_data['close'].pct_change().fillna(0).values
        labels = torch.LongTensor(
            [1 if r > 0.001 else (2 if r < -0.001 else 0) for r in returns]
        ).to(self.device)

        # Adapt
        for step in range(n_steps):
            logits = adapted_policy(features)
            loss = nn.CrossEntropyLoss()(logits, labels)
            adapted_policy.adapt(loss)

        self.logger.info(f"[MAML] Adaptation complete. Policy ready for current regime.")
        return adapted_policy.module  # Return the inner model

    def save(self, path: str):
        """Save the meta-learned policy."""
        if self.policy is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'input_dim': self.input_dim
        }, path)
        self.logger.info(f"[MAML] Meta-learned policy saved to {path}")

    def load(self, path: str):
        """Load a meta-learned policy."""
        if not os.path.exists(path):
            self.logger.warning(f"[MAML] No saved policy found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.logger.info(f"[MAML] Meta-learned policy loaded from {path}")


class MAMLIntegrationHelper:
    """
    Helper class to integrate MAML into the existing trading system.

    Usage:
    1. During initial training: Collect historical regime data
    2. Meta-train MAML policy across regimes
    3. At runtime: Fast-adapt to current regime in 5-10 bars
    4. Use adapted policy for trading decisions
    """

    @staticmethod
    def should_use_maml() -> bool:
        """Check if MAML is available and should be used."""
        return MAML_AVAILABLE

    @staticmethod
    def create_maml_model(feature_names: List[str]) -> MAMLMetaLearner:
        """
        Factory method to create MAML model with appropriate configuration.

        Args:
            feature_names: List of feature names

        Returns:
            Configured MAMLMetaLearner
        """
        input_dim = len(feature_names)
        return MAMLMetaLearner(
            input_dim=input_dim,
            hidden_dim=64,
            meta_lr=0.001,
            inner_lr=0.01,
            inner_steps=5,
            device='cuda'
        )

    @staticmethod
    def get_status() -> Dict[str, any]:
        """Get MAML system status."""
        return {
            'available': MAML_AVAILABLE,
            'library_installed': MAML_AVAILABLE,
            'requires': 'learn2learn' if not MAML_AVAILABLE else None
        }
