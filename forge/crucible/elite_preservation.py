"""
Elite Preservation System
=========================
Tracks and preserves successful live trading models that grow capital beyond initial balance.

Philosophy: "The winners MUST survive and pass down their DNA"
- Models that grow $200 → $200+ are ELITE
- Their DNA is preserved forever
- Their genetics seed future generations
- Evolution favors proven winners, not just backtested ones
"""

import os
import json
import dill
from datetime import datetime
from typing import Dict, List, Optional
import logging


class ElitePreservationSystem:
    """
    Monitors live trading performance and preserves successful models.

    ULTRA MFT Philosophy:
    - If a model grows capital in LIVE trading, it's proven
    - Save the model file
    - Save the DNA (GP tree)
    - Save performance metrics
    - Pass DNA to next generation
    """

    def __init__(self, elite_dir: str = "models/elite", logger=None):
        self.elite_dir = elite_dir
        self.logger = logger or logging.getLogger(__name__)

        # Create elite directory structure
        os.makedirs(self.elite_dir, exist_ok=True)
        os.makedirs(os.path.join(self.elite_dir, "dna"), exist_ok=True)
        os.makedirs(os.path.join(self.elite_dir, "models"), exist_ok=True)

        # Elite registry file
        self.registry_path = os.path.join(self.elite_dir, "elite_registry.json")
        self.elite_registry = self._load_registry()

        self.logger.info(f"[Elite Preservation] Initialized. Current elites: {len(self.elite_registry)}")

    def _load_registry(self) -> Dict:
        """Load the elite model registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"[Elite] Failed to load registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save the elite model registry to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.elite_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"[Elite] Failed to save registry: {e}")

    def check_and_preserve(self, agent, model_id: str, symbol: str, initial_capital: float = 200.0):
        """
        Check if an agent has grown capital beyond initial and preserve if elite.

        Args:
            agent: The live trading agent
            model_id: Model identifier
            symbol: Trading symbol
            initial_capital: Initial balance (default $200)

        Returns:
            bool: True if model was promoted to elite, False otherwise
        """
        current_balance = agent.virtual_balance

        # Check if model has grown capital (is profitable)
        if current_balance > initial_capital:
            profit = current_balance - initial_capital
            profit_pct = (profit / initial_capital) * 100.0

            # Check if already elite
            if model_id in self.elite_registry:
                # Update if performance improved
                old_profit_pct = self.elite_registry[model_id].get('profit_pct', 0)
                if profit_pct > old_profit_pct:
                    self.logger.info(f"[Elite] {model_id} improved: ${initial_capital:.2f} → ${current_balance:.2f} (+{profit_pct:.2f}%)")
                    self._update_elite(model_id, symbol, current_balance, profit_pct)
                return False  # Already elite

            # NEW ELITE DISCOVERED!
            self.logger.info(f"[Elite] 🏆 NEW ELITE DISCOVERED: {model_id}")
            self.logger.info(f"[Elite] Performance: ${initial_capital:.2f} → ${current_balance:.2f} (+{profit_pct:.2f}%)")

            # Preserve the elite model
            self._preserve_elite(agent, model_id, symbol, initial_capital, current_balance, profit_pct)
            return True

        return False

    def _preserve_elite(self, agent, model_id: str, symbol: str, initial_capital: float,
                       current_balance: float, profit_pct: float):
        """
        Preserve an elite model's DNA and performance data.
        """
        try:
            # Extract DNA if available (for GP models)
            dna = None
            model_architecture = "Unknown"

            # Try to get DNA from model
            if hasattr(agent, 'current_model') and agent.current_model:
                model = agent.current_model
                if hasattr(model, 'get_dna'):
                    dna = model.get_dna()
                    model_architecture = dna.get('architecture', 'Unknown') if dna else 'Unknown'

            # Create elite entry
            elite_entry = {
                'model_id': model_id,
                'symbol': symbol,
                'architecture': model_architecture,
                'initial_capital': initial_capital,
                'peak_balance': current_balance,
                'profit_pct': profit_pct,
                'promoted_at': datetime.now().isoformat(),
                'has_dna': dna is not None,
                'dna_path': None
            }

            # Save DNA if available
            if dna:
                dna_filename = f"{model_id}_{symbol}_dna.pkl"
                dna_path = os.path.join(self.elite_dir, "dna", dna_filename)

                with open(dna_path, 'wb') as f:
                    dill.dump(dna, f)

                elite_entry['dna_path'] = dna_path
                self.logger.info(f"[Elite] DNA saved: {dna_path}")
            else:
                self.logger.warning(f"[Elite] No DNA available for {model_id}")

            # Add to registry
            self.elite_registry[model_id] = elite_entry
            self._save_registry()

            self.logger.info(f"[Elite] ✅ {model_id} preserved successfully!")

        except Exception as e:
            self.logger.error(f"[Elite] Failed to preserve {model_id}: {e}", exc_info=True)

    def _update_elite(self, model_id: str, symbol: str, current_balance: float, profit_pct: float):
        """Update an existing elite model's performance."""
        if model_id in self.elite_registry:
            self.elite_registry[model_id]['peak_balance'] = current_balance
            self.elite_registry[model_id]['profit_pct'] = profit_pct
            self.elite_registry[model_id]['last_updated'] = datetime.now().isoformat()
            self._save_registry()

    def get_best_elite_dna(self, symbol: Optional[str] = None, architecture: str = "GP2_Evolved", model_instance_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get DNA from the best performing elite model.

        Args:
            symbol: Filter by symbol (optional)
            architecture: Filter by architecture (default: GP2_Evolved)
            model_instance_id: Filter by model instance ID for lineage-specific DNA (UPDATE V5)

        Returns:
            DNA dictionary or None
        """
        # Filter elites
        candidates = []
        for model_id, entry in self.elite_registry.items():
            if entry.get('has_dna', False):
                if symbol and entry.get('symbol') != symbol:
                    continue
                if architecture and entry.get('architecture') != architecture:
                    continue
                # UPDATE V5: Filter by model instance ID if provided (for lineage-specific inheritance)
                if model_instance_id and entry.get('model_id', '').startswith(model_instance_id):
                    # Match if the elite model_id starts with the instance ID
                    pass
                elif model_instance_id:
                    # If model_instance_id is provided but doesn't match, skip this entry
                    continue
                candidates.append(entry)

        if not candidates:
            filter_desc = f"symbol={symbol}, architecture={architecture}"
            if model_instance_id:
                filter_desc += f", instance={model_instance_id}"
            self.logger.info(f"[Elite] No elite DNA found for {filter_desc}")
            return None

        # Sort by profit percentage (best first)
        candidates.sort(key=lambda x: x.get('profit_pct', 0), reverse=True)
        best = candidates[0]

        # Load DNA
        dna_path = best.get('dna_path')
        if dna_path and os.path.exists(dna_path):
            try:
                with open(dna_path, 'rb') as f:
                    dna = dill.load(f)

                self.logger.info(f"[Elite] Loaded DNA from {best['model_id']} (profit: +{best['profit_pct']:.2f}%)")
                return dna
            except Exception as e:
                self.logger.error(f"[Elite] Failed to load DNA from {dna_path}: {e}")
                return None

        return None

    def get_elite_stats(self) -> Dict:
        """Get statistics about elite models."""
        if not self.elite_registry:
            return {
                'total_elites': 0,
                'avg_profit_pct': 0,
                'best_profit_pct': 0,
                'total_dna_saved': 0
            }

        profits = [e.get('profit_pct', 0) for e in self.elite_registry.values()]
        dna_count = sum(1 for e in self.elite_registry.values() if e.get('has_dna', False))

        return {
            'total_elites': len(self.elite_registry),
            'avg_profit_pct': sum(profits) / len(profits) if profits else 0,
            'best_profit_pct': max(profits) if profits else 0,
            'total_dna_saved': dna_count,
            'elites': list(self.elite_registry.values())
        }
