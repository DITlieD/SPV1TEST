import numpy as np
import os
import joblib
from typing import List, Dict

class HedgeEnsembleManager:
    """
    Manages the dynamic weighting of an ensemble of models using an EMA of performance.
    """
    def __init__(self, model_ids: List[str], learning_rate: float = 0.1, persistence_path: str = 'models/hedge_weights.joblib'):
        self.model_ids = model_ids if model_ids else []
        self.learning_rate = learning_rate # Alpha for EMA
        self.persistence_path = persistence_path
        self.performance_ema = self._load_state() if self.model_ids else {}

    def _initialize_weights(self):
        print("[Hedge] Initializing performance EMAs.")
        self.performance_ema = {model_id: 0.5 for model_id in self.model_ids} # Start with neutral performance

    def get_weights(self) -> Dict[str, float]:
        """
        Returns the current weights as a dictionary mapping model_id to its weight.
        """
        if not self.performance_ema:
            return {}
        
        # Apply softmax to performance scores to get weights
        scores = np.array(list(self.performance_ema.values()))
        exp_scores = np.exp(scores - np.max(scores)) # Stable softmax
        weights = exp_scores / np.sum(exp_scores)
        
        return dict(zip(self.performance_ema.keys(), weights))

    def update_weights(self, losses: Dict[str, float]):
        """
        Updates the model performance EMAs based on recent losses.
        """
        if not self.model_ids:
            print("[Hedge] No models to update weights for. Skipping.")
            return

        print(f"[Hedge] Updating weights with EMA alpha {self.learning_rate}...")

        for model_id, loss in losses.items():
            if model_id in self.performance_ema:
                # Convert loss [0, inf] to performance score [-1, 1]
                # Simple inversion: score = 1 - 2*loss (assuming loss is normalized 0-1)
                # A more robust approach might be to use percentile rank of PnL
                performance_score = 1 - (2 * min(loss, 1.0)) # Clip loss at 1.0
                
                # Update EMA
                self.performance_ema[model_id] = (performance_score * self.learning_rate) + \
                                                 (self.performance_ema[model_id] * (1 - self.learning_rate))
        
        print(f"[Hedge] New Weights: {self.get_weights()}")
        self._save_state()

    def add_model(self, model_id: str):
        if model_id in self.model_ids:
            return
        self.model_ids.append(model_id)
        self.performance_ema[model_id] = 0.5 # Initialize with neutral performance
        self._save_state()

    def remove_model(self, model_id: str):
        if model_id not in self.model_ids:
            return
        self.model_ids.remove(model_id)
        if model_id in self.performance_ema:
            del self.performance_ema[model_id]
        self._save_state()

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            joblib.dump(self.performance_ema, self.persistence_path)
        except Exception as e:
            print(f"[Hedge] ERROR: Could not save state: {e}")

    def _load_state(self) -> dict:
        if os.path.exists(self.persistence_path):
            try:
                loaded_ema = joblib.load(self.persistence_path)
                # Sync with current model_ids
                synced_ema = {mid: loaded_ema.get(mid, 0.5) for mid in self.model_ids}
                return synced_ema
            except Exception as e:
                print(f"[Hedge] ERROR: Could not load state. Re-initializing. Error: {e}")
        
        self._initialize_weights()
        return self.performance_ema
