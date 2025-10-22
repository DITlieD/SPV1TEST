# forge/evolution/symbiotic_crucible.py
import pandas as pd
import logging
import joblib
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

logger = logging.getLogger("SymbioticCrucible")

class ImplicitBrain:
    """
    The Teacher Model (GBM/DNN). Learns complex patterns from the combined CCA feature set.
    """
    def __init__(self, feature_list):
        # LightGBM Configuration (Tuned for high capacity)
        params = {
            'objective': 'multiclass',
            'num_class': 3, # Assuming 3 classes (Buy, Sell, Hold)
            'metric': 'multi_logloss',
            'n_estimators': 1500,
            'learning_rate': 0.01,
            'num_leaves': 127,    # Increased complexity
            'verbose': -1,
            'n_jobs': -1,         # Utilize all cores for training the teacher
            'random_state': 42,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
        }
        self.model = lgb.LGBMClassifier(**params)
        self.features = feature_list

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Training Implicit Brain (LightGBM)...")
        
        if X.empty or y.empty:
            logger.error("Training data is empty.")
            return False

        # Basic train-test split for evaluation (Shuffle=False for time series)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        try:
            self.model.fit(X_train, y_train, 
                           eval_set=[(X_val, y_val)], 
                           eval_metric='multi_logloss',
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            
            # Evaluate performance
            val_probs = self.model.predict_proba(X_val)
            # Ensure labels are correctly handled for log_loss
            val_loss = log_loss(y_val, val_probs, labels=self.model.classes_)
            logger.info(f"Implicit Brain training complete. Best Validation LogLoss: {val_loss:.4f}")
            return True
        except Exception as e:
            logger.error(f"Implicit Brain training failed: {e}", exc_info=True)
            return False

    def save_model(self, model_path):
        """Saves the trained model (including features list)."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save the whole object (including the model and features)
        joblib.dump(self, model_path)
        logger.info(f"Implicit Brain saved to {model_path}")