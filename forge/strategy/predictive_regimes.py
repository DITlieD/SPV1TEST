# forge/strategy/predictive_regimes.py

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)

class PredictiveRegimeModel:
    def __init__(self, n_components=4, n_iter=100, tol=1e-2):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=self.n_iter, tol=self.tol)
        self.scaler = StandardScaler()
        self.feature_columns = []

    def fit(self, features_df: pd.DataFrame):
        """Fits the HMM model to the provided features."""
        if features_df.empty:
            logger.error("[PredictiveRegimeModel] ERROR: Cannot fit on an empty dataframe.")
            return

        self.feature_columns = features_df.columns
        scaled_features = self.scaler.fit_transform(features_df)

        logger.info(f"[PredictiveRegimeModel] Fitting on {len(scaled_features)} samples using {len(self.feature_columns)} features...")
        self.model.fit(scaled_features)
        logger.info(f"[PredictiveRegimeModel] Fit complete. Model converged: {self.model.monitor_.converged}")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predicts the most likely regime for the given features."""
        if features_df.empty or len(self.feature_columns) == 0:
            return np.array([])

        scaled_features = self.scaler.transform(features_df[self.feature_columns])
        return self.model.predict(scaled_features)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predicts the probability of each regime for the given features."""
        if features_df.empty or len(self.feature_columns) == 0:
            return np.array([])

        scaled_features = self.scaler.transform(features_df[self.feature_columns])
        return self.model.predict_proba(scaled_features)

    def predict_next_regime_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predicts the probability of the next regime."""
        if features_df.empty or len(self.feature_columns) == 0:
            return np.array([])

        # Get the probability of the current state
        current_regime_probas = self.predict_proba(features_df)
        
        # Get the transition matrix
        transition_matrix = self.model.transmat_

        # Multiply the current state probabilities by the transition matrix
        # to get the probabilities of the next state
        next_regime_probas = np.dot(current_regime_probas, transition_matrix)
        
        return next_regime_probas

    def save(self, filepath: str):
        """Saves the model to a file."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        """Loads a model from a file."""
        return joblib.load(filepath)