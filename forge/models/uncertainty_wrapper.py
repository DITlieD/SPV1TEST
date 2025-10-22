# forge/models/uncertainty_wrapper.py
import numpy as np
from mapie.classification import MapieClassifier

class UncertaintyWrapper:
    """
    Wraps a model with a Conformal Prediction layer to quantify uncertainty.
    It uses the MAPIE library to generate prediction sets. The size of the
    set is a direct measure of the model's uncertainty.
    """
    def __init__(self, base_model, calibration_data):
        """
        Args:
            base_model: A trained, scikit-learn compatible classifier.
            calibration_data (tuple): A tuple of (X_cal, y_cal) for fitting MAPIE.
        """
        self.mapie_model = MapieClassifier(estimator=base_model, cv="prefit", method="cumulated_score")
        print("[Uncertainty] Initializing MAPIE wrapper...")
        
        X_cal, y_cal = calibration_data
        self.mapie_model.fit(X_cal, y_cal)
        print("[Uncertainty] MAPIE wrapper has been calibrated.")

    def get_uncertainty_score(self, X_new: np.ndarray, alpha=0.1) -> float:
        """
        Calculates an uncertainty score for a new observation.

        Args:
            X_new (np.ndarray): The new feature vector to predict on.
            alpha (float): The desired confidence level (e.g., 0.1 for 90% confidence).

        Returns:
            float: An uncertainty score. A score of 1 means one class is predicted
                   with high confidence. A score of 2 or 3 means the model is
                   uncertain and includes multiple classes in its prediction set.
        """
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        _, prediction_sets = self.mapie_model.predict(X_new, alpha=alpha)
        
        # The uncertainty score is the number of classes in the prediction set.
        # A smaller set means higher certainty.
        uncertainty_score = prediction_sets.sum(axis=1)[0]
        
        return float(uncertainty_score)
