import hdbscan
import numpy as np
import pandas as pd

class DynamicRegimeModel:
    """
    Uses HDBSCAN to dynamically identify market regimes from feature data.
    A key feature is its ability to identify 'noise' points that don't belong to any cluster,
    which is critical for avoiding unpredictable market conditions.
    """
    def __init__(self, min_cluster_size=50, min_samples=5):
        """
        Initializes the HDBSCAN clusterer.
        Args:
            min_cluster_size (int): The minimum size of a group to be considered a cluster.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples,
            prediction_data=True # Necessary for predicting new samples
        )
        self.is_fitted = False
        self.feature_columns = []

    def fit(self, features_df: pd.DataFrame):
        """
        Fits the HDBSCAN model to the historical feature data.
        Args:
            features_df (pd.DataFrame): A dataframe where each row is a timestep and each column is a feature.
        """
        if features_df.empty:
            print("[DynamicRegimeModel] ERROR: Cannot fit on an empty dataframe.")
            return
        
        self.feature_columns = features_df.columns.tolist()
        
        # HDBSCAN is sensitive to feature scaling, so we normalize.
        # Using a simple RobustScaler logic (quantile-based) to handle outliers.
        q1 = features_df.quantile(0.25)
        q3 = features_df.quantile(0.75)
        iqr = q3 - q1
        # Avoid division by zero for constant features
        iqr[iqr == 0] = 1.0
        scaled_features = (features_df - features_df.median()) / iqr
        
        print(f"[DynamicRegimeModel] Fitting on {len(scaled_features)} samples using {len(self.feature_columns)} features...")
        self.clusterer.fit(scaled_features)
        self.is_fitted = True
        print(f"[DynamicRegimeModel] Fit complete. Found {self.clusterer.labels_.max() + 1} distinct regimes.")

    def predict(self, features_df: pd.DataFrame):
        """
        Predicts the regime for new, unseen data points.
        Args:
            features_df (pd.DataFrame): The features for the new data points to classify.
        Returns:
            np.ndarray: An array of regime labels. Label -1 indicates a noise point.
        """
        if not self.is_fitted or features_df.empty:
            # Return -1 (noise) if the model isn't ready or there's no data
            return np.full(len(features_df), -1)

        # Ensure the prediction dataframe has the same columns as the training dataframe
        predict_df = features_df[self.feature_columns]

        # Apply the same scaling as in the fit method
        q1 = predict_df.quantile(0.25)
        q3 = predict_df.quantile(0.75)
        iqr = q3 - q1
        iqr[iqr == 0] = 1.0
        scaled_features = (predict_df - predict_df.median()) / iqr

        labels, _ = hdbscan.approximate_predict(self.clusterer, scaled_features)
        return labels