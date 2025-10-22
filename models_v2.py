import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import multiprocessing # Ensure this import is present
import numpy as np
from river import compose, preprocessing, tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from gpu_helper import get_lgbm_params_with_device

class BaseBatchModel:
    def __init__(self, **model_params):
        self.model = None # This will hold the CalibratedClassifierCV instance
        self.feature_names = None
        self.model_params = model_params
        self.is_trained = False

    def _select_features(self, df):
        return [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'label']]

    def _prepare_data(self, df):
        if 'label' not in df.columns: raise ValueError("DataFrame must contain 'label' column.")
        if self.feature_names is None:
            self.feature_names = self._select_features(df)
        X = df[self.feature_names].fillna(0)
        X = np.ascontiguousarray(X.astype(np.float32))
        y = df['label'].values
        return X, y

    def fit(self, df_train, df_val=None):
        raise NotImplementedError

    def predict(self, df):
        if not self.is_trained: return np.array([], dtype=int)
        X = df[self.feature_names].fillna(0)
        return self.model.predict(X)

    def predict_proba(self, df):
        if not self.is_trained: return np.empty((0, 3))
        X = df[self.feature_names].fillna(0)
        return self.model.predict_proba(X)

    def get_dna(self):
        """Returns the DNA (Parameters and features) for inheritance."""
        return {
            'architecture': self.__class__.__name__,
            'params': self.model_params,
            'features': self.feature_names if hasattr(self, 'feature_names') else []
        }

class LGBMWrapper(BaseBatchModel):
    def fit(self, df_train, df_val=None, device='cpu', generation=0):
        X_train, y_train = self._prepare_data(df_train)
        
        is_subprocess = multiprocessing.current_process().name != 'MainProcess'
        
        # Determine n_jobs for CalibrationCV AND the base model
        n_jobs_subprocess = 1 if is_subprocess else -1
        calibration_n_jobs = 1 if is_subprocess else None 
        
        default_params = {'objective': 'multiclass', 'num_class': 3, 'random_state': 42, 'verbose': -1}

        if is_subprocess:
            print("[LGBMWrapper] Detected subprocess execution (Forge). Forcing n_jobs=1.")
            default_params['n_jobs'] = 1
        else:
            default_params['n_jobs'] = -1
        
        if device.lower() in ['gpu', 'cuda']:
            print("[LGBMWrapper] Forcing training on GPU...")
            default_params['device'] = 'gpu'
        else:
            print("[LGBMWrapper] Training on CPU...")
            default_params['device'] = 'cpu'
            
        final_params = {**default_params, **self.model_params}
        base_model = lgb.LGBMClassifier(**final_params)
        
        if df_val is not None:
            X_val, y_val = self._prepare_data(df_val)
            base_model.fit(X_train, y_train)
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
            self.model.fit(X_val, y_val)
        else:
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            self.model.fit(X_train, y_train)
            
        self.is_trained = True
        return self

class XGBWrapper(BaseBatchModel):
    def fit(self, df_train, df_val=None, device='cpu', generation=0):
        X_train, y_train = self._prepare_data(df_train)
        
        is_subprocess = multiprocessing.current_process().name != 'MainProcess'
        
        calibration_n_jobs = 1 if is_subprocess else None 

        default_params = {'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss', 'random_state': 42}

        if is_subprocess:
            print("[XGBWrapper] Detected subprocess execution (Forge). Forcing n_jobs=1.")
            default_params['n_jobs'] = 1
        else:
            default_params['n_jobs'] = -1
        
        if self.model_params.get('device') == 'gpu':
            default_params['tree_method'] = 'hist'
            default_params['device'] = 'cuda'
            
        base_model = xgb.XGBClassifier(**{**default_params, **self.model_params})
        
        if df_val is not None:
            X_val, y_val = self._prepare_data(df_val)
            base_model.fit(X_train, y_train)
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
            self.model.fit(X_val, y_val)
        else:
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            self.model.fit(X_train, y_train)
            
        self.is_trained = True
        return self

class RiverOnlineModel:
    """
    A wrapper for the River online learning library, specifically using a
    Hoeffding Adaptive Tree Classifier. This model learns from a stream of data,
    one instance at a time.
    """
    def __init__(self):
        # Define the online learning pipeline
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=100,
                delta=1e-5,
                leaf_prediction='nb',
                nb_threshold=10,
                seed=42
            )
        )
        self.feature_names = None
        self.is_trained = False

    def _select_features(self, df):
        """Selects features, excluding non-feature columns."""
        return [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'label']]

    def fit(self, df_train: pd.DataFrame):
        """
        Pre-trains the online model by iterating through a batch of historical data.
        This is used to warm up the model before live deployment.
        """
        if 'label' not in df_train.columns:
            raise ValueError("DataFrame must contain 'label' column for training.")
            
        if self.feature_names is None:
            self.feature_names = self._select_features(df_train)
            
        X = df_train[self.feature_names].fillna(0)
        y = df_train['label']

        # Iterate through the data to train the model instance by instance
        for i in range(len(X)):
            x_instance = X.iloc[i].to_dict()
            y_instance = y.iloc[i]
            self.model.learn_one(x_instance, y_instance)
            
        self.is_trained = True
        return self

    def learn_one(self, features: dict, label: int):
        """Updates the model with a single new instance of data."""
        if self.is_trained:
            self.model.learn_one(features, label)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts labels for a batch of data."""
        if not self.is_trained:
            return np.zeros(len(df))
            
        X = df[self.feature_names].fillna(0)
        predictions = []
        for i in range(len(X)):
            x_instance = X.iloc[i].to_dict()
            pred = self.model.predict_one(x_instance)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for a batch of data."""
        if not self.is_trained:
            return np.full((len(df), 3), 1/3)

        X = df[self.feature_names].fillna(0)
        probabilities = []
        for i in range(len(X)):
            x_instance = X.iloc[i].to_dict()
            proba_dict = self.model.predict_proba_one(x_instance)
            # Ensure consistent 3-class output
            probs = [proba_dict.get(0, 0.0), proba_dict.get(1, 0.0), proba_dict.get(2, 0.0)]
            probabilities.append(probs)
        return np.array(probabilities)