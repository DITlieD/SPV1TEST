import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm # Import tqdm for progress bars

# Enable PyTorch CPU parallelization
torch.set_num_threads(torch.get_num_threads())  # Use all available threads

class TransformerWrapper:
    """
    A wrapper for the TimeSeriesTransformer model to provide a scikit-learn-like
    interface, handling data preparation, training, and prediction.
    """
    def __init__(self, sequence_length=60, d_model=128, nhead=8, num_layers=4, dropout=0.1, lr=0.001, epochs=50, batch_size=64, num_classes=3):
        self.sequence_length = sequence_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_trained = False
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
        # Store the core architectural parameters
        self.params = {
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_layers,
            'dim_feedforward': d_model * 4,
            'dropout': dropout,
            'num_classes': num_classes
        }
        
        # Store all parameters for logging/registration
        self.model_params = self.params.copy()
        self.model_params.update({
            'sequence_length': self.sequence_length,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        })

    def _prepare_data(self, df: pd.DataFrame):
        """Prepares data for the transformer, creating sequences."""
        if 'label' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'label' column.")
            
        self.feature_names = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'label']]
        
        # Fit scaler on training data only, and transform
        if not self.is_trained:
            self.scaler.fit(df[self.feature_names].fillna(0))
        
        df_scaled = self.scaler.transform(df[self.feature_names].fillna(0))

        # --- Pad or Truncate features to match d_model ---
        current_dim = df_scaled.shape[1]
        target_dim = self.params['d_model']

        if current_dim > target_dim:
            df_scaled = df_scaled[:, :target_dim]
        elif current_dim < target_dim:
            padding = np.zeros((df_scaled.shape[0], target_dim - current_dim))
            df_scaled = np.concatenate([df_scaled, padding], axis=1)
        
        X, y = [], []
        for i in range(len(df_scaled) - self.sequence_length):
            X.append(df_scaled[i:i+self.sequence_length])
            y.append(df['label'].iloc[i + self.sequence_length - 1]) # Label corresponds to the last element of the sequence
            
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

    def get_dna(self):
        """Returns the DNA (Parameters and features) for inheritance."""
        return {
            'architecture': self.__class__.__name__,
            'params': self.model_params,
            'features': self.feature_names if hasattr(self, 'feature_names') else []
        }

    def fit(self, df_train, device='cpu', **kwargs):
        # --- BRUTE FORCE GPU ---
        if device.lower() in ['gpu', 'cuda']:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU training was requested, but CUDA is not available. Halting.")
            resolved_device = torch.device('cuda')
        else:
            resolved_device = torch.device('cpu')
        
        print(f"[TransformerWrapper] Forcing training on device: {resolved_device}")

        # --- Data Size Check ---
        if len(df_train) < self.sequence_length * 2:
            self.is_trained = False
            print(f"[TransformerWrapper] WARNING: Dataframe length ({len(df_train)}) is less than 2x sequence length ({self.sequence_length}). Skipping training.")
            return self

        X_train, y_train = self._prepare_data(df_train)

        if X_train.shape[0] == 0:
            self.is_trained = False
            print("[TransformerWrapper] WARNING: Not enough data to create sequences. Skipping.")
            return self

        # Move tensors to device immediately
        X_train = X_train.to(resolved_device)
        y_train = y_train.to(resolved_device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False  # Already on device, no need to pin
        )

        self.model = TimeSeriesTransformer(**self.params)
        self.model.to(resolved_device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        epoch_iterator = tqdm(range(self.epochs), desc="  -> Training Transformer")
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                # Data already on device, no transfer needed
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Update progress bar with loss
            epoch_iterator.set_postfix({'loss': epoch_loss / len(train_loader)})
        
        self.is_trained = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for a batch of data."""
        if not self.is_trained or len(df) < self.sequence_length:
            # Return a neutral probability distribution for the whole dataframe
            return np.full((len(df), self.params['num_classes']), 1/self.params['num_classes'])

        # --- DEFINITIVE FIX: Use the stored feature names from training ---
        if not all(f in df.columns for f in self.feature_names):
            missing = [f for f in self.feature_names if f not in df.columns]
            raise ValueError(f"Prediction dataframe is missing required columns: {missing}")

        X_test_scaled = self.scaler.transform(df[self.feature_names].fillna(0))

        # --- Pad or Truncate features to match d_model ---
        current_dim = X_test_scaled.shape[1]
        target_dim = self.params['d_model']

        if current_dim > target_dim:
            X_test_scaled = X_test_scaled[:, :target_dim]
        elif current_dim < target_dim:
            padding = np.zeros((X_test_scaled.shape[0], target_dim - current_dim))
            X_test_scaled = np.concatenate([X_test_scaled, padding], axis=1)

        # Build sequences
        X_test_seq = []
        for i in range(len(X_test_scaled) - self.sequence_length + 1):
            X_test_seq.append(X_test_scaled[i:i+self.sequence_length])

        if len(X_test_seq) == 0:
            # Not enough data for sequences
            return np.full((len(df), self.params['num_classes']), 1/self.params['num_classes'])

        X_test_tensor = torch.tensor(np.array(X_test_seq), dtype=torch.float32)

        # Get device from model
        device = next(self.model.parameters()).device
        X_test_tensor = X_test_tensor.to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        # Pad the beginning with neutral probabilities to match input length
        num_padding_rows = len(df) - len(probabilities)
        if num_padding_rows > 0:
            padding_probs = np.full((num_padding_rows, self.params['num_classes']), 1/self.params['num_classes'])
            probabilities = np.vstack([padding_probs, probabilities])

        return probabilities

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts the class label for a batch of data.
        Returns signal labels: 0 (hold), 1 (buy/entry), 2 (sell/exit)
        """
        probabilities = self.predict_proba(df)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_classes, **kwargs):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Linear(d_model, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        if src.shape[-1] != self.d_model:
            raise ValueError(f"Input feature dimension ({src.shape[-1]}) does not match model's d_model ({self.d_model})")

        embedded_src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        encoded_output = self.transformer_encoder(embedded_src)
        
        mean_output = encoded_output.mean(dim=1)
        
        return self.output_layer(mean_output)