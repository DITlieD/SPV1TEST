import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    """A simple Autoencoder for anomaly detection."""
    def __init__(self, n_features, encoding_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_features),
            nn.Sigmoid() # Use Sigmoid because we scale data to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ImmuneSystem:
    """
    Monitors system and market data for anomalies using an Autoencoder.
    An abnormally high reconstruction error suggests an anomaly.
    """
    def __init__(self, n_features, device='cpu'):
        self.device = torch.device(device)
        self.model = Autoencoder(n_features=n_features).to(self.device)
        self.scaler = MinMaxScaler()
        self.loss_function = nn.MSELoss(reduction='none') # Use 'none' to get per-instance errors
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.is_trained = False
        self.reconstruction_error_threshold = None

    def fit(self, df_normal: pd.DataFrame, epochs=20, batch_size=32):
        """
        Trains the Autoencoder on a dataframe of 'normal' data in batches.
        """
        print("[ImmuneSystem] Training Autoencoder on normal data...")
        df_numeric = df_normal.select_dtypes(include=np.number).dropna()
        
        scaled_data = self.scaler.fit_transform(df_numeric)
        dataset = TensorDataset(torch.tensor(scaled_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for data in loader:
                inputs = data[0].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, inputs).mean() # Calculate mean loss for the batch
                loss.backward()
                self.optimizer.step()
        
        # Set the anomaly threshold using the trained model
        self._set_anomaly_threshold(dataset.tensors[0])
        
        self.is_trained = True
        print(f"[ImmuneSystem] Training complete. Anomaly threshold set to: {self.reconstruction_error_threshold:.6f}")

    def _set_anomaly_threshold(self, data_tensor):
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(data_tensor.to(self.device))
            errors = self.loss_function(reconstructions, data_tensor.to(self.device)).mean(axis=1)
            
            mean_error = torch.mean(errors)
            std_error = torch.std(errors)
            # Set threshold to 3 standard deviations above the mean reconstruction error
            self.reconstruction_error_threshold = (mean_error + 3 * std_error).item()

    def is_anomalous(self, df_live: pd.DataFrame) -> bool:
        """
        Checks if the live data point is anomalous.
        """
        if not self.is_trained or df_live.empty:
            return False

        df_numeric = df_live.select_dtypes(include=np.number).dropna()
        if df_numeric.empty:
            return False
            
        scaled_data = self.scaler.transform(df_numeric)
        dataset = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(dataset)
            errors = self.loss_function(reconstructions, dataset).mean(axis=1)
            
            if torch.any(errors > self.reconstruction_error_threshold):
                max_error = torch.max(errors).item()
                print(f"[ImmuneSystem] ANOMALY DETECTED! Max reconstruction error: {max_error:.6f}")
                return True
        
        return False