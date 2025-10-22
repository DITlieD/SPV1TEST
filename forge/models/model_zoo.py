import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from forge.models.torch_utils import DEVICE
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class BaseModel:
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs

    def train(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

class LightGBMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        default_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        # Merge defaults with any provided kwargs
        final_params = {**default_params, **self.params}
        self.model = lgb.LGBMClassifier(**final_params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)

# PyTorch Denoising Autoencoder Network
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16, num_classes=3):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Assuming input features are scaled to [0, 1] or similar
        )
        self.classifier = nn.Linear(encoding_dim, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification_output = self.classifier(encoded)
        return decoded, classification_output


class DenoisingAutoencoderModel(BaseModel):

    def __init__(self, input_dim=None, encoding_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.encoder = None
        self.decoder = None

    def _build_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be specified for DAE.")

        # Define the PyTorch model architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, self.encoding_dim)
        ).to(DEVICE)

        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, self.input_dim)
            # No activation on output layer when using StandardScaler
        ).to(DEVICE)

    def train(self, X: pd.DataFrame, y=None, epochs=50, batch_size=32, noise_factor=0.1):
        # y is ignored for unsupervised DAE training
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        if self.encoder is None:
            self._build_model()

        print(f"[DAE] Training Autoencoder on {X.shape[1]} features...")
        X_scaled = self.scaler.fit_transform(X)

        # Prepare data
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

        self.encoder.train()
        self.decoder.train()

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for data in dataloader:
                inputs = data[0]
                # Add noise for denoising capability
                noisy_inputs = inputs + noise_factor * torch.randn_like(inputs).to(DEVICE)

                optimizer.zero_grad()
                encoded = self.encoder(noisy_inputs)
                decoded = self.decoder(encoded)
                loss = criterion(decoded, inputs)  # Compare reconstruction to clean input
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Optional: Print loss every 10 epochs
            # if (epoch+1) % 10 == 0:
            #     print(f"[DAE] Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Uses the trained encoder to extract latent features."""
        if self.encoder is None:
            return pd.DataFrame(index=X.index)

        X_scaled = self.scaler.transform(X)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

        self.encoder.eval()
        with torch.no_grad():
            encoded_features = self.encoder(tensor_x).cpu().numpy()

        # Create new feature names
        latent_cols = [f'latent_{i}' for i in range(self.encoding_dim)]
        return pd.DataFrame(encoded_features, columns=latent_cols, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # This model is a feature extractor in this pipeline.
        raise NotImplementedError("DAE is used via transform() in the pipeline, not predict().")


class TransformerEncoderModel(nn.Module):
    """Helper class: The PyTorch Transformer Architecture."""
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder
        # Note: The constraint d_model % nhead == 0 must be handled by the optimizer/caller
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # Input shape: (batch_size, seq_len, input_dim)
        # Apply scaling factor after projection
        src = self.input_proj(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32).to(src.device))
        
        output = self.transformer_encoder(src)
        
        # Classification: Use the representation of the last time step
        output = self.output_layer(output[:, -1, :])
        return output

class InformerTransformerModel(BaseModel):
    def __init__(self, input_dim=None, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.model = None
        # Define parameters used by the Transformer (defaults if not provided by optimizer)
        self.params = {
            'seq_len': self.params.get('seq_len', 30),
            'd_model': self.params.get('d_model', 64),
            'nhead': self.params.get('nhead', 4),
            'num_layers': self.params.get('num_layers', 2),
            'epochs': 10 # Keep low for optimization runs
        }

    def _prepare_sequences(self, X_scaled, y=None):
        """Converts scaled array into 3D sequences (samples, seq_len, features)."""
        sequences = []
        labels = []
        seq_len = self.params['seq_len']
        for i in range(len(X_scaled) - seq_len + 1):
            sequences.append(X_scaled[i:i + seq_len])
            if y is not None:
                 # Label corresponding to the end of the sequence
                 labels.append(y.iloc[i + seq_len - 1])

        X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(DEVICE)
        if y is not None:
            y_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)
            return X_tensor, y_tensor
        return X_tensor

    def train(self, X: pd.DataFrame, y: pd.Series):
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        # Initialize the model
        self.model = TransformerEncoderModel(
            self.input_dim, self.num_classes, self.params['d_model'], 
            self.params['nhead'], self.params['num_layers']
        ).to(DEVICE)

        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences and labels
        X_seq, y_tensor = self._prepare_sequences(X_scaled, y)

        if len(X_seq) == 0:
             print("[Transformer] Not enough data to create sequences. Skipping training.")
             return

        dataset = TensorDataset(X_seq, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(self.params['epochs']):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
             return pd.Series(0, index=X.index)

        self.model.eval()
        seq_len = self.params['seq_len']
        
        # Process the dataset for prediction (used during optimization/validation)
        X_scaled = self.scaler.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        
        if len(X_seq) == 0:
            # Return empty series if no sequences can be formed
            return pd.Series(dtype=int)
        
        with torch.no_grad():
            outputs = self.model(X_seq)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        # Align predictions with the correct index (the end of the sequences)
        prediction_index = X.index[seq_len-1:]
        return pd.Series(predictions, index=prediction_index)

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive

class BayesianNN(PyroModule):
    """A Bayesian Neural Network for classification."""
    def __init__(self, in_features, out_features, hidden_features=64):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, hidden_features)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_features, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_features]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](hidden_features, out_features)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, hidden_features]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))
        
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits.to(DEVICE)), obs=y)
        return logits

class BayesianNNModel(BaseModel):
    def __init__(self, input_dim=None, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.model = None
        self.guide = None
        self.svi = None
        self.params = {
            'hidden_features': self.params.get('hidden_features', 64),
            'epochs': self.params.get('epochs', 50),
            'lr': self.params.get('lr', 0.01)
        }

    def train(self, X: pd.DataFrame, y: pd.Series):
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.model = BayesianNN(self.input_dim, self.num_classes, self.params['hidden_features']).to(DEVICE)
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model)
        
        optimizer = pyro.optim.Adam({"lr": self.params['lr']})
        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

    def predict(self, X: pd.DataFrame, num_samples=100) -> tuple[pd.Series, pd.Series]:
        if self.model is None:
            return pd.Series(0, index=X.index), pd.Series(1.0, index=X.index)

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        samples = predictive(X_tensor)
        
        # Get predictions
        pred_samples = samples['obs'].cpu().numpy()
        # Majority vote for the final prediction
        preds = pd.Series([np.bincount(p).argmax() for p in pred_samples.T], index=X.index)
        
        # Get uncertainty (entropy of the mean predictive distribution)
        probs = torch.nn.functional.softmax(samples['obs'].float(), dim=-1).mean(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).cpu().numpy()
        uncertainty = pd.Series(entropy, index=X.index)
        
        return preds, uncertainty

from forge.models.transformer_wrapper import TransformerWrapper

# A dictionary to easily retrieve model classes by name
MODEL_ZOO = {
    "LightGBM": LightGBMModel,
    "InformerTransformer": InformerTransformerModel,
    "DenoisingAutoencoder": DenoisingAutoencoderModel,
    "BayesianNN": BayesianNNModel,
    "Transformer": TransformerWrapper,
}