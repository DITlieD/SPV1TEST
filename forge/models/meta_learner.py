# forge/models/meta_learner.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from copy import deepcopy
import logging

class MetaLearner(nn.Module):
    """
    A Meta-Learning model using an LSTM architecture.
    This model is not trained to be a master of one strategy, but to
    quickly adapt to new market regimes with very few examples.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, n_layers: int = 2, output_dim: int = 3):
        super(MetaLearner, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input x should be of shape (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            # Add sequence length of 1 if not present
            x = x.unsqueeze(1)
            
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output of the last time step
        out = self.fc(out[:, -1, :])
        return self.softmax(out)

class ReptileTrainer:
    """
    Implements the Reptile meta-learning algorithm to train a MetaLearner model.
    Reptile works by repeatedly training on a task and moving the initial weights
    towards the trained weights.
    """
    def __init__(self, meta_model: MetaLearner, device: str = 'cpu', meta_lr: float = 0.1, inner_lr: float = 0.01, inner_steps: int = 5, reporter=None):
        self.meta_model = meta_model.to(device)
        self.device = device
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=meta_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
        self.reporter = reporter

    def _prepare_tasks(self, df: pd.DataFrame, features: list, n_tasks: int, task_size: int):
        """Splits the training data into a set of 'tasks' for meta-learning."""
        tasks = []
        max_start = len(df) - task_size
        if max_start <= 0:
            self.logger.warning("Dataframe too small to create tasks.")
            return []
            
        for _ in range(n_tasks):
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + task_size
            task_df = df.iloc[start_idx:end_idx]
            
            X = torch.tensor(task_df[features].values, dtype=torch.float32).to(self.device)
            y = torch.tensor(task_df['label'].values, dtype=torch.long).to(self.device)
            tasks.append((X, y))
        return tasks

    def _inner_loop(self, X_task, y_task):
        """Performs the inner loop of Reptile: trains a copy of the model on a single task."""
        # Create a temporary copy of the model for this task
        task_model = deepcopy(self.meta_model)
        task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)

        task_model.train()
        for _ in range(self.inner_steps):
            task_optimizer.zero_grad()
            predictions = task_model(X_task)
            loss = self.loss_fn(predictions, y_task)
            loss.backward()
            task_optimizer.step()
        
        return task_model

    def train(self, df_train: pd.DataFrame, features: list, n_epochs: int = 100, n_tasks_per_epoch: int = 20, task_size: int = 100):
        """
        The main meta-training loop.
        """
        self.logger.info(f"Starting Reptile meta-training for {n_epochs} epochs...")
        self.meta_model.train()

        for epoch in range(n_epochs):
            # Store the initial weights of the meta-model
            initial_weights = deepcopy(self.meta_model.state_dict())

            # Update UI progress
            if self.reporter:
                self.reporter.set_status("Meta-Learner Training", f"Epoch {epoch+1}/{n_epochs} - Preparing tasks...")

            # Prepare a batch of tasks for this epoch
            tasks = self._prepare_tasks(df_train, features, n_tasks_per_epoch, task_size)
            if not tasks:
                self.logger.error("No tasks could be prepared. Aborting training.")
                return

            # This will hold the final weights from each task
            task_final_weights = []

            for X_task, y_task in tasks:
                # Train a model on this specific task
                task_model = self._inner_loop(X_task, y_task)
                task_final_weights.append(deepcopy(task_model.state_dict()))

            # --- The Reptile Update Rule ---
            # Update the meta-model's weights by moving them towards the average of the
            # weights learned by the task-specific models.
            
            self.meta_optimizer.zero_grad()
            
            # Set the meta-model's weights to be the average of the task-trained weights
            avg_weights = self._average_state_dicts(task_final_weights)
            
            # Manually calculate the gradient for the meta-update
            for meta_param, initial_param, avg_param in zip(self.meta_model.parameters(), initial_weights.values(), avg_weights.values()):
                if meta_param.grad is None:
                    meta_param.grad = torch.zeros_like(meta_param.data)
                # The "gradient" is the direction from the initial weights to the average of the task weights
                meta_param.grad.data.add_((initial_param.data - avg_param.data) / self.meta_lr)

            self.meta_optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{n_epochs}] Meta-update complete.")
        
        self.logger.info("Reptile meta-training finished.")
        return self.meta_model

    def _average_state_dicts(self, state_dicts):
        """Averages a list of PyTorch state_dicts."""
        avg_dict = deepcopy(state_dicts[0])
        for key in avg_dict.keys():
            if avg_dict[key].data.dtype.is_floating_point:
                avg_dict[key].data.fill_(0.)
                for state_dict in state_dicts:
                    avg_dict[key].data += state_dict[key].data
                avg_dict[key].data /= len(state_dicts)
        return avg_dict
