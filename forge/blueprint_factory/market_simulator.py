# [cite_start]// preventing overfitting to the single historical timeline. [cite: 490, 492]
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class VAE(Model):
    """A Variational Autoencoder to learn and generate market data distributions."""
    def __init__(self, original_dim, latent_dim=10):
        super(VAE, self).__init__()
        self.original_dim = original_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_inputs = tf.keras.Input(shape=(original_dim,))
        x = layers.Dense(64, activation="relu")(encoder_inputs)
        x = layers.Dense(32, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

        # Decoder
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = layers.Dense(32, activation="relu")(latent_inputs)
        x = layers.Dense(64, activation="relu")(x)
        decoder_outputs = layers.Dense(original_dim, activation="sigmoid")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        reconstruction = self.decoder(z)
        return reconstruction

class MarketSimulator:
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.scaler = None
        self.vae = None
        print(f"Market simulator initialized with historical data. Shape: {self.historical_data.shape}")

    def train(self, epochs=50):
        from sklearn.preprocessing import MinMaxScaler
        print("[Simulator] Training VAE on historical data...")
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(self.historical_data)

        self.vae = VAE(original_dim=data_scaled.shape[1])
        self.vae.compile(optimizer='adam')
        self.vae.fit(data_scaled, data_scaled, epochs=epochs, batch_size=32, verbose=0)
        print("[Simulator] VAE training complete.")

    def generate(self, num_samples):
        if not self.vae:
            raise Exception("VAE model is not trained yet. Call train() first.")

        print(f"[Simulator] VAE is generating {num_samples} new market scenarios...")
        random_latent_vectors = tf.random.normal(shape=(num_samples, self.vae.latent_dim))
        generated_data_scaled = self.vae.decoder.predict(random_latent_vectors)

        # Inverse transform to original scale
        generated_data = self.scaler.inverse_transform(generated_data_scaled)
        return pd.DataFrame(generated_data, columns=self.historical_data.columns)

# ... (Then modify task_scheduler.py to use this simulator for training data)
