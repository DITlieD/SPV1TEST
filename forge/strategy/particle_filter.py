import numpy as np
import pandas as pd
from scipy.stats import norm

class MarketStateTracker:
    """
    Tracks the unobservable market state (e.g., momentum) using a Particle Filter.
    """
    def __init__(self, num_particles=100, state_dimension=1, process_noise=0.01, measurement_noise=0.1):
        self.num_particles = num_particles
        self.state_dimension = state_dimension
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles: state and weights
        self.particles = np.random.randn(num_particles, state_dimension)
        self.weights = np.ones(num_particles) / num_particles
        
        self.state_estimate = np.mean(self.particles, axis=0)
        # print("[Particle Filter] Initialized.")

    def predict(self):
        """
        Move particles according to the process model (a random walk).
        This represents the time evolution of the hidden market state.
        """
        self.particles += np.random.normal(0, self.process_noise, self.particles.shape)

    def update(self, measurement: float, measurement_uncertainty: float):
        """
        Update particle weights based on the new measurement and its uncertainty.
        
        Args:
            measurement (float): The new observation (e.g., market return).
            measurement_uncertainty (float): The uncertainty of the measurement,
                                             provided by the Bayesian model.
        """
        # Adapt the measurement noise based on the BNN's uncertainty
        adaptive_measurement_noise = self.measurement_noise + measurement_uncertainty
        
        # --- Stability Guard ---
        if not np.isfinite(adaptive_measurement_noise) or adaptive_measurement_noise <= 1e-9:
            # (Optional: Log this event if a logger is available)
            # print(f"Warning: Unstable adaptive_measurement_noise detected: {adaptive_measurement_noise}. Resetting.")
            adaptive_measurement_noise = 1e-9 # Reset to a safe, small value
        
        # Calculate the likelihood of the measurement for each particle
        likelihoods = norm.pdf(measurement, loc=self.particles.flatten(), scale=adaptive_measurement_noise)
        
        # Update weights
        self.weights *= likelihoods
        
        # Normalize weights
        self.weights += 1.e-300 # Avoid division by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample particles to avoid degeneracy. Particles with higher weights
        are more likely to be selected.
        """
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def step(self, measurement: float, measurement_uncertainty: float) -> np.ndarray:
        """
        Performs a full predict-update-resample step.
        
        Returns:
            np.ndarray: The new estimated hidden state.
        """
        self.predict()
        self.update(measurement, measurement_uncertainty)
        
        # Resample if the effective number of particles is too low
        if 1. / np.sum(self.weights**2) < self.num_particles / 2:
            self.resample()
            
        # Update the state estimate (weighted mean of particles)
        self.state_estimate = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        
        return self.state_estimate

    def get_state_uncertainty(self) -> float:
        """
        Calculates the uncertainty of the state estimate as the weighted
        standard deviation of the particles.
        """
        weighted_mean = self.state_estimate
        weighted_variance = np.sum(self.weights * (self.particles.flatten() - weighted_mean)**2)
        return np.sqrt(weighted_variance)
