import optuna
from forge.crucible.fitness_function import objective as fitness_objective
import numpy as np
import pandas as pd
import os
import logging

from forge.blueprint_factory.genetic_algorithm import ModelBlueprint, HYPERPARAMETER_RANGES

from models_v2 import LGBMWrapper, XGBWrapper
from forge.models.transformer_wrapper import TransformerWrapper

class BayesianOptimizer:
    """
    Uses Optuna to intelligently search the hyperparameter space, with support for device-specific studies.
    """
    def __init__(self, blueprint: ModelBlueprint, X_train, y_train, X_val, y_val, returns_val, device='cpu', n_trials=100, logger=None, generation=0):
        self.blueprint = blueprint
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.returns_val = returns_val
        self.n_trials = n_trials
        self.device = device
        self.study = None
        self.generation = generation
        self.model_map = {"LGBMWrapper": LGBMWrapper, "XGBWrapper": XGBWrapper, "TransformerWrapper": TransformerWrapper}
        self.logger = logger if logger else logging.getLogger(__name__)
        # FIX: Initialize the model once
        ModelClass = self.model_map.get(self.blueprint.architecture)
        if not ModelClass:
            raise ValueError(f"Invalid architecture specified in blueprint: {self.blueprint.architecture}")
        self.model = ModelClass()

    def _progress_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Logs progress after each trial."""
        self.logger.info(f"Trial {trial.number}/{self.n_trials} for {self.blueprint.architecture} on {self.device.upper()} complete.")
        self.logger.info(f"  -> Fitness: {trial.value}")
        if study.best_trial:
            self.logger.info(f"  -> Best Fitness So Far: {study.best_trial.value}")

    def optimize(self):
        self.logger.info(f"Starting Bayesian Optimization for {self.blueprint.architecture}...")
        
        # The objective function is now imported and requires a trial and the GA instance (or a mock)
        # We create a mock GA instance to pass the required data to the objective function
        class MockGA:
            def __init__(self, X_train, y_train, X_val, y_val, returns_val, blueprint):
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
                self.returns_val = returns_val
                self.base_blueprint = blueprint

            def create_blueprint_from_trial(self, trial):
                # For Bayesian optimization, we only tune hyperparameters, not architecture or features
                hyperparameters = {}
                arch = self.base_blueprint.architecture
                for p, r in HYPERPARAMETER_RANGES.get(arch, {}).items():
                    if isinstance(r[0], int):
                        hyperparameters[p] = trial.suggest_int(p, r[0], r[1])
                    else:
                        hyperparameters[p] = trial.suggest_float(p, r[0], r[1])
                
                # Keep other blueprint attributes constant
                return ModelBlueprint(
                    architecture=arch,
                    features=self.base_blueprint.features,
                    hyperparameters=hyperparameters,
                    training_horizon=self.base_blueprint.training_horizon,
                    risk_parameters=self.base_blueprint.risk_parameters
                )

        mock_ga_instance = MockGA(self.X_train, self.y_train, self.X_val, self.y_val, self.returns_val, self.blueprint)

        storage_name = f"optuna_studies_{self.blueprint.architecture}_{self.device}.db"
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_name))
        study = optuna.create_study(direction="maximize", storage=storage, study_name=f"{self.blueprint.architecture}_study", load_if_exists=True)
        
        study.optimize(lambda trial: fitness_objective(trial, mock_ga_instance), n_trials=self.n_trials, n_jobs=-1, callbacks=[self._progress_callback])

        # Filter out pruned trials and return the best ones
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            return []

        # Sort by the final value (log_wealth)
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        
        # Return a list of [hyperparameters, fitness_value]
        return [[t.params, t.value] for t in completed_trials]