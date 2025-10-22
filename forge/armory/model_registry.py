# forge/armory/model_registry.py
import os
import json
import joblib
import datetime
import shutil
import time
from typing import Optional
from config import MODEL_DIR
import dill

class ModelRegistry:
    """
    The Armory. Provides a thread-safe, managed inventory of models.
    """
    def __init__(self, registry_path=None):
        self.registry_path = registry_path if registry_path is not None else MODEL_DIR
        self.manifest_path = os.path.join(self.registry_path, "model_manifest.json")
        self.lock_file_path = os.path.join(self.registry_path, ".manifest.lock")
        os.makedirs(self.registry_path, exist_ok=True)

    def _acquire_lock(self, timeout=10):
        start_time = time.time()
        while True:
            try:
                self.lock_fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return
            except FileExistsError:
                if time.time() - start_time >= timeout:
                    raise TimeoutError("Could not acquire lock on manifest file.")
                time.sleep(0.1)

    def _release_lock(self):
        try:
            if hasattr(self, 'lock_fd') and self.lock_fd is not None:
                os.close(self.lock_fd)
                os.remove(self.lock_file_path)
                self.lock_fd = None
        except (OSError, AttributeError):
            pass

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return {"models": {}}
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"models": {}}

    def _save_manifest(self, manifest):
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)

    def get_all_model_paths_for_asset(self, asset_symbol: str) -> list:
        """Gets all model directories for a given asset."""
        paths = []
        manifest = self._load_manifest()
        for model_id, model_info in manifest.get("models", {}).items():
            if model_info.get("asset_symbol") == asset_symbol:
                paths.append(model_info["path"])
        return paths

    def load_model(self, model_id: str):
        """Loads a single model artifact from the registry."""
        manifest = self._load_manifest()
        model_info = manifest.get("models", {}).get(model_id)
        if not model_info:
            return None

        model_dir = model_info["path"]
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata.get("blueprint", {}).get("model_type")

        try:
            if model_type == "GP 2.0":
                gp_components_path = os.path.join(model_dir, "strategy_components.dill")
                if os.path.exists(gp_components_path):
                    from deap import base, creator, gp as deap_gp
                    if not hasattr(creator, "FitnessMax"):
                        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                    if not hasattr(creator, "Individual"):
                        creator.create("Individual", deap_gp.PrimitiveTree, fitness=creator.FitnessMax)
                    
                    with open(gp_components_path, 'rb') as f:
                        gp_components = dill.load(f)
                    
                    from validation_gauntlet import EvolvedStrategyWrapper
                    return EvolvedStrategyWrapper(
                        tree=gp_components['tree'],
                        pset=gp_components['pset'],
                        feature_names=gp_components['feature_names']
                    )
            else:
                model_path = os.path.join(model_dir, "model.joblib")
                if os.path.exists(model_path):
                    return joblib.load(model_path)
        except Exception as e:
            print(f"[ModelRegistry] FATAL: Could not load model {model_id}: {e}")
            return None
        return None

    def register_model(self, model_artifact, model_blueprint: dict, validation_metrics: dict, xai_report_path: Optional[str], asset_symbol: str, model_instance_id: Optional[str] = None):
        model_type = model_blueprint.get("model_type", "Unknown")
        
        if model_instance_id:
            # Use the provided instance ID, ensuring it's clean
            model_id = model_instance_id.replace('/', '').replace(':', '').replace('\\', '')
        else:
            # Fallback to the old method if no instance ID is given
            clean_symbol = asset_symbol.replace('/', '').replace(':', '').replace('\\', '')
            clean_type = model_type.replace(' ', '_').replace(':', '').replace('.', '_')
            model_id = f"{clean_symbol}_{clean_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_dir = os.path.join(self.registry_path, model_id)

        print(f"[ModelRegistry] Attempting to register {model_type} for {asset_symbol}")
        print(f"[ModelRegistry] Model ID: {model_id}")
        print(f"[ModelRegistry] Model dir: {model_dir}")

        try:
            os.makedirs(model_dir, exist_ok=True)

            if model_type == 'GP 2.0':
                strategy_components = {
                    'tree': model_artifact.tree,
                    'pset': model_artifact.pset,
                    'feature_names': model_artifact.feature_names
                }
                with open(os.path.join(model_dir, "strategy_components.dill"), 'wb') as f:
                    dill.dump(strategy_components, f)
            else:
                joblib.dump(model_artifact, os.path.join(model_dir, "model.joblib"))

            metadata = {
                "model_id": model_id,
                "asset_symbol": asset_symbol,
                "registration_time": datetime.datetime.now().isoformat(),
                "status": "pending_review",
                "blueprint": model_blueprint,
                "validation_metrics": validation_metrics,
            }
            
            with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=4)

            self._acquire_lock()
            manifest = self._load_manifest()
            manifest["models"][model_id] = {
                "path": model_dir, 
                "status": "pending_review", 
                "asset_symbol": asset_symbol,
                "registration_time": metadata["registration_time"],
                "validation_metrics": validation_metrics,
                "explanation": model_blueprint.get("explanation", "N/A")
            }
            self._save_manifest(manifest)
            self._release_lock()
            
            print(f"SUCCESS: Registered new model {model_id}")
            return model_id
            
        except Exception as e:
            print(f"FATAL: Could not register model {model_id}: {e}")
            if os.path.exists(model_dir): shutil.rmtree(model_dir)
            self._release_lock()
            return None