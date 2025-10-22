# In forge/blueprint_factory/genetic_algorithm.py

import random
import numpy as np
import pandas as pd
from collections import deque
from multiprocessing import Pool, cpu_count
import logging
import json
import os
import re
from models_v2 import LGBMWrapper, XGBWrapper
from forge.models.transformer_wrapper import TransformerWrapper
from forge.crucible.fitness_function import run_fitness_evaluation

# --- DYNAMIC FEATURE LOADING SECTION ---
def load_gp_feature_names_from_foundry():
    filepath = 'forge/data_processing/gp_feature_foundry.json'
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f: formulas = json.load(f)
        names = ["gp_" + re.sub(r'[^a-zA-Z0-9_]', '', f).lower()[:50] for f in formulas]
        print(f"Loaded {len(names)} custom features from GP Foundry for the Genetic Algorithm.")
        return names
    except Exception as e:
        print(f"Error loading GP features from {filepath}: {e}")
        return []

# --- NEW: Function to load evolved features ---
# This is a placeholder. Ideally, read from where Feature Synthesizer saves them.
# Assuming a max of 10 evolved features named evolved_feat_1, evolved_feat_2, etc.
def load_evolved_feature_names(max_features=10):
    names = [f"evolved_feat_{i+1}" for i in range(max_features)]
    print(f"Placeholder: Added {len(names)} potential evolved features to GA.")
    return names
# --- END NEW ---


MODEL_ARCHITECTURES = ["LGBMWrapper", "XGBWrapper", "TransformerWrapper"]

# --- MODIFICATION TO FEATURE_SUBSETS ---
# We must use the *prefixed* feature names that data_processing_v2.py creates.
# AND include the evolved features.
FEATURE_SUBSETS = {
    "tactical": [
        "tactical_SMA_20", "tactical_EMA_20", "tactical_RSI_14", "tactical_MACD_12_26_9",
        "tactical_BBL_20_2.0", "tactical_BBM_20_2.0", "tactical_BBU_20_2.0", "tactical_ATRr_14",
        "tactical_ADX_14", "tactical_volatility_20", "tactical_momentum_20", "tactical_skewness_20"
    ],
    "strategic": [
        "strategic_SMA_20", "strategic_EMA_20", "strategic_RSI_14", "strategic_MACD_12_26_9",
        "strategic_BBL_20_2.0", "strategic_BBM_20_2.0", "strategic_BBU_20_2.0", "strategic_ATRr_14",
        "strategic_ADX_14", "strategic_volatility_20", "strategic_momentum_20", "strategic_skewness_20"
    ],
    "microstructure": ["tfi"], # Adjusted name based on data_processing_v2.py merge
    "gp_foundry": load_gp_feature_names_from_foundry(),
    # --- NEW: Add evolved features ---
    "evolved": load_evolved_feature_names()
    # --- END NEW ---
}
# --- END OF FIX ---

# Clean up any empty categories to prevent errors
FEATURE_SUBSETS = {k: v for k, v in FEATURE_SUBSETS.items() if v}

HYPERPARAMETER_RANGES = {
    "LGBMWrapper": {"n_estimators": (50, 500), "num_leaves": (20, 100)},
    "XGBWrapper": {"n_estimators": (50, 500), "max_depth": (3, 8)},
    "TransformerWrapper": {
        "d_model": (16, 64),
        "nhead": (2, 8),
        "num_layers": (1, 4),
        "epochs": (5, 20),
        "sequence_length": (10, 30)
    }
    # Add BayesianNN ranges if needed
}
RISK_PARAMETER_RANGES = {
    "stop_loss_multiplier": (1.0, 5.0),
    "take_profit_multiplier": (1.0, 8.0)
}

class ModelBlueprint:
    def __init__(self, architecture, features, hyperparameters, training_horizon, risk_parameters):
        self.architecture = architecture
        self.features = frozenset(features) # Use frozenset for consistency
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.training_horizon = training_horizon
        self.risk_parameters = risk_parameters if risk_parameters is not None else {}
        self.fitness = -1.0 # Initialize fitness
        self.shared_fitness = -1.0

    def __repr__(self):
        sl_mult = self.risk_parameters.get('stop_loss_multiplier', 'N/A')
        tp_mult = self.risk_parameters.get('take_profit_multiplier', 'N/A')
        sl_str = f"{sl_mult:.2f}" if isinstance(sl_mult, float) else sl_mult
        tp_str = f"{tp_mult:.2f}" if isinstance(tp_mult, float) else tp_mult
        num_features = len(self.features)
        # Handle cases where fitness hasn't been evaluated yet
        fitness_str = f"{self.fitness:.4f}" if self.fitness != -1.0 else "Not Evaluated"
        return (f"Blueprint(Arch: {self.architecture}, "
                f"NumFeats: {num_features}, "
                f"Horizon: {self.training_horizon}, "
                f"SL: {sl_str}, TP: {tp_str}, "
                f"Fitness: {fitness_str})")


def create_random_blueprint(min_horizon, max_horizon):
    arch = random.choice(MODEL_ARCHITECTURES)
    
    # --- FIX: Ensure at least one feature group is selected ---
    if not FEATURE_SUBSETS:
         # No features available at all, return a dummy blueprint
         print("Warning: FEATURE_SUBSETS is empty in genetic_algorithm.py!")
         return ModelBlueprint(arch, frozenset(), {}, min_horizon, {})

    feats = []
    # Ensure at least one group is chosen, even if only one group exists
    k_groups = random.randint(1, max(1, len(FEATURE_SUBSETS)))
    chosen_groups = random.sample(list(FEATURE_SUBSETS.keys()), k=k_groups)

    for group in chosen_groups:
        if FEATURE_SUBSETS[group]: # Check if the group actually has features
             # --- FIX: Select a random subset of features from the chosen group ---
             max_features_in_group = len(FEATURE_SUBSETS[group])
             # Choose between 1 and all features in that group
             k_features = random.randint(1, max(1, max_features_in_group))
             feats.extend(random.sample(FEATURE_SUBSETS[group], k=k_features))
        else:
            print(f"Warning: Feature group '{group}' is empty.")

    # Remove duplicates
    unique_feats = list(set(feats))
    
    # Ensure at least one feature if possible
    if not unique_feats and any(FEATURE_SUBSETS.values()):
        # If somehow ended up with no features, pick one randomly from any non-empty group
        non_empty_groups = [g for g, f in FEATURE_SUBSETS.items() if f]
        if non_empty_groups:
            random_group = random.choice(non_empty_groups)
            unique_feats.append(random.choice(FEATURE_SUBSETS[random_group]))
            print("Warning: Random blueprint initially had no features, added one randomly.")
        else:
             print("Critical Warning: No features available in any FEATURE_SUBSET group!")


    hps = {}
    if arch in HYPERPARAMETER_RANGES:
         hps = {p: (random.randint(*r) if isinstance(r[0], int) else random.uniform(*r))
                for p, r in HYPERPARAMETER_RANGES[arch].items()}

    # --- Transformer Specific Hyperparameter Fix ---
    if arch == "TransformerWrapper":
        # Ensure nhead is positive and divides d_model
        if 'nhead' in hps and hps.get('d_model', 0) > 0:
            if hps['nhead'] <= 0:
                hps['nhead'] = 2 # Default to a safe value
            # Adjust d_model to be divisible by nhead
            hps['d_model'] = (hps['d_model'] // hps['nhead']) * hps['nhead']
            # Ensure d_model is not zero after adjustment
            if hps['d_model'] == 0:
                hps['d_model'] = hps['nhead'] # Smallest possible valid dim
        elif 'd_model' not in hps or hps.get('d_model', 0) <= 0:
             # Set default if missing or invalid
             hps['nhead'] = hps.get('nhead', 2) # Use existing or default
             if hps['nhead'] <=0 : hps['nhead'] = 2
             hps['d_model'] = 32 # Set a default d_model
             # Ensure divisibility
             hps['d_model'] = (hps['d_model'] // hps['nhead']) * hps['nhead']
             if hps['d_model'] == 0: hps['d_model'] = hps['nhead']


    risk_params = {p: random.uniform(*r) for p, r in RISK_PARAMETER_RANGES.items()}
    horizon = random.randint(min_horizon, max_horizon)

    return ModelBlueprint(arch, frozenset(unique_feats), hps, horizon, risk_params)


# --- MULTIPROCESSING OPTIMIZATION: Pool Initializer and Worker Data ---
# (Rest of the GeneticAlgorithm class remains the same as previously provided)
# ... (Keep the rest of the file identical to the version from 2 steps ago) ...

# Ensure these imports are present at the top
from multiprocessing import Pool, cpu_count
import multiprocessing
import os
import logging
from tqdm import tqdm # Ensure tqdm is imported for progress bars
# Ensure run_fitness_evaluation is correctly imported, e.g.:
from forge.crucible.fitness_function import run_fitness_evaluation

# Global dictionary to hold data in worker processes (only used when Pool is active)
worker_data = {}

def initialize_worker(X_train, y_train, X_val, y_val, returns_val):
    """Initializer function for the multiprocessing Pool to share data efficiently."""
    worker_data['X_train'] = X_train
    worker_data['y_train'] = y_train
    worker_data['X_val'] = X_val
    worker_data['y_val'] = y_val
    worker_data['returns_val'] = returns_val

def evaluate_fitness_parallel_pool(blueprint):
    """The task function executed by the Pool workers, accessing shared worker_data."""
    X_train = worker_data.get('X_train')
    y_train = worker_data.get('y_train')
    X_val = worker_data.get('X_val')
    y_val = worker_data.get('y_val')
    returns_val = worker_data.get('returns_val')

    if X_train is None or X_val is None: # Added check for X_val
        print(f"[Worker {os.getpid()}] Error: Shared data (X_train or X_val) is None.")
        return -1e9 # Return default error score

    try:
        metrics = run_fitness_evaluation(
            blueprint, X_train, y_train, X_val, y_val, returns_val
        )
        # Use log_wealth if available, otherwise a large negative number
        fitness = metrics.get("log_wealth", -1e9)
        # Ensure fitness is a finite number
        if not np.isfinite(fitness):
             # print(f"[Worker {os.getpid()}] Warning: Non-finite fitness encountered ({fitness}). Blueprint: {blueprint}")
             fitness = -1e9
        return fitness
    except Exception as e:
        # Log the specific error and blueprint details for better debugging
        print(f"[Worker {os.getpid()}] Error during evaluation for blueprint {blueprint}: {e}")
        # Optionally log the full traceback
        # import traceback
        # print(f"Traceback:\n{traceback.format_exc()}")
        return -1e9 # Return default error score

class GeneticAlgorithm:
    def __init__(self, X_train, y_train, X_val, y_val, returns_val, min_horizon, max_horizon,
                 population_size=40, generations=50, mutation_rate=0.2, crossover_rate=0.8, elitism_count=2,
                 n_islands=4, migration_interval=10, hall_of_fame_size=5,
                 environment="Undefined", champion_blueprint=None, num_processes=None, logger=None, device='cpu',
                 surrogate_manager=None, use_surrogate=False):

        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.returns_val = returns_val
        self.min_horizon, self.max_horizon = min_horizon, max_horizon
        self.population_size, self.generations = population_size, generations
        self.mutation_rate, self.crossover_rate = mutation_rate, crossover_rate
        self.elitism_count = elitism_count
        self.n_islands, self.migration_interval = n_islands, migration_interval
        self.hall_of_fame_size, self.hall_of_fame = hall_of_fame_size, []
        self.environment = environment
        self.champion_blueprint = champion_blueprint
        self.islands = []
        self.num_processes = num_processes if num_processes is not None else cpu_count()
        self.surrogate_manager = surrogate_manager
        self.use_surrogate = use_surrogate
        # Ensure model map is updated if new models are added (e.g., BayesianNN)
        self.model_map = {"LGBMWrapper": LGBMWrapper, "XGBWrapper": XGBWrapper, "TransformerWrapper": TransformerWrapper}
        # Add BayesianNN if it's used:
        # from your_bayesian_nn_module import BayesianNN # Assuming you have this
        # self.model_map["BayesianNN"] = BayesianNN

        self.logger = logger if logger else logging.getLogger(__name__)
        self.device = device # Note: device might not be used if models don't support it explicitly

    def _evaluate_blueprint_fitness(self, blueprint):
        # Avoid re-evaluating if fitness is already calculated
        # Use a small tolerance for floating point comparison if necessary
        if blueprint.fitness != -1.0: # Check against initial value
             return blueprint.fitness

        try:
             # Ensure data passed is valid before evaluation
             if self.X_train is None or self.X_val is None or self.y_train is None or self.y_val is None or self.returns_val is None:
                 self.logger.error("Missing data for fitness evaluation.")
                 return -1e9

             metrics = run_fitness_evaluation(
                 blueprint,
                 self.X_train, self.y_train,
                 self.X_val, self.y_val,
                 self.returns_val
             )

             # Use log_wealth, default to large negative if missing
             fitness = metrics.get("log_wealth", -1e9)
             # Ensure finite value
             if not np.isfinite(fitness):
                 # self.logger.warning(f"Non-finite fitness encountered ({fitness}) for blueprint: {blueprint}")
                 fitness = -1e9
             return fitness
        except Exception as e:
             self.logger.error(f"Error evaluating blueprint {blueprint}: {e}")
             # Optionally log traceback
             # import traceback
             # self.logger.error(f"Traceback:\n{traceback.format_exc()}")
             return -1e9


    def initialize_population(self):
        self.islands = [[] for _ in range(self.n_islands)]
        # Ensure population size is divisible by num islands, handle remainder
        base_island_pop_size = self.population_size // self.n_islands
        remainder = self.population_size % self.n_islands
        island_sizes = [base_island_pop_size + (1 if i < remainder else 0) for i in range(self.n_islands)]

        if self.champion_blueprint:
            # Add champion to the first island if valid
            try:
                # Recreate blueprint to ensure consistency, handle potential missing attrs
                champ_bp = ModelBlueprint(
                    architecture=self.champion_blueprint.get('architecture'),
                    features=frozenset(self.champion_blueprint.get('features', [])),
                    hyperparameters=self.champion_blueprint.get('hyperparameters', {}),
                    training_horizon=self.champion_blueprint.get('training_horizon'),
                    risk_parameters=self.champion_blueprint.get('risk_parameters', {})
                )
                self.islands[0].append(champ_bp)
                self.logger.info(f"Injected champion blueprint into island 0: {champ_bp}")
            except Exception as e:
                self.logger.error(f"Failed to inject champion blueprint: {e}. Starting purely random.")


        for i in range(self.n_islands):
            current_island_size = len(self.islands[i])
            needed = island_sizes[i] - current_island_size
            for _ in range(needed):
                # Ensure create_random_blueprint doesn't fail catastrophically
                try:
                    bp = create_random_blueprint(self.min_horizon, self.max_horizon)
                    if bp: # Check if blueprint creation was successful
                        self.islands[i].append(bp)
                    else:
                        self.logger.warning("create_random_blueprint returned None, skipping.")
                except Exception as e:
                     self.logger.error(f"Error creating random blueprint: {e}")
                     # Decide how to handle: skip, create dummy, etc.
                     # Skipping for now

        total_pop = sum(len(island) for island in self.islands)
        self.logger.info(f"Initialized population with {total_pop} individuals across {self.n_islands} islands.")


    def run(self):
        self.initialize_population()

        # Check for empty population edge case
        if not any(self.islands):
            self.logger.error("Population initialization failed. No individuals created. Aborting GA run.")
            return [] # Return empty hall of fame


        is_subprocess = multiprocessing.current_process().name != 'MainProcess'

        if is_subprocess or self.num_processes == 1:
            self.logger.info("[GA] Running in subprocess or num_processes=1. Using sequential execution.")
            self._run_evolution_sequential()
        else:
            self.logger.info(f"[GA] Running in main process. Using Pool with {self.num_processes} workers.")
            try:
                 self._run_evolution_parallel()
            except Exception as e:
                 self.logger.error(f"Error during parallel evolution: {e}. Falling back to sequential.")
                 # Fallback mechanism
                 # Consider if you need to reset population fitness values here
                 # for island in self.islands:
                 #     for bp in island:
                 #         bp.fitness = -1.0 # Reset fitness before sequential run
                 self._run_evolution_sequential()


        # Log final best blueprint from Hall of Fame
        if self.hall_of_fame:
             self.logger.info(f"GA Run Complete. Best blueprint in Hall of Fame: {self.hall_of_fame[0]}")
        else:
             self.logger.warning("GA Run Complete. Hall of Fame is empty.")

        return self.hall_of_fame

    def _run_evolution_sequential(self):
        """Runs the evolution loop sequentially."""
        for gen in range(self.generations):
            self.logger.info(f"--- Generation {gen+1}/{self.generations} (Sequential) ---")

            full_population = [ind for island in self.islands for ind in island]
            if not full_population:
                self.logger.warning(f"Generation {gen+1}: Population is empty. Stopping evolution.")
                break

            # Evaluate fitness sequentially
            evaluated_count = 0
            for bp in tqdm(full_population, desc=f"Gen {gen+1} Evaluating (Seq)"):
                try:
                    bp.fitness = self._evaluate_blueprint_fitness(bp)
                    evaluated_count +=1
                except Exception as e:
                     self.logger.error(f"Error evaluating blueprint {bp} sequentially: {e}")
                     bp.fitness = -1e9 # Assign error score


            self.logger.info(f"Gen {gen+1}: Evaluated {evaluated_count}/{len(full_population)} blueprints.")

            # Proceed only if some evaluations were successful potentially
            if evaluated_count > 0:
                 self._post_evaluation_steps(gen)
            else:
                 self.logger.error(f"Gen {gen+1}: No blueprints evaluated successfully. Stopping.")
                 break


    def _run_evolution_parallel(self):
        """Runs the evolution loop in parallel using a Pool with initializer."""
        initializer_args = (
            self.X_train, self.y_train, self.X_val, self.y_val, self.returns_val
        )

        # Use try-with-resources for Pool to ensure cleanup
        try:
            with Pool(
                processes=self.num_processes,
                initializer=initialize_worker,
                initargs=initializer_args
            ) as pool:
                for gen in range(self.generations):
                    self.logger.info(f"--- Generation {gen+1}/{self.generations} (Parallel) ---")

                    full_population = [ind for island in self.islands for ind in island]
                    if not full_population:
                        self.logger.warning(f"Generation {gen+1}: Population is empty. Stopping evolution.")
                        break

                    # Execute in parallel using imap for progress and memory efficiency
                    results = []
                    try:
                         # Wrap imap with tqdm for progress bar
                         result_iterator = pool.imap(evaluate_fitness_parallel_pool, full_population)
                         results = list(tqdm(result_iterator, total=len(full_population), desc=f"Gen {gen+1} Evaluating (Par)"))
                    except Exception as e:
                         self.logger.error(f"Error during pool.imap execution in Gen {gen+1}: {e}")
                         # Handle partial results? Or stop? Assign error score?
                         # For simplicity, assign error score to all for this gen and potentially stop
                         results = [-1e9] * len(full_population)


                    # Assign fitness back carefully, matching results to blueprints
                    if len(results) == len(full_population):
                        for i, bp in enumerate(full_population):
                            bp.fitness = results[i]
                        self.logger.info(f"Gen {gen+1}: Assigned fitness to {len(full_population)} blueprints.")
                        self._post_evaluation_steps(gen)
                    else:
                         self.logger.error(f"Gen {gen+1}: Mismatch between population size ({len(full_population)}) and results ({len(results)}). Stopping.")
                         break # Stop evolution if results are inconsistent

        except Exception as e:
            self.logger.error(f"Fatal error initializing or running the multiprocessing Pool: {e}")
            raise # Re-raise the exception to be caught by the fallback mechanism in run()


    def _post_evaluation_steps(self, gen):
        """Common steps after fitness evaluation (sharing, HoF update, evolution)."""
        all_pop_before_evolution = []
        for island_idx, island in enumerate(self.islands):
            if not island:
                self.logger.warning(f"Gen {gen+1}: Island {island_idx} is empty before applying fitness sharing.")
                continue # Skip empty islands

            try:
                self._apply_fitness_sharing(island)
                all_pop_before_evolution.extend(island)
            except Exception as e:
                 self.logger.error(f"Error applying fitness sharing to island {island_idx}: {e}")

        if not all_pop_before_evolution:
             self.logger.error(f"Gen {gen+1}: Entire population is empty after fitness sharing attempts. Stopping evolution.")
             return # Cannot proceed

        # Update Hall of Fame
        try:
            self._update_hall_of_fame(all_pop_before_evolution)
        except Exception as e:
             self.logger.error(f"Error updating Hall of Fame in Gen {gen+1}: {e}")


        # Log average/max fitness BEFORE evolution for the generation
        valid_fitnesses = [bp.fitness for bp in all_pop_before_evolution if np.isfinite(bp.fitness) and bp.fitness > -1e9]
        if valid_fitnesses:
            avg_fitness = np.mean(valid_fitnesses)
            max_fitness = np.max(valid_fitnesses)
            self.logger.info(f"Gen {gen+1} Stats (Pre-Evolution): Avg Fitness={avg_fitness:.4f}, Max Fitness={max_fitness:.4f}")
        else:
            self.logger.warning(f"Gen {gen+1}: No valid fitness scores found in population before evolution.")


        # Evolve each island
        next_generation_islands = [[] for _ in range(self.n_islands)]
        for i, island in enumerate(self.islands):
            if not island:
                 self.logger.warning(f"Gen {gen+1}: Skipping evolution for empty island {i}.")
                 continue # Skip empty island

            try:
                # Sort by shared_fitness for selection pressure
                island.sort(key=lambda x: x.shared_fitness, reverse=True)

                # Elitism: Carry over the best individuals
                num_elites = min(self.elitism_count, len(island))
                next_gen_island = island[:num_elites]

                # Selection and Reproduction for the rest
                num_to_reproduce = len(island) - num_elites
                if num_to_reproduce > 0:
                    # Select parents based on shared fitness
                    parents = self.selection(island) # Select enough parents

                    # Ensure we have pairs for crossover
                    num_parents_needed = (num_to_reproduce + 1) // 2 * 2 # Need pairs
                    if len(parents) < num_parents_needed:
                         # Handle case with insufficient parents (e.g., repeat selection or sample with replacement)
                         parents.extend(random.choices(island, k=num_parents_needed - len(parents)))


                    # Generate offspring through crossover and mutation
                    offspring_count = 0
                    parent_idx = 0
                    while offspring_count < num_to_reproduce and parent_idx < len(parents) -1:
                        p1 = parents[parent_idx]
                        p2 = parents[parent_idx + 1]
                        parent_idx += 2

                        c1, c2 = self.crossover(p1, p2)
                        
                        # Mutate offspring
                        mutated_c1 = self.mutate(c1)
                        next_gen_island.append(mutated_c1)
                        offspring_count += 1

                        if offspring_count < num_to_reproduce:
                            mutated_c2 = self.mutate(c2)
                            next_gen_island.append(mutated_c2)
                            offspring_count += 1

                # Replace old island population with the new generation
                next_generation_islands[i] = next_gen_island

            except Exception as e:
                self.logger.error(f"Error during evolution process for island {i} in Gen {gen+1}: {e}")
                # Keep the old island or an empty list? Keeping old might stall evolution.
                next_generation_islands[i] = island # Keep original island on error


        self.islands = next_generation_islands


        # --- Migration Step ---
        if self.n_islands > 1 and (gen + 1) % self.migration_interval == 0:
             self.logger.info(f"Gen {gen+1}: Performing migration...")
             try:
                 self._migrate()
             except Exception as e:
                 self.logger.error(f"Error during migration in Gen {gen+1}: {e}")


        # Log best in HoF again after evolution/migration
        best_hof = self.hall_of_fame[0] if self.hall_of_fame else None
        if best_hof:
             self.logger.info(f"Best in Hall of Fame (End of Gen {gen+1}): {best_hof}")


    def _migrate(self, migration_size=1):
         """Performs migration between islands."""
         if self.n_islands <= 1: return

         island_indices = list(range(self.n_islands))
         random.shuffle(island_indices) # Shuffle to randomize migration pairs/ring

         for i in range(self.n_islands):
             current_island_idx = island_indices[i]
             next_island_idx = island_indices[(i + 1) % self.n_islands] # Ring topology

             current_island = self.islands[current_island_idx]
             next_island = self.islands[next_island_idx]

             if not current_island: continue # Skip if source island is empty

             # Sort source island by fitness (or shared_fitness) to select migrants
             current_island.sort(key=lambda x: x.shared_fitness, reverse=True)

             num_to_migrate = min(migration_size, len(current_island))
             if num_to_migrate == 0: continue

             migrants = current_island[:num_to_migrate]
             # Remove migrants from source island (optional, prevents duplicates immediately)
             # self.islands[current_island_idx] = current_island[num_to_migrate:]

             # Add migrants to the destination island
             next_island.extend(migrants)

             # Optional: Trim destination island if it exceeds size limits
             # target_size = len(island) # Or use the calculated island_sizes from initialize
             # if len(next_island) > target_size:
             #     next_island.sort(key=lambda x: x.shared_fitness, reverse=True) # Keep best
             #     self.islands[next_island_idx] = next_island[:target_size]
             # else:
             self.islands[next_island_idx] = next_island # Update island list


         self.logger.info(f"Migration complete. Island sizes: {[len(island) for island in self.islands]}")


    def _jaccard_similarity(self, set1, set2):
        """Calculates Jaccard similarity between two sets (feature sets)."""
        # Ensure inputs are sets
        set1 = set(set1)
        set2 = set(set2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    def _apply_fitness_sharing(self, population, sharing_threshold=0.7, alpha=1.0):
        """Applies fitness sharing to a population based on feature similarity."""
        for i, ind1 in enumerate(population):
            niche_count = 0
            for j, ind2 in enumerate(population):
                # Calculate distance/similarity (using Jaccard on features)
                similarity = self._jaccard_similarity(ind1.features, ind2.features)
                # Apply sharing function (linear decay based on similarity)
                sh = max(0, 1 - (similarity / sharing_threshold)**alpha) if similarity < sharing_threshold else 0
                # Niche count is sum of sharing function values
                niche_count += (1-sh) # Corrected: niche_count sums the sharing values (closer = higher contribution)


            # Adjust fitness: divide by niche count (ensure niche_count > 0)
            ind1.shared_fitness = ind1.fitness / max(1.0, niche_count) # Avoid division by zero


    def selection(self, population, tournament_size=5):
        """Selects individuals using tournament selection based on shared fitness."""
        selected_parents = []
        population_size = len(population)
        if population_size == 0: return [] # Handle empty population

        # Ensure tournament size isn't larger than population
        actual_tournament_size = min(tournament_size, population_size)

        for _ in range(population_size): # Select as many parents as individuals needed
            # Sample participants for the tournament
            tournament_participants = random.sample(population, actual_tournament_size)
            # Choose the winner (highest shared_fitness)
            winner = max(tournament_participants, key=lambda x: x.shared_fitness)
            selected_parents.append(winner)
        return selected_parents

    def crossover(self, p1, p2):
        """Performs crossover between two parent blueprints."""
        # --- FIX: Create copies to avoid modifying parents directly ---
        c1 = ModelBlueprint(p1.architecture, p1.features, p1.hyperparameters.copy(), p1.training_horizon, p1.risk_parameters.copy())
        c2 = ModelBlueprint(p2.architecture, p2.features, p2.hyperparameters.copy(), p2.training_horizon, p2.risk_parameters.copy())


        if random.random() > self.crossover_rate:
             return c1, c2 # Return copies if no crossover happens


        # --- Crossover Logic ---
        # 1. Architecture Crossover (optional, simple swap)
        if random.random() < 0.5:
             c1.architecture, c2.architecture = c2.architecture, c1.architecture
             # Reset hyperparameters if architecture changes
             c1.hyperparameters = {p: (random.randint(*r) if isinstance(r[0], int) else random.uniform(*r))
                                   for p, r in HYPERPARAMETER_RANGES.get(c1.architecture, {}).items()}
             c2.hyperparameters = {p: (random.randint(*r) if isinstance(r[0], int) else random.uniform(*r))
                                   for p, r in HYPERPARAMETER_RANGES.get(c2.architecture, {}).items()}


        # 2. Training Horizon Crossover (average)
        avg_horizon = int((p1.training_horizon + p2.training_horizon) / 2)
        c1.training_horizon = avg_horizon
        c2.training_horizon = avg_horizon


        # 3. Feature Crossover (e.g., uniform crossover)
        all_features = list(p1.features.union(p2.features))
        child1_features = set()
        child2_features = set()
        for feature in all_features:
            if random.random() < 0.5:
                if feature in p1.features: child1_features.add(feature)
                if feature in p2.features: child2_features.add(feature)
            else:
                 if feature in p2.features: child1_features.add(feature)
                 if feature in p1.features: child2_features.add(feature)
        # Ensure children have at least one feature if possible
        if not child1_features and all_features: child1_features.add(random.choice(all_features))
        if not child2_features and all_features: child2_features.add(random.choice(all_features))
        c1.features = frozenset(child1_features)
        c2.features = frozenset(child2_features)


        # 4. Hyperparameter Crossover (e.g., blend or average for numeric)
        # Only crossover parameters relevant to the *current* architecture of the child
        valid_hp_c1 = HYPERPARAMETER_RANGES.get(c1.architecture, {})
        valid_hp_c2 = HYPERPARAMETER_RANGES.get(c2.architecture, {})

        # Blend common hyperparameters
        common_keys = set(p1.hyperparameters.keys()) & set(p2.hyperparameters.keys())
        for k in common_keys:
             if k in valid_hp_c1: # Apply to c1 only if valid for its arch
                 val1 = p1.hyperparameters[k]
                 val2 = p2.hyperparameters[k]
                 blend_alpha = random.random()
                 if isinstance(val1, int) and isinstance(val2, int):
                     c1.hyperparameters[k] = int(blend_alpha * val1 + (1 - blend_alpha) * val2)
                 elif isinstance(val1, float) or isinstance(val2, float):
                      c1.hyperparameters[k] = blend_alpha * float(val1) + (1 - blend_alpha) * float(val2)
                 # else: handle other types like strings? (simple swap maybe)

             if k in valid_hp_c2: # Apply to c2 only if valid for its arch
                  val1 = p1.hyperparameters[k]
                  val2 = p2.hyperparameters[k]
                  blend_alpha = random.random() # Recalculate alpha for C2
                  if isinstance(val1, int) and isinstance(val2, int):
                      c2.hyperparameters[k] = int(blend_alpha * val1 + (1 - blend_alpha) * val2)
                  elif isinstance(val1, float) or isinstance(val2, float):
                      c2.hyperparameters[k] = blend_alpha * float(val1) + (1 - blend_alpha) * float(val2)

        # 5. Risk Parameter Crossover (average)
        for k in RISK_PARAMETER_RANGES.keys():
             if k in p1.risk_parameters and k in p2.risk_parameters:
                 avg_risk = (p1.risk_parameters[k] + p2.risk_parameters[k]) / 2
                 c1.risk_parameters[k] = avg_risk
                 c2.risk_parameters[k] = avg_risk


        # Reset fitness for children
        c1.fitness = -1.0
        c1.shared_fitness = -1.0
        c2.fitness = -1.0
        c2.shared_fitness = -1.0


        return c1, c2

    def mutate(self, bp_orig):
        """Applies mutation to a blueprint."""
        # --- FIX: Create a copy to avoid modifying the original blueprint ---
        bp = ModelBlueprint(
             bp_orig.architecture,
             bp_orig.features, # Frozenset is immutable, ok to copy directly
             bp_orig.hyperparameters.copy(),
             bp_orig.training_horizon,
             bp_orig.risk_parameters.copy()
        )
        bp.fitness = -1.0 # Reset fitness after mutation
        bp.shared_fitness = -1.0

        mutated = False # Track if any mutation occurred

        # 1. Mutate Architecture
        if random.random() < self.mutation_rate:
            bp.architecture = random.choice(MODEL_ARCHITECTURES)
            # IMPORTANT: Reset hyperparameters when architecture changes
            bp.hyperparameters = {p: (random.randint(*r) if isinstance(r[0], int) else random.uniform(*r))
                                  for p, r in HYPERPARAMETER_RANGES.get(bp.architecture, {}).items()}
            # --- Transformer Specific Fix during mutation ---
            if bp.architecture == "TransformerWrapper":
                 if 'nhead' in bp.hyperparameters and bp.hyperparameters.get('d_model', 0) > 0:
                     if bp.hyperparameters['nhead'] <= 0: bp.hyperparameters['nhead'] = 2
                     bp.hyperparameters['d_model'] = (bp.hyperparameters['d_model'] // bp.hyperparameters['nhead']) * bp.hyperparameters['nhead']
                     if bp.hyperparameters['d_model'] == 0: bp.hyperparameters['d_model'] = bp.hyperparameters['nhead']
                 else: # Set defaults if missing/invalid
                      bp.hyperparameters['nhead'] = 2
                      bp.hyperparameters['d_model'] = 32
                      bp.hyperparameters['d_model'] = (bp.hyperparameters['d_model'] // bp.hyperparameters['nhead']) * bp.hyperparameters['nhead']
                      if bp.hyperparameters['d_model'] == 0: bp.hyperparameters['d_model'] = bp.hyperparameters['nhead']

            mutated = True


        # 2. Mutate Training Horizon
        if random.random() < self.mutation_rate:
            change = random.randint(-int(self.max_horizon * 0.1), int(self.max_horizon * 0.1))
            # Ensure horizon doesn't become zero or negative unless intended
            bp.training_horizon = np.clip(bp.training_horizon + change, self.min_horizon, self.max_horizon)
            mutated = True

        # 3. Mutate Risk Parameters
        if random.random() < self.mutation_rate:
            if RISK_PARAMETER_RANGES: # Check if dict is not empty
                 param_to_mutate = random.choice(list(RISK_PARAMETER_RANGES.keys()))
                 bp.risk_parameters[param_to_mutate] = random.uniform(*RISK_PARAMETER_RANGES[param_to_mutate])
                 mutated = True


        # 4. Mutate Hyperparameters (Architecture-Aware)
        if random.random() < self.mutation_rate:
            valid_params = HYPERPARAMETER_RANGES.get(bp.architecture, {})
            if valid_params: # Check if there are params to mutate for this arch
                param_to_mutate = random.choice(list(valid_params.keys()))
                r = valid_params[param_to_mutate]
                if isinstance(r[0], int):
                     # Ensure range is valid (min <= max) before randint
                     min_val, max_val = r
                     if min_val > max_val: min_val = max_val # Simple fix
                     bp.hyperparameters[param_to_mutate] = random.randint(min_val, max_val)
                else:
                     bp.hyperparameters[param_to_mutate] = random.uniform(*r)

                # --- Transformer Specific Fix after HP mutation ---
                if bp.architecture == "TransformerWrapper" and param_to_mutate in ['d_model', 'nhead']:
                     if 'nhead' in bp.hyperparameters and bp.hyperparameters.get('d_model', 0) > 0:
                         if bp.hyperparameters['nhead'] <= 0: bp.hyperparameters['nhead'] = 2
                         bp.hyperparameters['d_model'] = (bp.hyperparameters['d_model'] // bp.hyperparameters['nhead']) * bp.hyperparameters['nhead']
                         if bp.hyperparameters['d_model'] == 0: bp.hyperparameters['d_model'] = bp.hyperparameters['nhead']

                mutated = True


        # 5. Mutate Features (Add or Remove)
        if random.random() < self.mutation_rate:
             current_features = set(bp.features)
             all_possible_features = [f for group in FEATURE_SUBSETS.values() for f in group if f] # Flattened list of all defined features

             if all_possible_features: # Check if there are any features defined at all
                 action = random.choice(['add', 'remove'])

                 if action == 'add':
                     # Find features not currently in the blueprint
                     potential_additions = [f for f in all_possible_features if f not in current_features]
                     if potential_additions:
                         feature_to_add = random.choice(potential_additions)
                         current_features.add(feature_to_add)
                         mutated = True
                 elif action == 'remove':
                      if len(current_features) > 1: # Ensure at least one feature remains
                          feature_to_remove = random.choice(list(current_features))
                          current_features.remove(feature_to_remove)
                          mutated = True

                 bp.features = frozenset(current_features)


        # If mutation happened, fitness needs recalculation
        # if mutated: # This reset is now done at the start of mutate
        #     bp.fitness = -1.0
        #     bp.shared_fitness = -1.0

        return bp


    def _update_hall_of_fame(self, population):
        """Updates the Hall of Fame with the best individuals from the population."""
        if not population: return # Skip if population is empty

        # Consider only valid, evaluated individuals
        valid_individuals = [bp for bp in population if hasattr(bp, 'fitness') and np.isfinite(bp.fitness)]

        if not valid_individuals: return # Skip if no valid individuals

        # Sort valid individuals by fitness (descending)
        valid_individuals.sort(key=lambda x: x.fitness, reverse=True)

        # Merge with current Hall of Fame and keep the best unique ones
        combined = self.hall_of_fame + valid_individuals
        
        # --- FIX: Ensure uniqueness based on content, not just object ID ---
        # Represent each blueprint uniquely (e.g., by its attributes string or a hash)
        # Using string representation might be slow but simple for now
        unique_blueprints_dict = {}
        for bp in combined:
            # Create a unique key (careful with mutable types like lists/dicts in key)
            # A tuple of sorted features and sorted items of dicts might work
            try:
                unique_key = (
                     bp.architecture,
                     tuple(sorted(list(bp.features))),
                     tuple(sorted(bp.hyperparameters.items())),
                     bp.training_horizon,
                     tuple(sorted(bp.risk_parameters.items()))
                )
                # Keep the one with the best fitness if duplicates found
                if unique_key not in unique_blueprints_dict or bp.fitness > unique_blueprints_dict[unique_key].fitness:
                     unique_blueprints_dict[unique_key] = bp
            except Exception as e:
                 # Handle cases where creating the key fails (e.g., unhashable types)
                 # Fallback: Use a simpler key or just skip uniqueness check here
                 self.logger.warning(f"Could not create unique key for blueprint {bp}: {e}. Skipping uniqueness check for this one.")
                 # Simple fallback: use object id (less effective for duplicates)
                 if id(bp) not in unique_blueprints_dict: unique_blueprints_dict[id(bp)] = bp


        unique_population = list(unique_blueprints_dict.values())


        # Sort the unique combined population by fitness
        unique_population.sort(key=lambda x: x.fitness, reverse=True)

        # Update Hall of Fame with the top N unique individuals
        self.hall_of_fame = unique_population[:self.hall_of_fame_size]


# --- End of GeneticAlgorithm Class ---