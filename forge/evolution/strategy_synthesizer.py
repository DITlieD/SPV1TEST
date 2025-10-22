from deap import base, creator, tools, gp, algorithms # Ensure algorithms is imported
import random
import operator
import numpy as np
import pandas as pd
import multiprocessing
from loky import ProcessPoolExecutor as LokyPoolExecutor, cpu_count as loky_cpu_count
import os
import functools # Import for partial
import logging
import warnings
from sklearn.metrics import log_loss
from forge.evolution.surrogate_manager import SurrogateManager
from forge.analysis.complexity_manager import StrategyComplexityEstimator
from forge.evolution.symbiotic_crucible import ImplicitBrain
import torch
import config
import joblib
from forge.evolution.gp_primitives import setup_gp_primitives

# Suppress divide by zero warnings in GP operations (handled safely)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')

# ==============================================================================
# LOKY Worker State and Functions (for parallel GP evaluation)
# ==============================================================================
_worker_state = {}

def _init_worker(fitness_evaluator_func, pset, config, soft_labels_df):
    """Initializer function for loky workers."""
    os.environ['OMP_NUM_THREADS'] = '1'
    _worker_state['fitness_evaluator'] = fitness_evaluator_func
    _worker_state['pset'] = pset
    _worker_state['config'] = config
    _worker_state['soft_labels_df'] = soft_labels_df
    
    if not hasattr(creator, "FitnessVelocity"):
        creator.create("FitnessVelocity", base.Fitness, weights=(1.0,)) 
    if not hasattr(creator, "Individual"):
         if hasattr(creator, "FitnessVelocity"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessVelocity)

def _calculate_mimicry_score(predictions_series: pd.Series, soft_labels_df: pd.DataFrame) -> float:
    """Calculates the Mimicry Score (Inverse Log Loss) with robust alignment."""
    try:
        aligned_labels_df = soft_labels_df.loc[predictions_series.index]
        labels_np = aligned_labels_df.values
        predictions_np = predictions_series.values
        n_classes = labels_np.shape[1]
        pred_probs = np.zeros((len(predictions_np), n_classes))
        pred_probs[predictions_np == True, 1] = 1.0
        pred_probs[predictions_np == False, 0] = 1.0
        loss = log_loss(labels_np, pred_probs, labels=np.arange(n_classes))
        mimicry_score = 1.0 / (1.0 + loss)
        return mimicry_score
    except Exception:
        return 0.0

# Global counter for diagnostics
_eval_counter = [0]

_worker_fitness_context = None

def _evaluate_fitness_loky_worker(individual):
    """The task function executed by loky workers (Symbiotic Velocity)."""
    # Retrieve context
    evaluator = _worker_state.get('fitness_evaluator')
    pset = _worker_state.get('pset')
    config = _worker_state.get('config', {})
    soft_labels_df = _worker_state.get('soft_labels_df')
    
    parsimony_coeff = config.get('gp_parsimony_coeff', 0.01)
    mimicry_weight = config.get('gp_symbiotic_mimicry_weight', 0.3)

    if evaluator is None or pset is None: return (-1e9,)

    try:
        strategy_logic_func = gp.compile(expr=individual, pset=pset)
        
        # 1. Evaluate PnL Fitness (TTT) and get raw predictions (Pandas Series)
        metrics, predictions_series = evaluator(strategy_logic_func)
        velocity_fitness = metrics.get("velocity_fitness", -1e9)

        # 2. Calculate Mimicry Score (SDE)
        mimicry_score = 0.0
        if soft_labels_df is not None and mimicry_weight > 0 and predictions_series is not None:
            mimicry_score = _calculate_mimicry_score(predictions_series, soft_labels_df)

        # 3. Combine into Symbiotic Fitness
        symbiotic_fitness = (velocity_fitness * (1.0 - mimicry_weight)) + (mimicry_score * mimicry_weight)

        # 4. Apply Parsimony Pressure
        tree_size = len(individual)
        adjusted_fitness = symbiotic_fitness - (tree_size * parsimony_coeff)
        
        return (adjusted_fitness,)
    
    except Exception:
        return (-1e9,)

# --- Define Protected Operations ---
def protected_div(left, right):
    """Protected division - returns 1.0 for division by zero"""
    if abs(right) < 1e-10:  # Avoid division by zero/near-zero
        return 1.0
    return left / right

class StrategySynthesizer:
    def __init__(self, feature_names, fitness_evaluator, config_gp, logger=None, seed_dna=None, use_sae=True, reporter=None, asset_symbol=None, shared_state=None, soft_labels_df=None, X_train=None, y_train=None, X_val=None, y_val=None, returns_val=None):
        self.feature_names = feature_names
        self.fitness_evaluator = fitness_evaluator
        self.config = config_gp
        self.logger = logger if logger else logging.getLogger(__name__)
        self.seed_dna = seed_dna
        self.use_sae = use_sae
        self.reporter = reporter
        self.asset_symbol = asset_symbol
        self.shared_state = shared_state
        self.soft_labels_df = soft_labels_df

        from forge.evolution.opponent_templates import simple_rsi_opponent
        self.opponent_strategy_template = simple_rsi_opponent
        self.apex_predator = None
        self.implicit_brain = None

        self.fitness_cache = {}
        
        self.pset = setup_gp_primitives(self.feature_names)

        if self.use_sae:
            self.surrogate_manager = SurrogateManager(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, returns_val=returns_val,
                logger=self.logger
            )
            self.logger.info("[VELOCITY UPGRADE] Surrogate-Assisted Evolution ENABLED (using SurrogateManager)")

        if not hasattr(creator, "FitnessVelocity"):
            creator.create("FitnessVelocity", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessVelocity)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.get('gp_tournament_size', 5))
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("evaluate", _evaluate_fitness_loky_worker)
        self.executor = None

    def create_seeded_population(self, n):
        """Creates a population, seeding it with elite DNA if available."""
        population = self.toolbox.population(n=n)
        if self.seed_dna:
            self.logger.info(f"Seeding population with elite DNA.")
            num_to_replace = min(len(population), max(1, int(n * 0.1))) # Replace up to 10%
            seed_individuals = []
            for _ in range(num_to_replace):
                try:
                    # Create a tree from the string representation of the DNA
                    tree = gp.PrimitiveTree.from_string(self.seed_dna, self.pset)
                    ind = creator.Individual(tree)
                    seed_individuals.append(ind)
                except Exception as e:
                    self.logger.error(f"Error creating individual from seed DNA: {e}")
                    break
            
            # Replace the first n individuals with the seeded ones
            if seed_individuals:
                population[:len(seed_individuals)] = seed_individuals
        return population

    def run(self):
        pop_size = self.config.get('gp_population_size', 150)
        generations = self.config.get('gp_generations', 30)
        
        self.logger.info(f"[GP 2.0] Starting evolution (Symbiotic Velocity). Pop={pop_size}, Gen={generations}")

        is_subprocess = multiprocessing.current_process().name != 'MainProcess'
        if is_subprocess:
            self.logger.warning("[GP 2.0] Subprocess detected. Disabling Loky.")
            self.toolbox.register("map", map)
        else:
            # This function seems to be missing from the original file, assuming it exists elsewhere
            # and is imported correctly. If not, this will need to be addressed.
            # from some_utility_module import get_reusable_executor 
            try:
                executor = get_reusable_executor(
                    max_workers=max(1, cpu_count() - 1),
                    initializer=_init_worker,
                    initargs=(self.fitness_evaluator, self.pset, self.config, self.soft_labels_df),
                    timeout=300
                )
                self.toolbox.register("map", executor.map)
            except NameError:
                self.logger.warning("[GP 2.0] 'get_reusable_executor' not found. Falling back to sequential execution.")
                self.toolbox.register("map", map)


        pop = self.create_seeded_population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        try:
            if self.use_sae:
                pop, log = self._run_sae_evolution(pop, hof, stats, generations)
            else:
                pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=generations,
                                               stats=stats, halloffame=hof, verbose=True)
        except Exception as e:
            self.logger.error(f"[GP 2.0] Evolution failed: {e}", exc_info=True)
            return None
        
        self.logger.info("[GP 2.0] Evolution complete.")
        return hof[0] if hof else None

    def _run_sae_evolution(self, pop, hof, stats, generations):
        """
        VELOCITY UPGRADE: Custom evolution loop with Surrogate-Assisted Evolution.
        This version is updated to use the consolidated SurrogateManager.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Initial population evaluation using the surrogate
        self.logger.info(f"[SAE] Generation 0: Screening and evaluating initial population...")
        
        # The surrogate manager evaluates a subset and updates their fitness in place
        evaluated_pop = self.surrogate_manager.screen_and_evaluate(pop, generation=0)
        
        # For individuals not evaluated by the surrogate, assign a default low fitness
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = (-1e9,)

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(evaluated_pop), **record)
        self.logger.info(f"[SAE] Gen 0: {record}")

        # Evolution loop
        for gen in range(1, generations + 1):
            self.logger.info(f"[SAE] Generation {gen}/{generations}...")
            if self.reporter:
                self.reporter.set_status("GP 2.0 Evolution", f"Gen {gen}/{generations} - Best Fitness: {hof[0].fitness.values[0]:.4f}")

            # Standard DEAP evolution operators
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Identify individuals needing evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.logger.info(f"[SAE] Gen {gen}: {len(invalid_ind)} individuals need evaluation...")

            # Use the surrogate manager to screen and evaluate the subset
            evaluated_offspring = self.surrogate_manager.screen_and_evaluate(invalid_ind, generation=gen)

            # For offspring not evaluated by the surrogate, assign a default low fitness
            for ind in invalid_ind:
                if not ind.fitness.valid:
                    ind.fitness.values = (-1e9,)

            # The main population is replaced by the offspring
            pop[:] = offspring

            # Update hall of fame and statistics
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(evaluated_offspring), **record)
            self.logger.info(f"[SAE] Gen {gen}: {record}")

        return pop, logbook
