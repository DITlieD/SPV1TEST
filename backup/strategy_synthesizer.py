from deap import base, creator, tools, gp, algorithms # Ensure algorithms is imported
import random
import operator
import numpy as np
import multiprocessing
from loky import ProcessPoolExecutor as LokyPoolExecutor, cpu_count as loky_cpu_count
import os
import functools # Import for partial
import logging
import warnings
from sklearn.metrics import log_loss
from forge.evolution.surrogate_manager import DNAVectorizer, FitnessOracle, SurrogateAssistedEvolution
from forge.analysis.complexity_manager import StrategyComplexityEstimator
from forge.evolution.symbiotic_crucible import ImplicitBrain
import torch
import config
import joblib


# Suppress divide by zero warnings in GP operations (handled safely)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')

# ==============================================================================
# LOKY Worker State and Functions (for parallel GP evaluation)
# ==============================================================================
_worker_state = {}

def _init_loky_worker(fitness_evaluator_func, pset, parsimony_coeff, feature_names):
    """Initialize loky worker with single-threading and DEAP creators."""
    try:
        # DIAGNOSTIC
        with open("gp_diagnostic.txt", "a") as f:
            f.write(f"[INIT] Worker initialization started, PID={os.getpid()}\n")
            f.write(f"[INIT] Args: evaluator={fitness_evaluator_func is not None}, pset={pset is not None}\n")
            f.flush()

        # CRITICAL: Force single-threading in numpy/BLAS to avoid thread contention
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        # Store context in worker state
        _worker_state['fitness_evaluator'] = fitness_evaluator_func
        _worker_state['pset'] = pset
        _worker_state['parsimony_coeff'] = parsimony_coeff
        _worker_state['feature_names'] = feature_names

        # Initialize DEAP creators in worker
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        # DIAGNOSTIC
        with open("gp_diagnostic.txt", "a") as f:
            f.write(f"[INIT] Worker initialized successfully! state_keys={list(_worker_state.keys())}\n")
            f.flush()
    except Exception as e:
        with open("gp_diagnostic.txt", "a") as f:
            f.write(f"[INIT FATAL] Worker init failed: {e}\n")
            f.flush()
        raise

# Global counter for diagnostics
_eval_counter = [0]

_worker_fitness_context = None

def _evaluate_fitness_loky_worker(args_tuple):
    """
    Worker function to evaluate individual fitness.
    """
    global _worker_fitness_context
    individual, evaluator, pset, parsimony_coeff, feature_names, opponent_strategy_template, implicit_brain, shared_state, fitness_context_path = args_tuple

    if evaluator is None or pset is None:
        return (-1e9,)

    try:
        # Load fitness context once per worker
        if _worker_fitness_context is None and fitness_context_path:
            _worker_fitness_context = joblib.load(fitness_context_path)

        # Compile GP tree to executable function
        strategy_logic_func = gp.compile(expr=individual, pset=pset)

        # Evaluate using the fitness evaluator
        metrics, our_strategy_signals, fitness_context_from_eval = evaluator(strategy_logic_func)

        raw_fitness = metrics.get("adaptive_fitness", metrics.get("ttt_fitness", -1e9))

        # Parsimony pressure
        tree_size = len(individual)
        parsimony_penalty = tree_size * parsimony_coeff

        # Feature diversity bonus
        used_features = {node.name for node in individual if hasattr(node, 'name') and node.name in feature_names}
        num_features_used = len(used_features)
        total_features = len(feature_names)
        diversity_bonus = -10.0 if num_features_used < 3 else ((num_features_used - 3) / (total_features - 3) if total_features > 3 else 0)

        # KCM Complexity Penalty
        estimator = StrategyComplexityEstimator()
        complexity_eval = estimator.evaluate_complexity(str(individual))
        complexity_penalty = complexity_eval['penalty']

        # SDE Mimicry Score
        mimicry_score = 0.0
        if implicit_brain and _worker_fitness_context is not None:
            try:
                # VELOCITY V3.3: Check if all required features are present
                required_features = set(implicit_brain.features)
                available_features = set(_worker_fitness_context.columns)
                
                if required_features.issubset(available_features):
                    brain_features = _worker_fitness_context[implicit_brain.features]
                    soft_labels = implicit_brain.model.predict_proba(brain_features)
                    
                    gp_probs = np.zeros_like(soft_labels)
                    our_strategy_signals = our_strategy_signals.astype(int)
                    valid_indices = (our_strategy_signals >= 0) & (our_strategy_signals < 3)
                    gp_probs[np.arange(len(gp_probs))[valid_indices], our_strategy_signals[valid_indices]] = 1
                    
                    mimicry_loss = log_loss(gp_probs, soft_labels, labels=[0,1,2])
                    mimicry_score = -mimicry_loss
                else:
                    missing_features = required_features - available_features
                    # Log this only once to avoid spamming logs
                    if not getattr(_evaluate_fitness_loky_worker, 'logged_missing_features', False):
                        logging.getLogger(__name__).warning(f"SDE mimicry disabled: Missing features in fitness context: {missing_features}")
                        _evaluate_fitness_loky_worker.logged_missing_features = True

            except Exception as e:
                logging.getLogger(__name__).error(f"Error calculating SDE mimicry score: {e}")

        # Apex Predator Dual Fitness
        inferred_opponent_params = shared_state.get('inferred_opponent_params') if shared_state else None
        if inferred_opponent_params and _worker_fitness_context is not None:
            try:
                from forge.acp.acp_engines import ApexPredator, ChimeraEngine
                apex_predator = ApexPredator(chimera_engine=ChimeraEngine(strategy_template=opponent_strategy_template))
                opponent_signals = opponent_strategy_template(_worker_fitness_context, inferred_opponent_params)
                dual_fitness = apex_predator.calculate_dual_fitness(
                    our_strategy_signals=our_strategy_signals,
                    traditional_fitness=raw_fitness,
                    opponent_strategy_signals=opponent_signals,
                    price_df=_worker_fitness_context
                )
                raw_fitness = dual_fitness
            except Exception as e:
                logging.getLogger(__name__).error(f"Error calculating dual fitness: {e}. Using base fitness.")

        adjusted_fitness = raw_fitness - parsimony_penalty + diversity_bonus + (0.1 * mimicry_score) - (complexity_penalty * 0.2)
        return (adjusted_fitness,)

    except Exception as e:
        logging.getLogger(__name__).error(f"LOKY WORKER CRASH: {e}", exc_info=True)
        return (-1e9,)

# --- Define Protected Operations ---
def protected_div(left, right):
    """Protected division - returns 1.0 for division by zero"""
    if abs(right) < 1e-10:  # Avoid division by zero/near-zero
        return 1.0
    return left / right

def setup_gp_primitives(feature_names):
    pset = gp.PrimitiveSetTyped("MAIN", [float] * len(feature_names), bool)

    # --- Boolean Operations ---
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)

    # --- Relational Operations ---
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.gt, [float, float], bool)

    # --- Arithmetic Operations ---
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protected_div, [float, float], float)

    # --- Terminals ---
    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)
    pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1), float)
    
    arg_map = {f"ARG{i}": name for i, name in enumerate(feature_names)}
    pset.renameArguments(**arg_map)

    return pset

class StrategySynthesizer:
    def __init__(self, feature_names, fitness_evaluator, population_size=50, generations=20, logger=None, seed_dna=None, use_sae=True, reporter=None, asset_symbol=None, shared_state=None, fitness_context_path=None):
        self.feature_names = feature_names
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.generations = generations
        self.logger = logger if logger else logging.getLogger(__name__)
        self.seed_dna = seed_dna
        self.use_sae = use_sae
        self.reporter = reporter
        self.asset_symbol = asset_symbol
        self.shared_state = shared_state
        self.fitness_context_path = fitness_context_path

        from forge.evolution.opponent_templates import simple_rsi_opponent
        self.opponent_strategy_template = simple_rsi_opponent

        # Apex Predator is now instantiated in the worker if params are available
        self.apex_predator = None

        self.implicit_brain = None
        if self.asset_symbol:
            sanitized_symbol = self.asset_symbol.replace('/', '').replace(':', '_')
            brain_path = f"models/sde_implicit_brain_{sanitized_symbol}.pkl"
            if os.path.exists(brain_path):
                try:
                    self.implicit_brain = joblib.load(brain_path)
                    self.logger.info(f"Implicit Brain for {self.asset_symbol} (SDE - LightGBM) loaded successfully from {brain_path}.")
                except Exception as e:
                    self.logger.error(f"Failed to load Implicit Brain for {self.asset_symbol}: {e}. SDE disabled.")
            else:
                self.logger.warning(f"Implicit Brain model for {self.asset_symbol} not found at {brain_path}. SDE disabled for this asset.")
        else:
            self.logger.warning("No asset_symbol provided to StrategySynthesizer. SDE disabled.")

        self.fitness_cache = {}
        self.parsimony_coeff = 0.1  # MFT: Increased 10x for 1m noise resistance (was 0.001)

        self.pset = setup_gp_primitives(self.feature_names)

        # VELOCITY UPGRADE: Initialize SAE components
        if self.use_sae:
            self.vectorizer = DNAVectorizer(feature_names)
            self.oracle = FitnessOracle(self.vectorizer)
            self.sae = SurrogateAssistedEvolution(self.oracle, top_pct=0.02)
            self.logger.info("[VELOCITY UPGRADE] Surrogate-Assisted Evolution ENABLED")

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # Register wrapper for evaluate that packages args for the worker
        # This is needed because SAE calls toolbox.evaluate(ind) directly
        eval_call_counter = [0]

        def evaluate_single_individual(ind):
            """Wrapper that packages individual with evaluator for worker function."""
            eval_call_counter[0] += 1
            if eval_call_counter[0] <= 3:
                with open("gp_diagnostic.txt", "a") as f:
                    f.write(f"[EVALUATE_SINGLE #{eval_call_counter[0]}] Packaging individual for worker\n")
                    f.flush()
            return _evaluate_fitness_loky_worker((ind, self.fitness_evaluator, self.pset, self.parsimony_coeff, self.feature_names, self.opponent_strategy_template, self.implicit_brain, self.shared_state, self.fitness_context_path))

        self.toolbox.register("evaluate", evaluate_single_individual)

        # Loky executor will be initialized in run() method
        self.executor = None

    def evaluate_fitness(self, individual, adversarial_scenario=None):
        tree_str = str(individual)
        if tree_str in self.fitness_cache and adversarial_scenario is None:
            return self.fitness_cache[tree_str]

        strategy_logic_func = gp.compile(expr=individual, pset=self.pset)
        metrics = self.fitness_evaluator(strategy_logic_func, adversarial_scenario)

        # VELOCITY UPGRADE: Use Time-to-Target (TTT) fitness instead of log_wealth
        # TTT is negative (faster is more negative, so we maximize it)
        raw_fitness = metrics.get("ttt_fitness", -1e9)

        # Parsimony penalty: Penalize overly complex trees
        tree_size = len(individual)
        parsimony_penalty = tree_size * self.parsimony_coeff

        # FEATURE DIVERSITY BONUS: Reward strategies that use multiple features
        # Count unique features (ARG0, ARG1, etc. become feature names after renaming)
        used_features = set()
        for node in individual:
            if hasattr(node, 'name'):
                # Check if this is a feature terminal (will be in feature_names)
                if node.name in self.feature_names:
                    used_features.add(node.name)

        num_features_used = len(used_features)
        total_features = len(self.feature_names)

        # CRITICAL: MASSIVE penalty for single/dual-feature strategies
        # Must use at least 3 features to avoid penalty
        if num_features_used < 3:
            # Extreme penalty - makes single-feature strategies uncompetitive
            diversity_penalty = -10.0
            diversity_bonus = diversity_penalty
        else:
            # Bonus scales from 0 (3 features) to +1.0 (all features)
            diversity_ratio = (num_features_used - 3) / (total_features - 3) if total_features > 3 else 0
            diversity_bonus = diversity_ratio * 1.0

        adjusted_fitness = raw_fitness - parsimony_penalty + diversity_bonus

        result = (adjusted_fitness,)
        self.fitness_cache[tree_str] = result
        return result

    def create_seeded_population(self, n):
        pop = []
        seed_individual = None
        if self.seed_dna and self.seed_dna.get('architecture') == 'GP2_Evolved':
            try:
                seed_tree = self.seed_dna['tree']
                seed_individual = self.toolbox.clone(seed_tree)
                if hasattr(seed_individual, 'fitness') and seed_individual.fitness.valid:
                   del seed_individual.fitness.values
                pop.append(seed_individual)
                self.logger.info("[GP 2.0] Injected Champion DNA.")
            except Exception as e:
                self.logger.warning(f"[GP 2.0] Failed to inject seed DNA. Error: {e}")
                seed_individual = None

        if seed_individual:
            mutation_count = int(n * 0.2)
            for _ in range(mutation_count):
                try:
                    mutant = self.toolbox.clone(seed_individual)
                    mutant, = self.toolbox.mutate(mutant)
                    del mutant.fitness.values
                    pop.append(mutant)
                except Exception:
                    pass

        remaining = n - len(pop)
        if remaining > 0:
            random_pop = self.toolbox.population(n=remaining)
            pop.extend(random_pop)
            
        return pop

    def run(self):
        # DIAGNOSTIC
        with open("gp_diagnostic.txt", "a") as f:
            f.write(f"\n{'='*60}\n[RUN START] GP evolution beginning\n")
            f.flush()

        # CRITICAL DEADLOCK FIX: Detect Nested Parallelism
        # If we're already in a subprocess (e.g., Forge Worker), disable Loky to prevent deadlock
        is_subprocess = multiprocessing.current_process().name != 'MainProcess'

        if is_subprocess:
            self.logger.warning("[GP 2.0] Detected execution within a subprocess (Forge Worker). Disabling Loky to prevent deadlock...")
            with open("gp_diagnostic.txt", "a") as f:
                f.write(f"[RUN] SUBPROCESS DETECTED - Using sequential map()\n")
                f.flush()
            self.toolbox.register("map", map)
            self.executor = None
        else:
            # LOKY INITIALIZATION: Setup parallel executor with worker initialization
            # Only initialize if we're in the main process
            n_workers = loky_cpu_count()
            self.logger.info(f"[GP 2.0] Initializing loky executor with {n_workers} workers...")

            try:
                with open("gp_diagnostic.txt", "a") as f:
                    f.write(f"[RUN] Attempting loky setup, workers={n_workers}\n")
                    f.flush()

                # DIAGNOSTIC: Test if fitness_evaluator can be pickled (just for logging)
                import pickle
                try:
                    pickled = pickle.dumps(self.fitness_evaluator)
                    with open("gp_diagnostic.txt", "a") as f:
                        f.write(f"[RUN] fitness_evaluator IS picklable with standard pickle (size={len(pickled)} bytes)\n")
                        f.flush()
                except Exception as pickle_err:
                    with open("gp_diagnostic.txt", "a") as f:
                        f.write(f"[RUN INFO] fitness_evaluator NOT picklable with standard pickle: {type(pickle_err).__name__}\n")
                        f.write(f"[RUN INFO] This is OK - cloudpickle will handle it as function args\n")
                        f.flush()
                    # Don't raise - cloudpickle can handle closures as function args!

                # CRITICAL FIX: Don't use initializer (can't pickle closures)
                # Instead, pass evaluator with each function call (cloudpickle CAN handle this)
                self.executor = LokyPoolExecutor(
                    max_workers=n_workers,
                    timeout=600  # 10 minute timeout per evaluation
                )

                with open("gp_diagnostic.txt", "a") as f:
                    f.write(f"[RUN] LokyPoolExecutor created WITHOUT initializer\n")
                    f.flush()

                # Create custom map function that packages args for each individual
                def custom_loky_map(_, individuals):
                    """Custom map that packages evaluator with each individual."""
                    with open("gp_diagnostic.txt", "a") as f:
                        f.write(f"[CUSTOM_MAP] Packaging {len(individuals)} individuals with evaluator\n")
                        f.flush()

                    # Package each individual with evaluator, pset, etc.
                    args_list = [
                        (ind, self.fitness_evaluator, self.pset, self.parsimony_coeff, self.feature_names, self.opponent_strategy_template, self.implicit_brain, self.shared_state, self.fitness_context_path)
                        for ind in individuals
                    ]

                    with open("gp_diagnostic.txt", "a") as f:
                        f.write(f"[CUSTOM_MAP] Calling executor.map with packaged args\n")
                        f.flush()

                    # Call the worker function with packaged args
                    results = list(self.executor.map(_evaluate_fitness_loky_worker, args_list))

                    with open("gp_diagnostic.txt", "a") as f:
                        f.write(f"[CUSTOM_MAP] Got {len(results)} results\n")
                        f.flush()

                    return results

                # Register custom map with DEAP toolbox
                self.toolbox.register("map", custom_loky_map)
                self.logger.info(f"[VELOCITY] Loky executor initialized - TRUE parallelization enabled!")

                with open("gp_diagnostic.txt", "a") as f:
                    f.write(f"[RUN] Loky map registered\n")
                    f.flush()

            except Exception as e:
                self.logger.error(f"[GP 2.0] Failed to initialize loky: {e}. Falling back to sequential.")
                with open("gp_diagnostic.txt", "a") as f:
                    f.write(f"[RUN FATAL] Loky failed: {e}\n")
                    f.flush()
                self.toolbox.register("map", map)
                self.executor = None

        pop = self.create_seeded_population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.logger.info(f"[GP 2.0] Starting evolution for {self.generations} generations, Pop: {self.population_size}...")

        try:
            if self.use_sae:
                # VELOCITY UPGRADE: Use SAE-enhanced evolution
                pop, log = self._run_sae_evolution(pop, hof, stats)
            else:
                # Standard evolution
                pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.generations,
                                               stats=stats, halloffame=hof, verbose=True)
        except Exception as e:
            self.logger.error(f"[GP 2.0] Error during evolution: {e}", exc_info=True)
            return None
        finally:
            # CRITICAL: Shut down the loky executor after the run if it was initialized
            if self.executor:
                self.logger.info("[GP 2.0] Shutting down loky executor.")
                # Use shutdown(wait=True) to ensure all processes terminate cleanly
                self.executor.shutdown(wait=True)
                self.executor = None

        self.logger.info("[GP 2.0] Evolution complete.")

        return hof[0] if hof else None

    def _run_sae_evolution(self, pop, hof, stats):
        """
        VELOCITY UPGRADE: Custom evolution loop with Surrogate-Assisted Evolution.

        The SAE loop:
        1. Oracle screens entire population
        2. Only top 2% undergo actual backtesting
        3. Evolution proceeds with actual fitness
        4. Oracle is continuously updated

        Result: 99%+ speedup in evaluation time.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Evaluate initial population with SAE
        self.logger.info(f"[SAE] Generation 0: Evaluating initial population...")
        evaluated = self.sae.screen_and_evaluate(pop, self.fitness_evaluator, self.toolbox)

        # Assign fitness to evaluated individuals
        for ind, fitness in evaluated:
            # Find this individual in the population and set its fitness
            for p_ind in pop:
                if str(p_ind) == str(ind):
                    p_ind.fitness.values = fitness
                    break

        # For unevaluated individuals, assign worst fitness
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = (-1e9,)

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(evaluated), **record)
        self.logger.info(f"[SAE] Gen 0: {record}")

        # Evolution loop
        for gen in range(1, self.generations + 1):
            self.logger.info(f"[SAE] Generation {gen}/{self.generations}...")
            if self.reporter:
                self.reporter.set_status("GP 2.0 Evolution", f"Gen {gen}/{self.generations} - Best Fitness: {hof[0].fitness.values[0]:.4f}")

            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # SAE Evaluation: Screen and evaluate only top candidates
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.logger.info(f"[SAE] Gen {gen}: {len(invalid_ind)} individuals need evaluation...")

            evaluated = self.sae.screen_and_evaluate(invalid_ind, self.fitness_evaluator, self.toolbox)

            # Assign fitness to evaluated individuals
            for ind, fitness in evaluated:
                for inv_ind in invalid_ind:
                    if str(inv_ind) == str(ind):
                        inv_ind.fitness.values = fitness
                        break

            # For unevaluated individuals, assign worst fitness
            for ind in invalid_ind:
                if not ind.fitness.valid:
                    ind.fitness.values = (-1e9,)

            # Replace population
            pop[:] = offspring

            # Update hall of fame and stats
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(evaluated), **record)
            self.logger.info(f"[SAE] Gen {gen}: {record}")

        return pop, logbook
