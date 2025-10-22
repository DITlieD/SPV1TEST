
# forge/evolution/adversarial_forge.py

from deap import base, creator, tools, gp, algorithms
import random
import operator
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# Predator: Evolves adversarial market scenarios
# ==============================================================================

# --- Predator Primitives (functions that modify market data) ---

def add_noise(data, column, noise_level):
    """Adds Gaussian noise to a column."""
    data[column] += np.random.normal(0, noise_level, len(data))
    return data

def create_stop_hunt(data, column, window, strength):
    """Simulates a stop-hunt by creating a sharp price movement."""
    for i in range(window, len(data)):
        if random.random() < 0.01: # Trigger a stop hunt
            direction = 1 if random.random() < 0.5 else -1
            data.loc[i-window:i, column] *= (1 + direction * strength)
    return data

# --- Predator GP Setup ---

def setup_predator_primitives():
    pset = gp.PrimitiveSet("PREDATOR", 1) # 1 input: the dataframe
    pset.addPrimitive(add_noise, 3) # func, arg1, arg2, arg3
    pset.addPrimitive(create_stop_hunt, 4) # func, arg1, arg2, arg3, arg4
    
    # Terminals for the primitives
    pset.addTerminal('close')
    pset.addTerminal('open')
    pset.addTerminal('high')
    pset.addTerminal('low')
    pset.addTerminal(0.01)
    pset.addTerminal(0.05)
    pset.addTerminal(10)
    pset.addTerminal(20)
    
    return pset

# ==============================================================================
# Co-evolutionary Forge
# ==============================================================================

class AdversarialForge:
    def __init__(self, strategy_synthesizer, X_gp_input, X_fitness_context, logger=None):
        self.prey_synthesizer = strategy_synthesizer
        self.X_gp_input = X_gp_input
        self.X_fitness_context = X_fitness_context
        self.logger = logger if logger else logging.getLogger(__name__)

        # Predator setup
        self.predator_pset = setup_predator_primitives()
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "PredatorIndividual"):
            creator.create("PredatorIndividual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        
        self.predator_toolbox = base.Toolbox()
        self.predator_toolbox.register("expr", gp.genHalfAndHalf, pset=self.predator_pset, min_=1, max_=2)
        self.predator_toolbox.register("individual", tools.initIterate, creator.PredatorIndividual, self.predator_toolbox.expr)
        self.predator_toolbox.register("population", tools.initRepeat, list, self.predator_toolbox.individual)
        self.predator_toolbox.register("compile", gp.compile, pset=self.predator_pset)
        self.predator_toolbox.register("select", tools.selTournament, tournsize=3)
        self.predator_toolbox.register("mate", gp.cxOnePoint)
        self.predator_toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.predator_toolbox.register("mutate", gp.mutUniform, expr=self.predator_toolbox.expr_mut, pset=self.predator_pset)

    def run(self, n_generations=50):
        self.logger.info("[Adversarial Forge] Starting co-evolution...")

        # Initialize populations
        prey_pop = self.prey_synthesizer.create_seeded_population(n=self.prey_synthesizer.population_size)
        predator_pop = self.predator_toolbox.population(n=50) # Smaller predator population

        for gen in range(n_generations):
            self.logger.info(f"[Adversarial Forge] Co-evolution Generation {gen + 1}/{n_generations}")

            # --- 1. Evaluate and Evolve Prey ---
            self.logger.info("  -> Evolving Prey...")
            for prey in prey_pop:
                # Select a random predator to compete against
                predator = random.choice(predator_pop)
                adversarial_scenario = self.predator_toolbox.compile(expr=predator)
                
                # Evaluate prey fitness in the adversarial scenario
                prey.fitness.values = self.prey_synthesizer.evaluate_fitness(prey, adversarial_scenario)

            # Evolve prey population
            prey_offspring = algorithms.varAnd(prey_pop, self.prey_synthesizer.toolbox, cxpb=0.5, mutpb=0.2)
            prey_pop[:] = prey_offspring

            # --- 2. Evaluate and Evolve Predators ---
            self.logger.info("  -> Evolving Predators...")
            for predator in predator_pop:
                total_prey_fitness = 0
                for prey in prey_pop:
                    adversarial_scenario = self.predator_toolbox.compile(expr=predator)
                    fitness = self.prey_synthesizer.evaluate_fitness(prey, adversarial_scenario)
                    total_prey_fitness += fitness[0]
                
                # Predator fitness is the inverse of the average prey fitness
                predator.fitness.values = (-total_prey_fitness / len(prey_pop),)

            # Evolve predator population
            predator_offspring = algorithms.varAnd(predator_pop, self.predator_toolbox, cxpb=0.5, mutpb=0.2)
            predator_pop[:] = predator_offspring

        # Return the best prey individual from the final population
        best_prey = tools.selBest(prey_pop, k=1)[0]
        return best_prey
