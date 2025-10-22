import optuna
from forge.blueprint_factory.genetic_algorithm import create_random_blueprint

class BlueprintFactory:
    """
    Manages the evolution of model blueprints based on Arena results,
    aggressively favoring the DNA of recent winners.
    """
    def __init__(self):
        self.elite_pool = [] # A list of high-performing model blueprints (DNA)
        self.max_elite_size = 10
        print("[Blueprint Factory] Initialized.")

    def add_to_elite_pool(self, elite_blueprint):
        """Adds a successful model's DNA to the elite pool."""
        # Avoid duplicates
        if any(e.hyperparameters == elite_blueprint.hyperparameters for e in self.elite_pool):
            return
            
        self.elite_pool.append(elite_blueprint)
        # Maintain the size of the pool
        if len(self.elite_pool) > self.max_elite_size:
            # Sort by fitness and keep the best
            self.elite_pool.sort(key=lambda bp: bp.fitness, reverse=True)
            self.elite_pool = self.elite_pool[:self.max_elite_size]
        
        print(f"[Blueprint Factory] Added new DNA to Elite Pool. Pool size: {len(self.elite_pool)}")

    def create_new_study(self) -> optuna.Study:
        """
        Creates a new Optuna study, aggressively warm-starting it with
        the DNA from the entire Elite Pool.
        """
        study = optuna.create_study(directions=['maximize', 'maximize'])
        
        if not self.elite_pool:
            print("[Blueprint Factory] Elite Pool is empty. Starting with random trials.")
            # Enqueue a random blueprint to kickstart the process
            random_bp = create_random_blueprint()
            study.enqueue_trial(random_bp.hyperparameters)
        else:
            print(f"[Blueprint Factory] Seeding new study with {len(self.elite_pool)} elite blueprints.")
            # Multi-Parent Warm Start
            for elite_bp in self.elite_pool:
                study.enqueue_trial(elite_bp.hyperparameters)
                
        return study
