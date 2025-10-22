# forge/evolution/gp_primitives.py
import operator
import numpy as np
from deap import gp

# Define protected division
def protectedDiv(left, right):
    # Ensure inputs are treated as floats
    left = float(left)
    right = float(right)
    if abs(right) < 1e-6:
        return 1.0
    return left / right

# --- Protected Logical Operators ---
def protected_and(left, right):
    return bool(left) and bool(right)

def protected_or(left, right):
    return bool(left) or bool(right)

def protected_not(val):
    return not bool(val)

def setup_gp_primitives(feature_names):
    """Sets up the GP primitive set for strategy evolution."""
    # The arity (number of inputs) matches the number of features
    pset = gp.PrimitiveSet("MAIN", len(feature_names))

    # 1. Mathematical Operators
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    
    # 2. Comparison Operators (Crucial for evolving boolean conditions: True=Buy)
    pset.addPrimitive(operator.gt, 2)
    pset.addPrimitive(operator.lt, 2)

    # 3. Logical Operators (Protected)
    pset.addPrimitive(protected_and, 2, name="AND")
    pset.addPrimitive(protected_or, 2, name="OR")
    pset.addPrimitive(protected_not, 1, name="NOT")

    # 4. Terminals
    pset.addEphemeralConstant("rand_float", lambda: np.random.uniform(-1, 1))
    
    # Rename arguments for interpretability
    rename_map = {f"ARG{i}": name for i, name in enumerate(feature_names)}
    pset.renameArguments(**rename_map)
    
    return pset
