# forge/monitoring/ai_analyst.py

from deap import gp
import operator

class AIAnalyst:
    """
    Simulates an AI Analyst by translating a DEAP genetic programming tree 
    into a human-readable string representation of the trading strategy.
    """
    def __init__(self, pset):
        self.pset = pset
        # A map to make the output more readable
        self.op_map = {
            'and_': 'AND',
            'or_': 'OR',
            'not_': 'NOT',
            'gt': '>',
            'lt': '<',
            'add': '+',
            'sub': '-',
            'mul': '*',
            'protected_div': '/',
        }

    def interpret_strategy(self, strategy_tree):
        """Converts a DEAP GP tree into a more readable string."""
        nodes, edges, labels = gp.graph(strategy_tree)
        
        # Build a readable string representation
        string_representation = ""
        for i, node in enumerate(nodes):
            # --- ROBUSTNESS FIX ---
            # Check if the node is an integer (can happen with malformed trees)
            if isinstance(node, int):
                op_name = str(node) # Convert int to string
            else:
                # Proceed as normal if it's a valid node object
                op_name = self.op_map.get(node.name, node.name)
            
            string_representation += f"Step {i}: {op_name}\n"
            
        # (Further logic for analyzing edges and labels can be added here)
        
        return f"Strategy Logic:\n{string_representation}"

def get_strategy_explanation(strategy_tree: gp.PrimitiveTree, pset) -> str:
    """
    High-level function to get the explanation for a given strategy tree.
    """
    if not strategy_tree:
        return "No valid strategy tree provided."
        
    analyst = AIAnalyst(pset)
    return analyst.interpret_strategy(strategy_tree)