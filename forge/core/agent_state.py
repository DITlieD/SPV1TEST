# forge/core/agent_state.py
import json
import os
import threading

class AgentStateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentStateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, state_file='data/agent_states.json'):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.state_file = state_file
        self.states = self._load_states()

    def _load_states(self):
        with self._lock:
            if not os.path.exists(self.state_file):
                # Create the data directory if it doesn't exist
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
                return {}
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}

    def get_state(self, agent_id, default_state=None):
        """Gets the state for a specific agent."""
        return self.states.get(agent_id, default_state)

    def save_state(self, agent_id, state_data):
        """Saves the state for a specific agent."""
        with self._lock:
            self.states[agent_id] = state_data
            self._persist_states()

    def remove_state(self, agent_id):
        """Removes the state for a specific agent."""
        with self._lock:
            if agent_id in self.states:
                del self.states[agent_id]
                self._persist_states()

    def _persist_states(self):
        """Writes the current states dictionary to the JSON file."""
        # This is called within a lock
        with open(self.state_file, 'w') as f:
            json.dump(self.states, f, indent=4)

# Singleton instance
state_manager = AgentStateManager()
