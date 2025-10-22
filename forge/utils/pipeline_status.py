
import threading

class PipelineStatus:
    """
    A thread-safe class to store and report the status of the Forge pipeline.
    This acts as the bridge between the background pipeline thread and the main Flask thread.
    """
    def __init__(self):
        self._lock = threading.RLock() # Changed to RLock to prevent deadlocks
        self.is_running = False
        self.stage = "Idle"
        self.detail = ""
        self.logs = []
        self._metadata = {}

    def add_metadata(self, key: str, value: any):
        """Adds or updates a metadata entry."""
        with self._lock:
            self._metadata[key] = value

    def start(self):
        """Marks the pipeline as running and clears old state."""
        with self._lock:
            self.is_running = True
            self.stage = "Initializing"
            self.detail = "Preparing to start..."
            self.logs = []
            self._metadata = {} # Clear metadata on start

    def stop(self):
        """Marks the pipeline as finished."""
        with self._lock:
            self.is_running = False
            self.stage = "Finished"
            self.detail = ""

    def set_status(self, stage: str, detail: str):
        """Updates the current stage and detail of the pipeline."""
        with self._lock:
            if self.is_running:
                self.stage = stage
                self.detail = detail
                self.log(f"{stage}: {detail}")

    def log(self, message: str):
        """Adds a log message to the list."""
        with self._lock:
            try:
                self.logs.append(message)
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to append log message '{message}' to self.logs: {e}")

    def get_status(self) -> dict: # Renamed from get_status_and_clear_logs
        """
        Retrieves the current status.
        """
        with self._lock:
            status = {
                "is_running": self.is_running,
                "stage": self.stage,
                "detail": self.detail,
                "logs": self.logs[:],
                "metadata": self._metadata.copy() # Return a copy of the metadata
            }
            # The self.logs.clear() line is now removed.
            return status
