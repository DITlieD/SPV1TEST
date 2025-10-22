
from river import drift
import pandas as pd

class DriftDetector:
    """
    Enhanced wrapper for the ADWIN drift detection algorithm.
    It can now run on a historical series of data to find the last drift point,
    which is crucial for determining the minimum relevant training window.
    """
    def __init__(self, delta=0.005, grace_period=100):
        """
        Initializes the ADWIN detector.
        Args:
            delta (float): Confidence value. A smaller delta means higher confidence is required to signal drift.
            grace_period (int): The minimum number of instances to observe before checking.
        """
        self.detector = drift.ADWIN(delta=delta, grace_period=grace_period)
        self._drift_detected = False
        self.drift_point = -1 # Will store the index of the last detected drift
        self.n = 0

    def update(self, value: float):
        """Updates the detector with a single new value."""
        if not isinstance(value, (int, float)):
            return

        self.n += 1
        self.detector.update(value)
        
        if self.detector.drift_detected:
            # ADWIN flags drift *after* processing the value that caused it.
            # The change occurred at the start of the new window.
            self.drift_point = self.n - self.detector.width
            self._drift_detected = True
            print(f"🔥 Drift detected at index {self.drift_point} (Window size: {self.detector.width})")

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def reset(self):
        """Resets the detector's state."""
        self.detector._reset()
        self._drift_detected = False
        self.drift_point = -1
        self.n = 0

    def run_on_series(self, data: pd.Series) -> int:
        """
        Runs the detector over an entire pandas Series to find the last drift point.

        Args:
            data (pd.Series): A series of market data, e.g., volatility or returns.

        Returns:
            int: The index of the start of the last detected market regime, or 0 if no drift.
        """
        print(f"[DriftDetector] Analyzing series of length {len(data)} for structural breaks...")
        self.reset()
        for value in data:
            self.update(value)
        
        if self.drift_point != -1:
            print(f"[DriftDetector] Last significant market drift detected at index: {self.drift_point}")
            return self.drift_point
        else:
            print("[DriftDetector] No significant drift detected in the series.")
            return 0
