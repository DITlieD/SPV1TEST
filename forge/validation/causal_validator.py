import pandas as pd
import numpy as np
from dowhy import CausalModel

class CausalValidator:
    """
    Uses the DoWhy library to validate the causal relationship between a feature
    and future returns, controlling for common confounders.
    """
    def __init__(self, df: pd.DataFrame, treatment: str, outcome: str, common_causes: list):
        self.df = df
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.model = self._create_model()

    def _create_model(self):
        """Initializes the DoWhy CausalModel."""
        try:
            # Create a causal model from the data and given graph
            model = CausalModel(
                data=self.df,
                treatment=self.treatment,
                outcome=self.outcome,
                common_causes=self.common_causes
            )
            return model
        except Exception as e:
            print(f"[CausalValidator] Error creating causal model: {e}")
            return None

    def estimate_effect(self):
        """Estimates the causal effect of the treatment on the outcome."""
        if self.model is None: return None
        try:
            # Identify the causal effect
            identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=True)

            # Estimate the effect using a linear regression model
            causal_estimate = self.model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                test_significance=True
            )
            return causal_estimate
        except Exception as e:
            print(f"[CausalValidator] Error estimating effect: {e}")
            return None

    def refute_estimate(self, estimate) -> bool:
        """Performs refutation tests to check the robustness of the estimate."""
        if estimate is None: return False
        try:
            # 1. Placebo Treatment Refuter
            placebo_refuter = self.model.refute_estimate(
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute"
            )
            if placebo_refuter.p_value > 0.05:
                print(f"[CausalValidator] Refutation Failed (Placebo): p-value ({placebo_refuter.p_value:.2f}) is not significant.")
                return False

            # 2. Random Common Cause Refuter
            random_cause_refuter = self.model.refute_estimate(
                estimate, 
                method_name="random_common_cause"
            )
            # If the new estimate is not close to the original, it's a good sign
            if abs(random_cause_refuter.new_effect - estimate.value) < 1e-6:
                print("[CausalValidator] Refutation Failed (Random Cause): Effect did not change significantly.")
                return False

            print("[CausalValidator] ✅ All refutation tests passed.")
            return True
        except Exception as e:
            print(f"[CausalValidator] Error during refutation: {e}")
            return False