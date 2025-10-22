
import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for background threads
import matplotlib.pyplot as plt
import seaborn as sns
import shap # Import shap

class XaiReporter:
    """
    Generates a comprehensive XAI (Explainable AI) report for a given model blueprint.
    """
    def __init__(self, model_artifact, X_train: pd.DataFrame, blueprint):
        self.model_artifact = model_artifact
        self.X_train = X_train
        self.blueprint = blueprint
        self.report_dir = "./xai_reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def _get_explainer(self):
        # Use TreeExplainer for tree-based models, KernelExplainer for others
        if self.blueprint.architecture == "LightGBM":
            # Access the .model attribute inside our LightGBMModel wrapper
            return shap.TreeExplainer(self.model_artifact.model)

        # For TensorFlow/Keras models like our DAE
        elif self.blueprint.architecture == "DenoisingAutoencoder":
            # KernelExplainer needs a prediction function and a background dataset
            # We pass the actual TensorFlow model's predict method
            return shap.KernelExplainer(self.model_artifact.model.predict, shap.sample(self.X_train, 100))

    def _generate_global_explanation(self, trade_log: pd.DataFrame):
        print("--- Generating Global XAI Explanations (SHAP Summary Plot) ---")
        try:
            explainer = self._get_explainer()
            shap_values = explainer.shap_values(self.X_train)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_train, show=False)
            plot_path = os.path.join(self.report_dir, "xai_report_global.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close() # Close the plot to free memory
            print(f"Global explanation plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            print(f"Error generating global SHAP explanation: {e}")
            return None

    def _generate_local_explanation(self, trade_log: pd.DataFrame):
        print("--- Generating Local XAI Explanations (SHAP Force Plots) ---")
        local_explanation_paths = []
        
        print(f"DEBUG: XaiReporter - trade_log received. Is empty: {trade_log.empty}, Columns: {trade_log.columns.tolist()}")

        if trade_log.empty or 'pnl' not in trade_log.columns:
            print("Warning: trade_log is empty or missing 'pnl' column. Skipping local explanations.")
            return []

        try:
            explainer = self._get_explainer()

            # Select a few interesting trades (e.g., biggest wins, biggest losses, recent trades)
            sorted_by_pnl = trade_log.sort_values(by='pnl', ascending=False)
            top_wins = sorted_by_pnl.head(3)
            top_losses = sorted_by_pnl.tail(3)
            recent_trades = trade_log.tail(3)

            selected_trades = pd.concat([top_wins, top_losses, recent_trades]).drop_duplicates()
            print(f"Selected {len(selected_trades)} key trades for local explanation.")

            for i, (idx, trade) in enumerate(selected_trades.iterrows()):
                # Get the corresponding feature vector from X_train (or X_val if applicable)
                # Assuming X_train contains all features used by the model
                if idx in self.X_train.index:
                    instance = self.X_train.loc[[idx]]
                else:
                    # Fallback if index not in X_train, e.g., if trade_log indices are from X_val
                    # For simplicity, we'll just skip or take a random sample if not found
                    print(f"Warning: Trade index {idx} not found in X_train for local explanation. Skipping.")
                    continue

                shap_values_instance = explainer.shap_values(instance)
                
                plt.figure(figsize=(12, 4))
                # Force plot for a single instance
                shap.force_plot(explainer.expected_value, shap_values_instance[0], instance.iloc[0], matplotlib=True, show=False)
                plot_path = os.path.join(self.report_dir, f"xai_report_local_{idx.strftime('%Y%m%d%H%M%S')}_{i}.png")
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                local_explanation_paths.append(plot_path)
                print(f"Local explanation plot for trade {idx} saved to {plot_path}")

        except Exception as e:
            print(f"Error generating local SHAP explanations: {e}")

        return local_explanation_paths

    def generate_full_report(self, trade_log: pd.DataFrame) -> str:
        print("--- Generating Full XAI Justification Report ---")
        report_filename = os.path.join(self.report_dir, "xai_full_report.md")

        global_plot_path = self._generate_global_explanation(trade_log)
        local_plots_paths = self._generate_local_explanation(trade_log)

        with open(report_filename, "w") as f:
            f.write(f"# XAI Justification Report for Model: {self.blueprint.architecture}\n\n")
            f.write(f"## Blueprint Details\n")
            f.write(f"- **Architecture:** {self.blueprint.architecture}\n")
            f.write(f"- **Features Used:** {', '.join(self.blueprint.features)}\n")
            f.write(f"- **Optimized Hyperparameters:** {json.dumps(self.blueprint.hyperparameters, indent=2)}\n")
            f.write(f"- **Fitness (Validation Metric):** {self.blueprint.fitness:.4f}\n\n")

            f.write(f"## Global Explanations\n")
            f.write(f"Overall feature importance and model behavior.\n")
            if global_plot_path:
                f.write(f"![Global Feature Importance]({os.path.basename(global_plot_path)})\n\n")
            else:
                f.write("*(Global explanation plot could not be generated.)*\n\n")

            f.write(f"## Local Explanations (Key Trades)\n")
            f.write(f"Detailed explanations for specific trade decisions.\n")
            if local_plots_paths:
                for lp_path in local_plots_paths:
                    f.write(f"![Local Explanation]({os.path.basename(lp_path)})\n\n")
            else:
                f.write("*(Local explanation plots could not be generated, possibly due to empty trade log or no significant trades.)*\n\n")
            
            f.write(f"## Conclusion\n")
            f.write(f"This report provides insights into how the model {self.blueprint.architecture} makes its predictions, highlighting key features and decision rationales.\n")
        
        print(f"Full XAI report in Markdown format saved to {report_filename}")
        return report_filename
