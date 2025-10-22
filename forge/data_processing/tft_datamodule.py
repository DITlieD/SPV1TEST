# forge/data_processing/tft_datamodule.py
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

def create_tft_dataset(data: pd.DataFrame, max_encoder_length=30, max_prediction_length=5):
    """
    Creates a TimeSeriesDataSet for the Temporal Fusion Transformer.
    """
    # Add a time index
    data["time_idx"] = range(len(data))
    
    # Add a dummy group column
    data["group"] = 0
    
    # Define the dataset
    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="close",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[col for col in data.columns if col not in ["time_idx", "group", "close"]],
    )
    
    return dataset
