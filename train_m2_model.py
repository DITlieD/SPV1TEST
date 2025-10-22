# train_m2_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing_v2 import get_market_data
from models_v2 import SimpleConfluenceModel

def create_meta_labels(df, m1_model, horizon=5):
    """
    Generates labels for the M2 meta-model.
    A label is 1 if the M1 signal was 'correct', 0 otherwise.
    """
    # Get M1 predictions
    m1_predictions = m1_model.predict(df)

    # Determine the outcome
    future_returns = df['close'].pct_change(periods=horizon).shift(-horizon)
    
    # Create meta-labels
    meta_labels = []
    for i in range(len(df)):
        signal = m1_predictions[i]
        outcome = future_returns.iloc[i]
        
        if signal == 1 and outcome > 0:
            meta_labels.append(1) # Correct buy
        elif signal == 2 and outcome < 0:
            meta_labels.append(1) # Correct sell
        else:
            meta_labels.append(0) # Incorrect or hold signal
            
    return pd.Series(meta_labels, index=df.index)

from config import ASSET_UNIVERSE

def train_m2_model():
    """
    Trains a single, global M2 meta-model by aggregating performance data 
    from all M1 models in the asset universe.
    """
    M2_MODEL_PATH = 'models/M2_Meta.joblib'
    TIMEFRAME = '4h'

    print("\n[Genesis] --- Genesis M2 Meta-Model Trainer (Global) ---")

    all_meta_features = []
    all_meta_labels = []

    for asset in ASSET_UNIVERSE:
        print(f"\n[Genesis] Processing M1 performance for {asset}...")
        m1_model_path = f'models/{asset.replace("/", "_")}_{TIMEFRAME}_unified.joblib'

        if not os.path.exists(m1_model_path):
            print(f"  -> WARNING: M1 model for {asset} not found. Skipping.")
            continue
        
        m1_model = SimpleConfluenceModel.load(m1_model_path)
        
        print(f"  -> Fetching data for {asset}...")
        df = get_market_data(asset, TIMEFRAME, limit=5000)
        if df.empty:
            print(f"  -> WARNING: No data for {asset}. Skipping.")
            continue
            
        df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        df.dropna(inplace=True)

        print("  -> Generating meta-labels...")
        meta_labels = create_meta_labels(df, m1_model)
        
        print("  -> Generating meta-features...")
        m1_probabilities = m1_model.predict_proba(df)
        df['m1_prob_0'] = m1_probabilities[:, 0]
        df['m1_prob_1'] = m1_probabilities[:, 1]
        df['m1_prob_2'] = m1_probabilities[:, 2]

        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])
        
        all_meta_features.append(X)
        all_meta_labels.append(meta_labels)

    if not all_meta_features:
        print("FATAL: No M1 models found to train the M2 model. Halting.")
        return

    # Combine data from all assets
    print("\n[Genesis] Aggregating data from all assets...")
    X_combined = pd.concat(all_meta_features)
    y_combined = pd.concat(all_meta_labels)
    
    X_combined['label'] = y_combined
    X_combined.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined.drop(columns=['label']), 
        X_combined['label'], 
        test_size=0.2, 
        random_state=42, 
        shuffle=False
    )

    print(f"[Genesis] Training Global M2 Meta-Model on {len(X_train)} total samples...")
    df_train = X_train.copy()
    df_train['label'] = y_train
    
    m2_model = SimpleConfluenceModel(model_type='lightgbm', objective='binary', metric='auc')
    m2_model.fit(df_train)

    accuracy = m2_model.model.score(X_test, y_test)
    print(f"[Genesis] Global M2 model accuracy on test set: {accuracy:.2f}")

    print(f"[Genesis] Saving Global M2 meta-model to: {M2_MODEL_PATH}")
    m2_model.save(M2_MODEL_PATH)
    print("[Genesis] --- M2 Training Complete ---")

if __name__ == "__main__":
    train_m2_model()
