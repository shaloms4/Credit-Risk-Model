import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from pathlib import Path
from src.features.feature_pipeline import preprocess_data

from src.target.rfm_target import calculate_rfm, cluster_rfm, assign_high_risk

def main():
    # Load raw data
    df = pd.read_csv("./data/raw/data.csv")
    print(f"Raw data shape: {df.shape}")

    # Feature Engineering (Task 3)
    from src.features.feature_pipeline import preprocess_data
    X = preprocess_data(df)
    print(f"Feature matrix shape: {X.shape}")

    # RFM Target Variable Engineering (Task 4)
    rfm_df = calculate_rfm(df)
    clustered_rfm = cluster_rfm(rfm_df)
    risk_labels = assign_high_risk(clustered_rfm)

    # Merge high-risk label back to original df
    df = df.merge(risk_labels, on="CustomerId", how="left")
    df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)

    print("\nSample with is_high_risk target column:")
    print(df[['CustomerId', 'is_high_risk']].drop_duplicates().head())

if __name__ == "__main__":
    main()

