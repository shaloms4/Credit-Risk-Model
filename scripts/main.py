# main.py

import pandas as pd
from pathlib import Path
from src.features.feature_pipeline import preprocess_data

def main():
    # Load raw data
    data_path = Path("data/raw/data.csv")
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    # Optional: Check raw shape
    print(f"Raw data shape: {df.shape}")

    # Process features
    print("Running feature pipeline...")
    X = preprocess_data(df)

    # Show result
    print(f"Feature matrix shape: {X.shape}")

    # Optional: Save processed features (as NumPy array or sparse matrix)
    # from scipy.sparse import save_npz
    # save_npz("data/processed_features.npz", X)

if __name__ == "__main__":
    main()
