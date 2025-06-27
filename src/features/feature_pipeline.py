# src/features/feature_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.features.aggregate_features import add_aggregate_features
from src.features.time_features import add_time_features

def preprocess_data(df):
    # Step 1: Feature Extraction
    df = add_time_features(df)
    df = add_aggregate_features(df)

    # Step 2: Define Columns
    categorical_cols = ['ChannelId', 'ProductCategory', 'CurrencyCode']
    numeric_cols = ['Amount', 'Value', 'TotalTransactionAmount', 
                    'AvgTransactionAmount', 'TransactionCount', 
                    'StdTransactionAmount', 'TransactionHour', 
                    'TransactionDay', 'TransactionMonth']

    # Step 3: Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),

        ("cat", Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols),
    ])

    # Step 4: Fit-Transform
    X = preprocessor.fit_transform(df)

    return X
