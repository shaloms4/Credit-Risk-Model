# src/features/aggregate_features.py

import pandas as pd

def add_aggregate_features(df):
    agg = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std']
    }).reset_index()
    
    agg.columns = ['CustomerId', 
                   'TotalTransactionAmount', 
                   'AvgTransactionAmount', 
                   'TransactionCount', 
                   'StdTransactionAmount']
    
    df = df.merge(agg, on='CustomerId', how='left')
    return df
