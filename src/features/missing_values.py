from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def get_missing_value_pipeline(numeric_cols):
    return Pipeline([
        ("imputer", SimpleImputer(strategy='median'))
    ])
