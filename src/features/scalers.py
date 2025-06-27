from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_scaler_pipeline():
    return Pipeline([
        ("scaler", StandardScaler())
    ])
