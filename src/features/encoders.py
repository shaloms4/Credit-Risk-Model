from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_categorical_transformer(categorical_cols):
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')
