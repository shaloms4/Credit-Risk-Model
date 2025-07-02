import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

params = {
    "LogisticRegression": {"C": [0.01, 0.1, 1]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [3, 5]},
    "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
}

mlflow.set_experiment("credit-risk-model")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        clf = GridSearchCV(model, params[name], cv=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        # Log metrics
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("roc_auc", auc)

        # Save best model
        mlflow.sklearn.log_model(clf.best_estimator_, "model")
        mlflow.set_tag("model", name)
