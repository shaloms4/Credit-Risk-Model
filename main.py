from fastapi import FastAPI
from src.api.pydantic_models import InputData, Prediction
import mlflow.pyfunc
import pandas as pd

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/credit-risk-model/Production")  # Update as needed

@app.post("/predict", response_model=Prediction)
def predict_risk(data: InputData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict(df)[0]
    return Prediction(risk_probability=prob)
