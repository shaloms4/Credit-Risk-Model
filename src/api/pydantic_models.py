from pydantic import BaseModel

class InputData(BaseModel):
    age: int
    income: float
    employment_length: int
    # Add all features used in model

class Prediction(BaseModel):
    risk_probability: float
