from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the trained model
model = joblib.load('stress_model.joblib')

# Define the input data model (update fields as per your dataset)
class StressInput(BaseModel):
    Gender: int
    Age: int
    # Add all other features here, matching the order used in training
    # Example: Feature1: int, Feature2: int, ...
    # For demonstration, only Gender and Age are included

@app.post('/predict')
def predict_stress(data: StressInput):
    # Convert input to numpy array and reshape
    input_data = np.array(list(data.dict().values())).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return {'predicted_stress_type': int(prediction)}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
