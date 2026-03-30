from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
threshold = joblib.load("threshold.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    insurance_type: str
    prev_readmit_group: int
    los_group: str
    dc_location: str
    primary_dx_tier: str
    age_bin: int



def preprocess_input(data: PatientData):
    # 1. Convert to dict
    input_dict = data.dict()
    
    # 2. Create a template dataframe with all zeros based on training columns
    df_encoded = pd.DataFrame(0, index=[0], columns=columns)
    
    # 3. Fill in the values
    for key, value in input_dict.items():
        # Handle categorical columns (One-Hot style)
        column_name = f"{key}_{value}"
        if column_name in columns:
            df_encoded.at[0, column_name] = 1
        # Handle numerical columns (like risk_score_bin)
        elif key in columns:
            df_encoded.at[0, key] = value
            
    return df_encoded


@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.post("/predict")
def predict(data: PatientData):
    # Preprocess incoming data
    X = preprocess_input(data)

    # Get probability
    prob = model.predict_proba(X)[0][1]

    # Apply threshold
    risk_flag = int(prob >= threshold)

    return {
        "probability": float(prob),
        "risk_flag": risk_flag
    }
