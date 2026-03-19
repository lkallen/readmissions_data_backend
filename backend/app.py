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
    risk_score_bin: int
    dc_location: str
    primary_dx_tier: str
    age_bin: int



def preprocess_input(data: PatientData):
    # Convert incoming data to a DataFrame
    df = pd.DataFrame([data.dict()])

    # Dummy encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Reindex to match training columns
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

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
