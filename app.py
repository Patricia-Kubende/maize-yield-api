import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Load the trained ML model
with open("final_model.pkl", "rb") as file:
    model = pickle.load(file)

# Pydantic model for incoming prediction data
class PredictionInput(BaseModel):
    Soil_Type: str
    pH: float
    Seed_Variety: str
    Rainfall_mm: float
    Temperature_C: float
    Humidity_percent: float
    Planting_Date: str
    Fertilizer_Type: str

@app.get("/")
def root():
    return {"message": "✅ Maize Yield Prediction API is running!"}

@app.post("/predict/")
def predict_yield(data: PredictionInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])
        df = pd.get_dummies(df)

        # Align features with model input
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]

        # Predict
        predicted_yield = model.predict(df)[0]
        lower = round(predicted_yield * 0.9, 2)
        upper = round(predicted_yield * 1.1, 2)

        # Determine recommendation
        category = "High Yield" if predicted_yield > 30 else "Moderate Yield" if predicted_yield > 20 else "Low Yield"
        recommendation = (
            "✅ Maintain current practices." if category == "High Yield" else
            "⚠️ Improve soil or irrigation." if category == "Moderate Yield" else
            "❌ Use more fertilizer, adjust planting date."
        )

        return {
            "predicted_yield": round(predicted_yield, 2),
            "confidence_range": f"{lower} - {upper} bags per acre",
            "category": category,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
