import os
from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Load the trained model once at startup
with open("final_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "Welcome to the Maize Yield Prediction API!"}

@app.post("/predict/")
def predict_yield(data: dict):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)  # Ensure categorical features are handled

        # Ensure all missing features are added with value 0
        for feature in model.feature_names_in_:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match the model
        df = df[model.feature_names_in_]

        # Make prediction
        predicted_yield = model.predict(df)[0]

        # 🔍 Confidence range (±10%)
        lower_bound = round(predicted_yield * 0.9, 2)
        upper_bound = round(predicted_yield * 1.1, 2)

        # 🔍 Yield category
        if predicted_yield > 30:
            category = "High Yield"
            recommendation = "✅ Optimal conditions. Maintain current farming practices."
        elif predicted_yield > 20:
            category = "Moderate Yield"
            recommendation = "⚠️ Consider improving soil quality and irrigation."
        else:
            category = "Low Yield"
            recommendation = "❌ Apply more fertilizer, optimize planting date."

        return {
            "predicted_yield": round(predicted_yield, 2),
            "confidence_range": f"{lower_bound} - {upper_bound} bags per acre",
            "category": category,
            "recommendation": recommendation,
            "input_summary": data
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Read the PORT from environment variables (default to 8000 if not set)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
