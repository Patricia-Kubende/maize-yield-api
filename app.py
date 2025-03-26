import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, User, Prediction
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Load trained model once at startup
with open("final_model.pkl", "rb") as file:
    model = pickle.load(file)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 🆕 **Signup Endpoint (NO HASHING)**
@app.post("/signup/")
def signup(username: str, password: str, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="❌ Username already exists. Choose another.")

    # 🔹 Store password directly (not secure, only for debugging!)
    new_user = User(username=username, password=password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "✅ Signup successful! You can now log in."}

# 🆕 **Login Endpoint (NO HASHING)**
@app.post("/login/")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or user.password != password:  # No hashing check
        raise HTTPException(status_code=401, detail="❌ Invalid username or password.")

    return {"message": "✅ Login successful! You can now access predictions."}

# **🔐 Protected Prediction Endpoint**
@app.post("/predict/")
def predict_yield(data: dict, username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="❌ Please log in to access this feature.")

    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)

        # Ensure missing features are added with 0
        for feature in model.feature_names_in_:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match the model
        df = df[model.feature_names_in_]

        # Make prediction
        predicted_yield = model.predict(df)[0]

        # Confidence range (±10%)
        lower_bound = round(predicted_yield * 0.9, 2)
        upper_bound = round(predicted_yield * 1.1, 2)

        # Yield category
        if predicted_yield > 30:
            category = "High Yield"
            recommendation = "✅ Maintain current farming practices."
        elif predicted_yield > 20:
            category = "Moderate Yield"
            recommendation = "⚠️ Consider improving soil quality and irrigation."
        else:
            category = "Low Yield"
            recommendation = "❌ Apply more fertilizer and optimize planting date."

        # Store prediction in database
        db_prediction = Prediction(
            username=username,
            Soil_Type=data["Soil_Type"],
            pH=data["pH"],
            Seed_Variety=data["Seed_Variety"],
            Rainfall_mm=data["Rainfall_mm"],
            Temperature_C=data["Temperature_C"],
            Humidity_percent=data["Humidity_%"],
            Planting_Date=data["Planting_Date"],
            Fertilizer_Type=data["Fertilizer_Type"],
            Predicted_Yield=predicted_yield,
            Confidence_Range=f"{lower_bound} - {upper_bound} bags per acre",
            Category=category,
            Recommendation=recommendation
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return {
            "predicted_yield": round(predicted_yield, 2),
            "confidence_range": f"{lower_bound} - {upper_bound} bags per acre",
            "category": category,
            "recommendation": recommendation,
            "input_summary": data
        }

    except Exception as e:
        return {"error": str(e)}

# **🆕 View Past Predictions API**
@app.get("/predictions/{username}")
def view_past_predictions(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="❌ Please log in to access past predictions.")

    past_predictions = db.query(Prediction).filter(Prediction.username == username).all()

    if not past_predictions:
        return {"message": "⚠️ No past predictions found for this user."}

    return {"past_predictions": past_predictions}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
