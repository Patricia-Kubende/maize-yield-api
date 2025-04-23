import os
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, User, Prediction
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Add this root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Maize Yield Prediction API!"}

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

# Pydantic models for request validation
class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

# **Signup Endpoint**
@app.post("/signup/")
def signup(user: SignupRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="❌ Username already exists. Choose another.")

    new_user = User(username=user.username, password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "✅ Signup successful! You can now log in."}

# **Login Endpoint**
@app.post("/login/")
def login(user: LoginRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if not existing_user or existing_user.password != user.password:
        raise HTTPException(status_code=401, detail="❌ Invalid username or password.")

    return {"message": "✅ Login successful! You can now access predictions."}

# **Prediction Endpoint**
@app.post("/predict/")
def predict_yield(data: dict, username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="❌ Please log in to access this feature.")

    try:
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        for feature in model.feature_names_in_:
            if feature not in df.columns:
                df[feature] = 0
        df = df[model.feature_names_in_]

        predicted_yield = model.predict(df)[0]
        lower_bound = round(predicted_yield * 0.9, 2)
        upper_bound = round(predicted_yield * 1.1, 2)

        category = "High Yield" if predicted_yield > 30 else "Moderate Yield" if predicted_yield > 20 else "Low Yield"
        recommendation = (
            "✅ Maintain current farming practices." if category == "High Yield" else
            "⚠️ Consider improving soil quality and irrigation." if category == "Moderate Yield" else
            "❌ Apply more fertilizer and optimize planting date."
        )

        db_prediction = Prediction(
            username=username,
            Soil_Type=data.get("Soil_Type"),
            pH=data.get("pH"),
            Seed_Variety=data.get("Seed_Variety"),
            Rainfall_mm=data.get("Rainfall_mm"),
            Temperature_C=data.get("Temperature_C"),
            Humidity_percent=data.get("Humidity_%"),
            Planting_Date=data.get("Planting_Date"),
            Fertilizer_Type=data.get("Fertilizer_Type"),
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

# **View Past Predictions**
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
