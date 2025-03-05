from fastapi import FastAPI
import pickle
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
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

        # Convert categorical variables to match the trained model
        df = pd.get_dummies(df)

        # Load the trained model
        with open("final_model.pkl", "rb") as file:
            model = pickle.load(file)

        # Debugging: Print expected vs actual feature names
        model_features = model.feature_names_in_  # Expected feature names
        input_features = df.columns.tolist()  # Incoming feature names
        print("\n🔍 Model Features:", model_features)
        print("\n🔍 Input Features:", input_features)

        # Ensure all missing features are added with value 0
        for feature in model_features:
            if feature not in df.columns:
                df[feature] = 0  # Add missing columns with default value 0

        # Reorder columns to match the model
        df = df[model_features]

        # Make prediction
        prediction = model.predict(df)

        return {"predicted_yield": prediction[0]}

    except Exception as e:
        return {"error": str(e)}

