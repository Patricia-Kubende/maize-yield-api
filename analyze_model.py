import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
df = pd.read_csv("maize_yield_dataset_2000cleaned.csv")

# Define features (X) and target (y)
target_column = "Yield_Bags_Per_Acre"
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical variables to numeric
X = pd.get_dummies(X)

# Split into training and testing sets (ensure same split as during training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
with open("final_model.pkl", "rb") as file:
    model = pickle.load(file)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\n🔍 Model Evaluation Metrics:")
print(f"📌 Mean Squared Error (MSE): {mse:.3f}")
print(f"📌 Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"📌 Mean Absolute Error (MAE): {mae:.3f}")
print(f"📌 R² Score: {r2:.3f}")
