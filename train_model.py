import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load cleaned dataset
df = pd.read_csv("maize_yield_dataset_2000cleaned.csv")

# Define features (X) and target (y)
target_column = "Yield_Bags_Per_Acre" 
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

# Convert categorical variables to numeric
X = pd.get_dummies(X)  # One-hot encoding for categorical variables

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("final_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model successfully trained and saved as 'final_model.pkl'")
