import pandas as pd

# Load your cleaned dataset
df = pd.read_csv("maize_yield_dataset_2000cleaned.csv")  

# Display the first few rows
print(df.head())

# Show column names and data types
print("\nColumn Names and Data Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
