import pandas as pd
import os

# Load raw data
df = pd.read_csv(r"C:\Users\lenovo\Documents\customer_churn_prediction\data\data.csv")

# Convert TotalCharges to numeric (dirty numeric column)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Quick numeric summary
print(df.describe())

#folder exists
os.makedirs("../data/data", exist_ok=True)

# Save cleaned data
df.to_csv("data/cleaned_churn.csv", index=False)
print("Data cleaning done. Saved to cleaned_churn.csv")

