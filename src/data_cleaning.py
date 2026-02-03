import pandas as pd

# Load raw data
df = pd.read_csv("../data/data.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Summary
print(df.describe())

# Save cleaned data
df.to_csv("../data/cleaned_churn.csv", index=False)

print("Data cleaning done. Saved to cleaned_churn.csv")

