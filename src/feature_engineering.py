import pandas as pd

# Load cleaned data
df = pd.read_csv("../data/cleaned_churn.csv")

# Encode target
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Select categorical columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

# Remove target if present
if "Churn" in categorical_cols:
    categorical_cols.remove("Churn")

# One-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save encoded data
df.to_csv("../data/encoded_churn.csv", index=False)

print("Feature engineering completed successfully.")
print("Encoded data saved to encoded_churn.csv")
