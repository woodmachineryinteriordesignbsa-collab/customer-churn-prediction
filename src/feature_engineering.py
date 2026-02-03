import pandas as pd

# Load cleaned data
df = pd.read_csv("../data/cleaned_churn.csv")

# -----------------------------
# Separate categorical columns
# -----------------------------
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()


# Remove target column if present
if "Churn" in categorical_cols:
    categorical_cols.remove("Churn")

# -----------------------------
# Encode target variable
# -----------------------------
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# One-Hot Encode all categorical features
# -----------------------------
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------
# Save encoded data
# -----------------------------
df.to_csv("data/encoded_churn.csv", index=False)

print("Feature engineering completed successfully.")
print("Encoded data saved as data/encoded_churn.csv")
