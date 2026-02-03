# Customer Churn Prediction (Machine Learning Project)

## Project Overview
This project predicts whether a customer will churn based on demographic,
contract, and usage information using Machine Learning.

## Dataset
The dataset contains customer information such as:
- Age
- Gender
- Tenure
- Monthly Charges
- Contract Type
- Payment Method
- Internet Usage
- Churn (Target Variable)

## Steps Performed
- Data cleaning (missing values, duplicates, type conversion)
- Feature engineering (encoding categorical variables)
- Feature scaling
- Model training using Logistic Regression
- Model evaluation using accuracy and classification report

## Results
The trained model achieved ~72% accuracy on test data.

## Tools & Libraries
- Python
- Pandas
- NumPy
- scikit-learn

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run data cleaning:
   python src/data_cleaning.py
3. Run feature engineering:
   python src/feature_engineering.py
4. Train model:
   python src/model_training.py
