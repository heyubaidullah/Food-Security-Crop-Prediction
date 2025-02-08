import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/yield_df.csv")

# Define the correct target column
target_column = "hg/ha_yield"

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "Area", "Item"])  # Remove non-numeric columns

# Ensure the target column exists
if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")

# Define features (X) and target (y)
X = df.drop(columns=[target_column])  
y = df[target_column]  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in the Streamlit app
joblib.dump(scaler, "scaler.pkl")

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}.pkl")  # Save each model

print("✅ Models trained and saved successfully!")
