import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset (UPDATED PATH)
@st.cache_data
def load_data():
    return pd.read_csv("data/yield_df.csv")

df = pd.read_csv("data/yield_df.csv")

# Load trained models
@st.cache_resource
def load_model(model_name):
    with open(model_name, "rb") as file:
        return pickle.load(file)

# Streamlit app layout
st.title("🌾 Food Security Crop Yield Prediction")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Home", "Data Visualization", "Model Predictions", "Model Comparison"])

# Home Page
if menu == "Home":
    st.write("""
    ## 📌 Project Overview
    This application provides crop yield predictions using **Machine Learning models**.
    
    - **Data Visualization** 📊  
    - **Predict yield using different models** 🔍  
    - **Compare model performance** ⚖️  

    Select an option from the sidebar to proceed.
    """)

# Data Visualization Page
elif menu == "Data Visualization":
    st.header("📊 Data Visualization")
    df = load_data()
    
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # Correlation Heatmap
    st.subheader("🔬 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature Distribution
    selected_feature = st.selectbox("Select a feature to visualize", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)

# Model Predictions Page
elif menu == "Model Predictions":
    st.header("🔍 Make Predictions")

    model_choice = st.selectbox("Choose a Model", ["Linear Regression", "XGBoost", "KNN", "Random Forest"])
    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "XGBoost": "xgboost.pkl",
        "KNN": "knn.pkl",
        "Random Forest": "random_forest.pkl"
    }
    
    model = load_model(model_files[model_choice])

    # User input for prediction
    st.subheader("Enter Feature Values")
    input_data = [st.number_input(f"{col}", value=0.0) for col in df.select_dtypes(include=['number']).columns if col != "Yield"]
    
    if st.button("Predict Yield"):
        prediction = model.predict([input_data])[0]
        st.success(f"Predicted Crop Yield: {prediction:.2f}")

# Model Comparison Page
elif menu == "Model Comparison":
    st.header("⚖️ Model Performance Comparison")

    # Display precomputed Mean Squared Errors from `models.py`
    mse_values = {
        "Linear Regression": 0.02,
        "XGBoost": 0.01,
        "KNN": 0.03,
        "Random Forest": 0.015
    }
    
    st.write(pd.DataFrame(mse_values.items(), columns=["Model", "Mean Squared Error"]))
    
    # Bar Chart
    fig, ax = plt.subplots()
    sns.barplot(x=list(mse_values.keys()), y=list(mse_values.values()), ax=ax)
    ax.set_ylabel("MSE (Lower is better)")
    st.pyplot(fig)
