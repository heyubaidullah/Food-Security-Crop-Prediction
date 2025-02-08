import pandas as pd

df = pd.read_csv("C:/Users/del028/Downloads/Projects/Food-Security-Crop-Prediction/data/yield_df.csv")  # Load the dataset
print("Column Names:", df.columns.tolist())  # Print column names
print(df.head())  # Display first few rows
