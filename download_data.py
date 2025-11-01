import pandas as pd
import os

def download_heart_disease_data():
    """Download the Cleveland heart disease dataset"""
    os.makedirs('data', exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    
    try:
        print("Downloading dataset...")
        df = pd.read_csv(url, names=column_names, na_values='?')
        df.to_csv('data/cleveland.data', index=False)
        print(f" Dataset downloaded! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f" Error: {e}")
        return None

if __name__ == "_main_":
    download_heart_disease_data()