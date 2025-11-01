import pandas as pd
import os

def download_heart_disease_data():
    """Download the Cleveland heart disease dataset"""
    os.makedirs('data', exist_ok=True)  # Create data directory if it doesn't exist
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"  # Dataset URL
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]  # Column names for the dataset
    
    try:
        print("Downloading dataset...")  # Print download status message
        df = pd.read_csv(url, names=column_names, na_values='?')  # Read CSV from URL with specified column names and missing value marker
        df.to_csv('data/cleveland.data', index=False)  # Save dataset to CSV file without index
        print(f" Dataset downloaded! Shape: {df.shape}")  # Print success message with dataset shape
        return df  # Return the downloaded DataFrame
    except Exception as e:
        print(f" Error: {e}")  # Print error message if download fails
        return None  # Return None if download fails

if __name__ == "__main__":  # Check if script is run directly 
    download_heart_disease_data()  # Call the download function when script is executed directly