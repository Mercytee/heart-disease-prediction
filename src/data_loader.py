import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
import os

class DataLoader:
    """
    Data loader class for heart disease dataset with OOP principles
    Handles data loading, validation, and basic information extraction
    """
    
    def _init_(self, file_path: str):
        self.file_path = file_path
        self.column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        self.logger = self._setup_logger()
        self._data = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data loading operations"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def load_data(self) -> pd.DataFrame:
        """
        Load Cleveland heart disease data with proper error handling
        
        Returns:
            pd.DataFrame: Loaded heart disease data
        """
        try:
            self.logger.info(f"Loading data from {self.file_path}")
            
            # Check if file exists
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Data file not found at {self.file_path}")
            
            # Load data with proper configuration and convert to numeric
            df = pd.read_csv(
                self.file_path, 
                names=self.column_names, 
                na_values='?',
                skipinitialspace=True,
                encoding='utf-8'
            )
            
            # Convert all columns to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            self._data = df
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("Data file is empty")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    
    def get_data_info(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: DataFrame to analyze. If None, uses loaded data
            
        Returns:
            dict: Comprehensive dataset information
        """
        if df is None:
            if self._data is None:
                self.logger.warning("No data loaded. Call load_data() first.")
                return {}
            df = self._data
            
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'description': df.describe().to_dict(),
            'memory_usage': df.memory_usage(deep=True).to_dict(),
            'target_distribution': df['num'].value_counts().to_dict() if 'num' in df.columns else {}
        }
        
        self.logger.info(f"Dataset info: {df.shape[0]} samples, {df.shape[1]} features")
        return info
    
    def validate_data(self, df: pd.DataFrame = None) -> bool:
        """
        Validate the loaded dataset for required columns and basic integrity
        
        Args:
            df: DataFrame to validate. If None, uses loaded data
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if df is None:
            if self._data is None:
                self.logger.error("No data to validate")
                return False
            df = self._data
            
        required_columns = set(self.column_names)
        actual_columns = set(df.columns)
        
        # Check required columns
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            self.logger.error(f"Missing required columns: {missing}")
            return False
            
        # Check if dataset is empty
        if df.empty:
            self.logger.error("Dataset is empty")
            return False
            
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percentage[missing_percentage > 50]
        if not high_missing.empty:
            self.logger.warning(f"Columns with high missing values: {high_missing.to_dict()}")
            
        self.logger.info("Data validation passed")
        return True
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get medical descriptions for each feature
        
        Returns:
            dict: Feature descriptions
        """
        descriptions = {
            "age": "Age in years",
            "sex": "Sex (1 = male; 0 = female)",
            "cp": "Chest pain type (1-4): 1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
            "restecg": "Resting electrocardiographic results (0,1,2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes; 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of the peak exercise ST segment (1,2,3)",
            "ca": "Number of major vessels (0-3) colored by fluoroscopy",
            "thal": "Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)",
            "num": "Diagnosis of heart disease (0 = no; 1-4 = yes)"
        }
        return descriptions
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Get sample data from the dataset
        
        Args:
            n: Number of samples to return
            
        Returns:
            pd.DataFrame: Sample data
        """
        if self._data is None:
            self.logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        
        return self._data.head(n)