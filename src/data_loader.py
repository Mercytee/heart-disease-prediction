import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
import os
from .base_classes import BaseDataLoader

class DataLoader(BaseDataLoader):
    """
    Data loader class for heart disease dataset with enhanced OOP
    Implements BaseDataLoader interface with property decorators
    """
    
    def _init_(self, file_path: str):
        self._file_path = file_path
        self._column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        self.logger = self._setup_logger()
        self._data = None
        
    @property
    def file_path(self) -> str:
        """Getter for file_path with validation"""
        return self._file_path
    
    @file_path.setter
    def file_path(self, value: str):
        """Setter for file_path with validation"""
        if not isinstance(value, str):
            raise ValueError("File path must be a string")
        if not value.endswith('.csv'):
            raise ValueError("File must be a CSV file")
        self._file_path = value
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Getter for data with access control"""
        return self._data
    
    @property
    def column_names(self) -> list:
        """Getter for column names"""
        return self._column_names.copy()  # Return copy to prevent modification
    
    @property
    def is_data_loaded(self) -> bool:
        """Property to check if data is loaded"""
        return self._data is not None and not self._data.empty
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def load_data(self) -> pd.DataFrame:
        """Implementation of abstract method from BaseDataLoader"""
        try:
            self.logger.info(f"Loading data from {self.file_path}")
            
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Data file not found at {self.file_path}")
            
            # Load data with proper configuration and convert to numeric
            df = pd.read_csv(
                self.file_path, 
                names=self.column_names, 
                na_values='?',
                skipinitialspace=True
            )
            
            # Convert all columns to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            self._data = df
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
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
        """Implementation of abstract method from BaseDataLoader"""
        if df is None:
            if self._data is None:
                self.logger.error("No data to validate")
                return False
            df = self._data
            
        required_columns = set(self.column_names)
        actual_columns = set(df.columns)
        
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            self.logger.error(f"Missing required columns: {missing}")
            return False
            
        if df.empty:
            self.logger.error("Dataset is empty")
            return False
            
        self.logger.info("Data validation passed")
        return True
    
    def get_data_info(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Implementation of abstract method from BaseDataLoader"""
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
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {}
        }
        
        return info

# Enhanced inheritance hierarchy
class MedicalDataLoader(DataLoader):
    """Specialized data loader for medical datasets"""
    
    def __init__(self, file_path: str, dataset_type: str = "heart_disease"):
        super().__init__(file_path)
        self.dataset_type = dataset_type
        self._patient_ids = None
    
    @property
    def patient_ids(self) -> Optional[pd.Series]:
        """Get patient IDs if available"""
        return self._patient_ids
    
    def load_data(self) -> pd.DataFrame:
        """Override with medical-specific loading"""
        df = super().load_data()
        # Medical-specific validation
        self._validate_medical_data(df)
        return df
    
    def _validate_medical_data(self, df: pd.DataFrame):
        """Medical data specific validation"""
        # Check for reasonable medical ranges
        if 'age' in df.columns:
            if (df['age'] < 0).any() or (df['age'] > 120).any():
                self.logger.warning("Age values outside reasonable range")
        
        if 'trestbps' in df.columns:  # blood pressure
            if (df['trestbps'] < 50).any() or (df['trestbps'] > 250).any():
                self.logger.warning("Blood pressure values outside reasonable range")