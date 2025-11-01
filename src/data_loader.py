import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
import os

class DataLoader:
    def __init__(self, file_path: str):
        self._file_path = file_path  # Private variable for file path
        self._column_names = [  # Define column names for the dataset
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        self.logger = self._setup_logger()  # Initialize logger
        self._data = None  # Private variable to store loaded data
        
    @property
    def file_path(self) -> str:
        """Getter for file_path with validation"""
        return self._file_path  # Return file path
    
    @file_path.setter
    def file_path(self, value: str):
        """Setter for file_path with validation"""
        if not isinstance(value, str):  # Validate input type
            raise ValueError("File path must be a string")
        self._file_path = value  # Set file path
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Getter for data with access control"""
        return self._data  # Return loaded data
    
    @property
    def column_names(self) -> list:
        """Getter for column names"""
        return self._column_names.copy()  # Return copy to prevent modification
    
    @property
    def is_data_loaded(self) -> bool:
        """Property to check if data is loaded"""
        return self._data is not None and not self._data.empty  # Check if data exists and is not empty
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)  # Create logger instance
        if not logger.handlers:  # Check if handlers already exist
            handler = logging.StreamHandler()  # Create console handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Log format
            handler.setFormatter(formatter)  # Apply format to handler
            logger.addHandler(handler)  # Add handler to logger
            logger.setLevel(logging.INFO)  # Set logging level
        return logger  # Return configured logger
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        try:
            self.logger.info(f"Loading data from {self.file_path}")  # Log loading attempt
            
            if not os.path.exists(self.file_path):  # Check if file exists
                raise FileNotFoundError(f"Data file not found at {self.file_path}")  # Raise error if not found
            
            # Load data with proper configuration
            df = pd.read_csv(  # Read CSV file
                self.file_path, 
                names=self.column_names,  # Use predefined column names
                na_values='?',  # Treat '?' as missing values
                skipinitialspace=True  # Skip initial whitespace
            )
            
            # Convert all columns to numeric where possible
            for col in df.columns:  # Iterate through all columns
                df[col] = pd.to_numeric(df[col], errors='ignore')  # Convert to numeric, ignore errors
            
            self._data = df  # Store loaded data
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")  # Log success
            return df  # Return loaded DataFrame
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")  # Log error
            raise  # Re-raise exception
    
    def get_data_info(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Get comprehensive information about the dataset"""
        if df is None:  # Check if DataFrame is provided
            if self._data is None:  # Check if data is loaded
                self.logger.warning("No data loaded. Call load_data() first.")  # Log warning
                return {}  # Return empty dictionary
            df = self._data  # Use stored data
            
        info = {  # Create information dictionary
            'shape': df.shape,  # Dataset dimensions
            'columns': list(df.columns),  # Column names
            'data_types': df.dtypes.to_dict(),  # Data types for each column
            'missing_values': df.isnull().sum().to_dict(),  # Count of missing values per column
            'description': df.describe().to_dict(),  # Statistical description
            'target_distribution': df['num'].value_counts().to_dict() if 'num' in df.columns else {}  # Target variable distribution
        }
        
        self.logger.info(f"Dataset info: {df.shape[0]} samples, {df.shape[1]} features")  # Log dataset info
        return info  # Return information dictionary
    
    def validate_data(self, df: pd.DataFrame = None) -> bool:
        """Validate data structure"""
        if df is None:  # Check if DataFrame is provided
            if self._data is None:  # Check if data is loaded
                self.logger.error("No data to validate")  # Log error
                return False  # Return validation failure
            df = self._data  # Use stored data
            
        required_columns = set(self.column_names)  # Convert to set for comparison
        actual_columns = set(df.columns)  # Get actual column names as set
        
        if not required_columns.issubset(actual_columns):  # Check if all required columns exist
            missing = required_columns - actual_columns  # Find missing columns
            self.logger.error(f"Missing required columns: {missing}")  # Log missing columns
            return False  # Return validation failure
            
        if df.empty:  # Check if DataFrame is empty
            self.logger.error("Dataset is empty")  # Log error
            return False  # Return validation failure
            
        self.logger.info("Data validation passed")  # Log validation success
        return True  # Return validation success


class MedicalDataLoader(DataLoader):
    """Specialized data loader for medical datasets"""
    
    def __init__(self, file_path: str, dataset_type: str = "heart_disease"):
        super().__init__(file_path)  # Initialize parent class
        self.dataset_type = dataset_type  # Store dataset type
        self._patient_ids = None  # Initialize patient IDs
    
    @property
    def patient_ids(self) -> Optional[pd.Series]:
        """Get patient IDs if available"""
        return self._patient_ids  # Return patient IDs
    
    def load_data(self) -> pd.DataFrame:
        """Override with medical-specific loading"""
        df = super().load_data()  # Call parent load_data method
        # Medical-specific validation
        self._validate_medical_data(df)  # Perform medical validation
        return df  # Return loaded data
    
    def _validate_medical_data(self, df: pd.DataFrame):
        """Medical data specific validation"""
        # Check for reasonable medical ranges
        if 'age' in df.columns:  # Check if age column exists
            if (df['age'] < 0).any() or (df['age'] > 120).any():  # Check age range validity
                self.logger.warning("Age values outside reasonable range")  # Log warning
        
        if 'trestbps' in df.columns:  # Check if blood pressure column exists
            if (df['trestbps'] < 50).any() or (df['trestbps'] > 250).any():  # Check blood pressure range
                self.logger.warning("Blood pressure values outside reasonable range")  # Log warning