import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging

class DataPreprocessor:
<<<<<<< HEAD
    """
    Data preprocessing class for heart disease dataset with OOP design
    Handles missing values, encoding, scaling, and data splitting
    """
    
    def _init_(self):
=======
    def __init__(self):
>>>>>>> 563c5b8 (Complete working heart disease prediction system with 83.6% accuracy)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.logger = self._setup_logger()
        self.is_fitted = False
        self.feature_names = []
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'num') -> Tuple[pd.DataFrame, pd.Series]:
        self.logger.info("Starting data preprocessing pipeline")
        data = df.copy()
        
        self.logger.info("Converting target variable to binary classification")
<<<<<<< HEAD
        
        # Convert target column to numeric first
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        
        # Handle any NaN values that resulted from conversion
        data[target_col].fillna(0, inplace=True)
        
        # Now convert to binary classification
        data[target_col] = data[target_col].apply(lambda x: 1 if x > 0 else 0)
=======
        # FIX: Ensure proper numeric conversion
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        data[target_col].fillna(0, inplace=True)
        data[target_col] = data[target_col].apply(lambda x: 1 if float(x) > 0 else 0)  # FIX: float conversion
>>>>>>> 563c5b8 (Complete working heart disease prediction system with 83.6% accuracy)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        self.feature_names = X.columns.tolist()
        
        X = self._handle_missing_values(X)
        X = self._encode_categorical_features(X)
        X = self._scale_features(X)
        
        self.is_fitted = True
        self.logger.info("Data preprocessing completed successfully")
        return X, y
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Handling missing values")
        missing_count = X.isnull().sum().sum()
        if missing_count == 0:
            self.logger.info("No missing values found")
            return X
        
        self.logger.info(f"Found {missing_count} missing values")
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if not numerical_cols.empty:
            X_numerical = X[numerical_cols]
            if X_numerical.isnull().any().any():
                self.logger.info(f"Imputing numerical columns: {list(numerical_cols)}")
                X[numerical_cols] = self.imputer.fit_transform(X_numerical)
        
        for col in categorical_cols:
            if X[col].isnull().any():
                self.logger.info(f"Imputing categorical column: {col}")
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col].fillna(mode_val, inplace=True)
        
        self.logger.info("Missing value handling completed")
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Encoding categorical features")
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            self.logger.info("No categorical features to encode")
            return X
        
        self.logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            self.logger.info(f"Encoded {col} with {len(self.label_encoders[col].classes_)} classes")
        
        return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Scaling numerical features")
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.logger.info("No numerical features to scale")
            return X
        
        self.logger.info(f"Scaling numerical columns: {list(numerical_cols)}")
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        return X
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        self.logger.info(f"Splitting data with test size: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            self.logger.warning("Preprocessor not fitted yet")
            return {}
        
        info = {
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'scaler_fitted': hasattr(self.scaler, 'mean_'),
            'imputer_fitted': hasattr(self.imputer, 'statistics_'),
            'label_encoders': list(self.label_encoders.keys()),
            'numerical_columns_count': len([col for col in self.feature_names if col not in self.label_encoders]),
            'categorical_columns_count': len(self.label_encoders)
        }
        return info
