import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List  # ADDED List import
import logging
from .base_classes import BasePreprocessor

<<<<<<< HEAD
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
=======
class DataPreprocessor(BasePreprocessor):
    """
    Enhanced data preprocessor with property decorators and inheritance
    """
    
    def __init__(self):
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy='median')
        self._label_encoders = {}
>>>>>>> 719e0e1 (feat: implement advanced OOP concepts)
        self.logger = self._setup_logger()
        self._is_fitted = False
        self._feature_names = []
        
    # Property decorators
    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor is fitted"""
        return self._is_fitted
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names.copy()
    
    @property
    def scaler(self) -> StandardScaler:
        """Get scaler instance"""
        return self._scaler
    
    @property
    def numerical_columns_count(self) -> int:
        """Get count of numerical columns"""
        if not self._is_fitted:
            return 0
        return len([col for col in self._feature_names if col not in self._label_encoders])
    
    @property
    def categorical_columns_count(self) -> int:
        """Get count of categorical columns"""
        return len(self._label_encoders)
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """Implementation of abstract method from BasePreprocessor"""
        self.logger.info("Starting data preprocessing pipeline")
        
        data = df.copy()
        
<<<<<<< HEAD
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
=======
        # Handle target variable
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        data[target_col].fillna(0, inplace=True)
        data[target_col] = data[target_col].apply(lambda x: 1 if x > 0 else 0)
>>>>>>> 719e0e1 (feat: implement advanced OOP concepts)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        self._feature_names = X.columns.tolist()
        
        X = self._handle_missing_values(X)
        X = self._encode_categorical_features(X)
        X = self._scale_features(X)
        
        self._is_fitted = True
        self.logger.info("Data preprocessing completed successfully")
        
        return X, y
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values")
        
        # Check for missing values
        missing_count = X.isnull().sum().sum()
        if missing_count == 0:
            self.logger.info("No missing values found")
            return X
        
        self.logger.info(f"Found {missing_count} missing values")
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Impute numerical columns
        if not numerical_cols.empty:
            X_numerical = X[numerical_cols]
            if X_numerical.isnull().any().any():
                self.logger.info(f"Imputing numerical columns: {list(numerical_cols)}")
                X[numerical_cols] = self._imputer.fit_transform(X_numerical)
        
        # For categorical columns, use mode imputation
        for col in categorical_cols:
            if X[col].isnull().any():
                self.logger.info(f"Imputing categorical column: {col}")
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col].fillna(mode_val, inplace=True)
        
        self.logger.info("Missing value handling completed")
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding"""
        self.logger.info("Encoding categorical features")
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            self.logger.info("No categorical features to encode")
            return X
        
        self.logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            self._label_encoders[col] = LabelEncoder()
            X[col] = self._label_encoders[col].fit_transform(X[col].astype(str))
            self.logger.info(f"Encoded {col} with {len(self._label_encoders[col].classes_)} classes")
        
        return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        self.logger.info("Scaling numerical features")
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.logger.info("No numerical features to scale")
            return X
        
        self.logger.info(f"Scaling numerical columns: {list(numerical_cols)}")
        X[numerical_cols] = self._scaler.fit_transform(X[numerical_cols])
        
        return X
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets with stratification
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.logger.info(f"Splitting data with test size: {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        self.logger.info(f"Train target distribution: {pd.Series(y_train).value_counts().to_dict()}")
        self.logger.info(f"Test target distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing steps applied
        
        Returns:
            dict: Preprocessing information
        """
        if not self._is_fitted:
            self.logger.warning("Preprocessor not fitted yet")
            return {}
        
        info = {
            'is_fitted': self._is_fitted,
            'feature_names': self._feature_names,
            'scaler_fitted': hasattr(self._scaler, 'mean_'),
            'imputer_fitted': hasattr(self._imputer, 'statistics_'),
            'label_encoders': list(self._label_encoders.keys()),
            'numerical_columns_count': self.numerical_columns_count,
            'categorical_columns_count': self.categorical_columns_count
        }
        
        return info

# Inheritance hierarchy
class MedicalDataPreprocessor(DataPreprocessor):
    """Specialized preprocessor for medical data"""
    
    def __init__(self):
        super().__init__()
        self._medical_ranges = self._initialize_medical_ranges()
    
    @property
    def medical_ranges(self) -> Dict[str, Tuple]:
        """Get medical value ranges"""
        return self._medical_ranges.copy()
    
    def _initialize_medical_ranges(self) -> Dict[str, Tuple]:
        """Initialize acceptable medical value ranges"""
        return {
            'age': (0, 120),
            'trestbps': (50, 250),  # blood pressure
            'chol': (100, 600),     # cholesterol
            'thalach': (60, 220)    # max heart rate
        }
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """Override with medical-specific preprocessing"""
        # First validate medical data ranges
        self._validate_medical_ranges(df)
        
        # Then call parent preprocessing
        return super().preprocess_data(df, target_col)
    
    def _validate_medical_ranges(self, df: pd.DataFrame):
        """Validate that medical values are within reasonable ranges"""
        for feature, (min_val, max_val) in self._medical_ranges.items():
            if feature in df.columns:
                out_of_range = ((df[feature] < min_val) | (df[feature] > max_val)).sum()
                if out_of_range > 0:
                    self.logger.warning(f"{out_of_range} values out of medical range for {feature}")

class HeartDiseasePreprocessor(MedicalDataPreprocessor):
    """Highly specialized preprocessor for heart disease data"""
    
    def __init__(self):
        super().__init__()
        self._heart_disease_specific = True
    
    @property
    def is_heart_disease_specific(self) -> bool:
        """Check if this is heart disease specific preprocessor"""
        return self._heart_disease_specific
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """Heart disease specific preprocessing"""
        self.logger.info("Applying heart disease specific preprocessing")
        
        # Heart disease specific steps
        df = self._handle_heart_disease_specific_issues(df)
        
        return super().preprocess_data(df, target_col)
    
    def _handle_heart_disease_specific_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle issues specific to heart disease dataset"""
        # Example: Ensure chest pain types are valid
        if 'cp' in df.columns:
            invalid_cp = ~df['cp'].isin([1, 2, 3, 4])
            if invalid_cp.any():
                self.logger.warning(f"Found {invalid_cp.sum()} invalid chest pain types")
        
        return df