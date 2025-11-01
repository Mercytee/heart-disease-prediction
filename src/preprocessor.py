import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()  # For standardizing numerical features
        self.imputer = SimpleImputer(strategy='median')  # For handling missing values
        self.label_encoders = {}  # Store encoders for categorical features
        self.logger = self._setup_logger()  # Initialize logging
        self.is_fitted = False  # Track if preprocessor has been fitted
        self.feature_names = []  # Store names of processed features
        
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
        data = df.copy()  # Create copy to avoid modifying original data
        
        self.logger.info("Converting target variable to binary classification")
        # FIX: Ensure proper numeric conversion
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')  # Convert to numeric, handle errors
        data[target_col].fillna(0, inplace=True)  # Fill any remaining NaN values with 0
        data[target_col] = data[target_col].apply(lambda x: 1 if float(x) > 0 else 0)  # FIX: float conversion - convert to binary classification
        
        X = data.drop(columns=[target_col])  # Features - all columns except target
        y = data[target_col]  # Target variable
        self.feature_names = X.columns.tolist()  # Store feature names for reference
        
        X = self._handle_missing_values(X)  # Step 1: Handle missing values
        X = self._encode_categorical_features(X)  # Step 2: Encode categorical variables
        X = self._scale_features(X)  # Step 3: Scale numerical features
        
        self.is_fitted = True  # Mark preprocessor as fitted
        self.logger.info("Data preprocessing completed successfully")
        return X, y  # Return processed features and target
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Handling missing values")
        missing_count = X.isnull().sum().sum()  # Count total missing values
        if missing_count == 0:
            self.logger.info("No missing values found")
            return X  # Return early if no missing values
        
        self.logger.info(f"Found {missing_count} missing values")
        numerical_cols = X.select_dtypes(include=[np.number]).columns  # Get numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns  # Get categorical columns
        
        if not numerical_cols.empty:  # If there are numerical columns
            X_numerical = X[numerical_cols]
            if X_numerical.isnull().any().any():  # Check if any numerical columns have missing values
                self.logger.info(f"Imputing numerical columns: {list(numerical_cols)}")
                X[numerical_cols] = self.imputer.fit_transform(X_numerical)  # Impute with median
        
        for col in categorical_cols:  # Handle categorical columns
            if X[col].isnull().any():  # If column has missing values
                self.logger.info(f"Imputing categorical column: {col}")
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'  # Get mode or default
                X[col].fillna(mode_val, inplace=True)  # Fill with mode
        
        self.logger.info("Missing value handling completed")
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Encoding categorical features")
        
        # Define actual categorical columns based on medical dataset knowledge
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Only encode columns that exist and are actually categorical
        categorical_cols = [col for col in categorical_cols if col in X.columns]  # Filter existing columns
        
        if len(categorical_cols) == 0:
            self.logger.info("No categorical features to encode")
            return X  # Return early if no categorical columns
        
        self.logger.info(f"Encoding categorical columns: {categorical_cols}")
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()  # Create encoder for each column
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))  # Encode categorical values
            self.logger.info(f"Encoded {col} with {len(self.label_encoders[col].classes_)} classes")  # Log encoding details
        
        return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Scaling numerical features")
        
        # Define actual numerical columns (exclude categorical ones we encoded)
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        # Only scale columns that exist and are numerical
        numerical_cols = [col for col in numerical_cols if col in X.columns]  # Filter existing columns
        
        # Also include the engineered features if they exist
        engineered_cols = ['bp_category', 'chol_category', 'age_group', 'hr_zone']
        numerical_cols.extend([col for col in engineered_cols if col in X.columns])  # Add engineered features
        
        if len(numerical_cols) == 0:
            self.logger.info("No numerical features to scale")
            return X  # Return early if no numerical columns
        
        self.logger.info(f"Scaling numerical columns: {numerical_cols}")
        
        # Ensure all numerical columns are actually numeric
        for col in numerical_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # Convert to numeric, handle errors
        
        # Handle any NaN values created during conversion
        if X[numerical_cols].isnull().any().any():  # Check for new missing values
            X[numerical_cols] = self.imputer.fit_transform(X[numerical_cols])  # Re-impute if needed
        
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])  # Scale numerical features
        return X
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        self.logger.info(f"Splitting data with test size: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)  # Split with stratification
        self.logger.info(f"Training set: {X_train.shape}")  # Log training set size
        self.logger.info(f"Test set: {X_test.shape}")  # Log test set size
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            self.logger.warning("Preprocessor not fitted yet")
            return {}  # Return empty dict if not fitted
        
        info = {
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'scaler_fitted': hasattr(self.scaler, 'mean_'),  # Check if scaler was fitted
            'imputer_fitted': hasattr(self.imputer, 'statistics_'),  # Check if imputer was fitted
            'label_encoders': list(self.label_encoders.keys()),  # List encoded columns
            'numerical_columns_count': len([col for col in self.feature_names if col not in self.label_encoders]),  # Count numerical columns
            'categorical_columns_count': len(self.label_encoders)  # Count categorical columns
        }
        return info


# Additional preprocessor classes for medical data
class MedicalDataPreprocessor(DataPreprocessor):
    """Enhanced preprocessor specifically for medical data"""
    
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.medical_validation_rules = {  # Define valid ranges for medical features
            'age': (0, 120),
            'trestbps': (50, 250),
            'chol': (100, 600),
            'thalach': (50, 250),
            'oldpeak': (0, 10)
        }
    
    def validate_medical_ranges(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate medical data ranges"""
        self.logger.info("Validating medical data ranges")
        
        for col, (min_val, max_val) in self.medical_validation_rules.items():  # Iterate through validation rules
            if col in X.columns:  # Check if column exists
                # Identify outliers
                outliers = (X[col] < min_val) | (X[col] > max_val)  # Find values outside valid range
                if outliers.any():  # If outliers found
                    self.logger.warning(f"Found {outliers.sum()} outliers in {col}")
                    # Clip values to valid range
                    X[col] = X[col].clip(min_val, max_val)  # Constrain values to valid range
        
        return X
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'num') -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced preprocessing with medical validation"""
        self.logger.info("Starting medical data preprocessing")
        
        # Perform standard preprocessing
        X, y = super().preprocess_data(df, target_col)  # Call parent preprocessing
        
        # Apply medical validation
        X = self.validate_medical_ranges(X)  # Additional medical validation
        
        self.logger.info("Medical data preprocessing completed")
        return X, y


class HeartDiseasePreprocessor(MedicalDataPreprocessor):
    """Specialized preprocessor for heart disease dataset"""
    
    def __init__(self):
        super().__init__()  # Initialize parent medical preprocessor
        self.heart_disease_features = {  # Group features by medical category
            'demographic': ['age', 'sex'],
            'symptoms': ['cp', 'exang'],
            'medical_history': ['fbs', 'restecg'],
            'test_results': ['trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca', 'thal']
        }
    
    def create_clinical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create additional clinical features"""
        self.logger.info("Creating clinical features")
    
        # Age groups
        X['age_group'] = pd.cut(X['age'], bins=[0, 40, 60, 100], labels=[0, 1, 2])  # Create age categories
        
        # Blood pressure categories
        X['bp_category'] = pd.cut(X['trestbps'], 
                                bins=[0, 120, 140, 1000], 
                                labels=[0, 1, 2])  # Normal, Elevated, High
        
        # Cholesterol ratio (using approximation)
        if 'chol' in X.columns and 'thalach' in X.columns:
            X['risk_ratio'] = X['chol'] / X['thalach']  # Create risk ratio feature
        
        return X
    
def preprocess_data(self, df: pd.DataFrame, target_col: str = 'num') -> Tuple[pd.DataFrame, pd.Series]:
    self.logger.info("Starting enhanced data preprocessing pipeline")
    data = df.copy()  # Create copy of data
    
    # Convert target
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')  # Convert target to numeric
    data[target_col].fillna(0, inplace=True)  # Fill missing target values
    data[target_col] = data[target_col].apply(lambda x: 1 if float(x) > 0 else 0)  # Convert to binary
    
    X = data.drop(columns=[target_col])  # Features
    y = data[target_col]  # Target
    
    # Enhanced preprocessing pipeline in correct order
    X = self._handle_missing_values(X)  # First handle missing values
    X = self._create_medical_features(X)  # Then create medical features
    X = self._encode_categorical_features(X)  # Then encode categoricals
    X = self._scale_features(X)  # Finally scale numericals
    
    self.feature_names = X.columns.tolist()  # Store final feature names
    self.is_fitted = True  # Mark as fitted
    self.logger.info("Enhanced data preprocessing completed successfully")
    return X, y

def _create_medical_features(self, X: pd.DataFrame) -> pd.DataFrame:
    """Create medically relevant features"""
    self.logger.info("Creating medical feature engineering")
    
    # Ensure numerical columns are properly converted first
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach']  # Core numerical features
    for col in numerical_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # Ensure numeric type
    
    # Blood pressure categories
    if 'trestbps' in X.columns:
        X['bp_category'] = pd.cut(X['trestbps'], 
                                 bins=[0, 120, 140, 200, 300],  # Define BP ranges
                                 labels=[0, 1, 2, 3])  # Category labels
    
    # Cholesterol categories
    if 'chol' in X.columns:
        X['chol_category'] = pd.cut(X['chol'],
                                   bins=[0, 200, 240, 300, 600],  # Define cholesterol ranges
                                   labels=[0, 1, 2, 3])  # Category labels
    
    # Age groups
    if 'age' in X.columns:
        X['age_group'] = pd.cut(X['age'],
                               bins=[20, 40, 50, 60, 70, 100],  # Define age groups
                               labels=[0, 1, 2, 3, 4])  # Age category labels
    
    # Heart rate zones
    if 'thalach' in X.columns:
        X['hr_zone'] = pd.cut(X['thalach'],
                             bins=[0, 100, 120, 140, 160, 200, 300],  # Define heart rate zones
                             labels=[0, 1, 2, 3, 4, 5])  # Zone labels
    
    self.logger.info("Medical feature engineering completed")
    return X