from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List  # ADDED List import
import pandas as pd
import numpy as np

class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass  # Abstract method for loading data - must be implemented by subclasses
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        pass  # Abstract method for data validation - must be implemented by subclasses
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        pass  # Abstract method for getting data information - must be implemented by subclasses

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors"""
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        pass  # Abstract method for data preprocessing - must be implemented by subclasses
    
    @abstractmethod
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple:
        pass  # Abstract method for data splitting - must be implemented by subclasses

class BaseModel(ABC):
    """Abstract base class for ML models"""
    
    @abstractmethod
    def train(self, X_train, y_train) -> Dict[str, Any]:
        pass  # Abstract method for model training - must be implemented by subclasses
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass  # Abstract method for making predictions - must be implemented by subclasses
    
    @abstractmethod
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        pass  # Abstract method for model evaluation - must be implemented by subclasses

class ModelStrategy(ABC):
    """Strategy pattern interface for different model algorithms"""
    
    @abstractmethod
    def create_model(self) -> Any:
        pass  # Abstract method for creating model instance - must be implemented by subclasses
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        pass  # Abstract method for getting hyperparameters - must be implemented by subclasses