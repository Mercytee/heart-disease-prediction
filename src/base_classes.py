from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List  # ADDED List import
import pandas as pd
import numpy as np

class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        pass

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors"""
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        pass
    
    @abstractmethod
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple:
        pass

class BaseModel(ABC):
    """Abstract base class for ML models"""
    
    @abstractmethod
    def train(self, X_train, y_train) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        pass

class ModelStrategy(ABC):
    """Strategy pattern interface for different model algorithms"""
    
    @abstractmethod
    def create_model(self) -> Any:
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        pass