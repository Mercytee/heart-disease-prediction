"""
Heart Disease Prediction System
A machine learning project for medical analytics.
"""

__version__ = "2.0.0"  # Updated version
__author__ = "Mercy Thokozani Ngwenya & Mediator Nhongo"
__description__ = "Heart Disease Prediction using Advanced OOP and Machine Learning"

# Import main classes for easier access
from .data_loader import DataLoader, MedicalDataLoader
from .preprocessor import DataPreprocessor, MedicalDataPreprocessor, HeartDiseasePreprocessor
from .model_factory import ModelFactory, RandomForestStrategy, LogisticRegressionStrategy, SVMStrategy, GradientBoostingStrategy
from .trainer import ModelTrainer, BaseTrainer, MedicalModelTrainer
from .predictor import Predictor, BasePredictor, MedicalPredictor
from .base_classes import BaseDataLoader, BasePreprocessor, BaseModel, ModelStrategy

# Export all for easy importing
__all__ = [
    # Core classes
    'DataLoader', 'DataPreprocessor', 'ModelFactory', 'ModelTrainer', 'Predictor',
    # Enhanced classes
    'MedicalDataLoader', 'MedicalDataPreprocessor', 'HeartDiseasePreprocessor',
    'MedicalModelTrainer', 'MedicalPredictor',
    # Abstract base classes
    'BaseDataLoader', 'BasePreprocessor', 'BaseModel', 'BaseTrainer', 'BasePredictor', 'ModelStrategy',
    # Strategy implementations
    'RandomForestStrategy', 'LogisticRegressionStrategy', 'SVMStrategy', 'GradientBoostingStrategy'
]