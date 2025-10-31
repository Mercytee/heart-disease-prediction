"""
Heart Disease Prediction System
A machine learning project for medical analytics.
"""

__version__ = "2.0.0"
__author__ = "Mercy Thokozani Ngwenya & Mediator Nhongo"
__description__ = "Heart Disease Prediction using Advanced OOP and Machine Learning"

# Import main classes for easier access
from src.data_loader import DataLoader, MedicalDataLoader
from src.preprocessor import DataPreprocessor
from src.model_factory import ModelFactory
from src.trainer import ModelTrainer

__all__ = [
    'DataLoader', 'MedicalDataLoader', 'DataPreprocessor', 
    'ModelFactory', 'ModelTrainer'
]