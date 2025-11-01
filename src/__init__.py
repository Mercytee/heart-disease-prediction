"""
Heart Disease Prediction ML Pipeline
"""

_version_ = "1.0.0"
_author_ = "Mediator Nhongo and Mercy Ngwenya"
_description_ = "Machine Learning pipeline for heart disease prediction"

# Core components - ONLY import what actually exists
from .data_loader import DataLoader, MedicalDataLoader
from .preprocessor import DataPreprocessor
from .model_factory import ModelFactory
from .trainer import ModelTrainer, MedicalModelTrainer, BaseTrainer

_all_ = [
    'DataLoader',
    'MedicalDataLoader', 
    'DataPreprocessor',
    'ModelFactory',
    'ModelTrainer', 
    'MedicalModelTrainer',
    'BaseTrainer',
]