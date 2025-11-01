"""
Heart Disease Prediction ML Pipeline
"""

_version_ = "1.0.0"  # Version number of the ML pipeline
_author_ = "Mediator Nhongo and Mercy Ngwenya"  # Authors of the pipeline
_description_ = "Machine Learning pipeline for heart disease prediction"  # Description of the package

# Core components - ONLY import what actually exists
from .data_loader import DataLoader, MedicalDataLoader  # Import data loading classes
from .preprocessor import DataPreprocessor  # Import data preprocessing class
from .model_factory import ModelFactory  # Import model creation and training class
from .trainer import ModelTrainer, MedicalModelTrainer, BaseTrainer  # Import training classes

_all_ = [  # List of public classes and functions exposed by this package
    'DataLoader',  # Main data loader class
    'MedicalDataLoader',  # Specialized medical data loader
    'DataPreprocessor',  # Data preprocessing class
    'ModelFactory',  # Model creation and training factory
    'ModelTrainer',  # Main model trainer class
    'MedicalModelTrainer',  # Specialized medical model trainer
    'BaseTrainer',  # Abstract base trainer class
]