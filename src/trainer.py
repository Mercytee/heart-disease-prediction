import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
from abc import ABC, abstractmethod
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .model_factory import ModelFactory

class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    @abstractmethod
    def run_pipeline(self) -> Dict[str, Any]:
        # This method must be implemented by child classes to run the full ML pipeline
        pass
    
    @abstractmethod
    def generate_report(self) -> pd.DataFrame:
        # This method must be implemented by child classes to generate performance reports
        pass
    
    @abstractmethod
    def get_detailed_results(self) -> Dict[str, Any]:
        # This method must be implemented by child classes to provide detailed results
        pass

class ModelTrainer(BaseTrainer):
    """
    Enhanced Main training class with  OOP features
    Coordinates data loading, preprocessing, model training, and evaluation
    """
    
    def __init__(self, data_path: str):
        # Initialize the trainer with data path and component instances
        self._data_path = data_path
        self._data_loader = DataLoader(data_path)  # Handles data loading operations
        self._preprocessor = DataPreprocessor()    # Handles data preprocessing
        self._model_factory = ModelFactory()       # Handles model creation and training
        self.logger = self._setup_logger()         # Setup logging for tracking operations
        self._results = {}                         # Store pipeline results
        self._pipeline_executed = False            # Track if pipeline has been run
    
    # Property decorators for better encapsulation and controlled access
    @property
    def data_path(self) -> str:
        """Get data path - read-only property"""
        return self._data_path
    
    @property
    def model_factory(self):
        """Get model factory instance - ADDED THIS PROPERTY"""
        return self._model_factory
    
    @property
    def results(self) -> Dict[str, Any]:
        """Get results with access control - prevents access before pipeline execution"""
        if not self._pipeline_executed:
            raise ValueError("Pipeline not executed. Call run_pipeline() first.")
        return self._results.copy()  # Return copy to prevent external modification
    
    @property
    def best_model(self):
        """Get best model with validation - ensures pipeline was executed first"""
        if not self._pipeline_executed:
            raise ValueError("Pipeline not executed. Call run_pipeline() first.")
        return self._results.get('best_model')
    
    @property
    def best_model_name(self) -> Optional[str]:
        """Get best model name - returns None if no best model available"""
        if not self._pipeline_executed:
            raise ValueError("Pipeline not executed. Call run_pipeline() first.")
        return self._results.get('best_model_name')
    
    @property
    def pipeline_executed(self) -> bool:
        """Check if pipeline has been executed - simple status flag"""
        return self._pipeline_executed
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training operations - private method for internal use"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # Avoid adding multiple handlers if logger already exists
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete ML pipeline from data loading to model evaluation
        Implementation of abstract method from BaseTrainer
        """
        self.logger.info("Starting complete ML pipeline")
        
        try:
            # Step 1: Load and validate data - ensures data quality before processing
            self.logger.info("Step 1: Loading and validating data")
            df = self._load_and_validate_data()
            
            # Step 2: Preprocess data - transform raw data into model-ready format
            self.logger.info("Step 2: Preprocessing data")
            X_train, X_test, y_train, y_test = self._preprocess_data(df)
            
            # Step 3: Train and evaluate models - core ML operations
            self.logger.info("Step 3: Training and evaluating models")
            self._train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            self._pipeline_executed = True  # Mark pipeline as completed
            self.logger.info("ML pipeline completed successfully")
            return self._results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise  # Re-raise exception to let caller handle it
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Private method for data loading and validation - internal implementation detail"""
        df = self._data_loader.load_data()  # Delegate data loading to DataLoader
        
        data_info = self._data_loader.get_data_info(df)
        self.logger.info(f"Data loaded: {data_info['shape'][0]} samples, {data_info['shape'][1]} features")
        
        if not self._data_loader.validate_data(df):
            raise ValueError("Data validation failed")  # Stop pipeline if data is invalid
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple:
        """Private method for data preprocessing - separates features and target, splits data"""
        X, y = self._preprocessor.preprocess_data(df)  # Feature engineering, cleaning, etc.
        return self._preprocessor.split_data(X, y)     # Train-test split
    
    def _train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Private method for model training and evaluation - core ML operations"""
        # Train multiple models using the factory pattern
        training_results = self._model_factory.train(X_train, y_train)
        # Evaluate all trained models on test data
        evaluation_results = self._model_factory.evaluate_all_models(X_test, y_test)
        
        # Store complete results for later access
        self._results = {
            'data_info': self._data_loader.get_data_info(),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'best_model': self._model_factory.best_model,      # Best performing model instance
            'best_model_name': self._model_factory.best_model_name,  # Name of best model
            'best_score': self._model_factory.best_score,      # Performance score of best model
            'feature_names': self._preprocessor.feature_names, # Names of features used
            'preprocessing_info': self._preprocessor.get_preprocessing_info()  # How data was preprocessed
        }
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate comprehensive report of model performances
        Implementation of abstract method from BaseTrainer
        """
        if not self._pipeline_executed:
            self.logger.warning("No results available. Run pipeline first.")
            return pd.DataFrame()  # Return empty DataFrame instead of raising error
        
        # Delegate report generation to model factory
        report_df = self._model_factory.get_model_comparison(self._results['evaluation_results'])
        self.logger.info("Performance report generated")
        return report_df
    
    def plot_feature_importance(self, top_n: int = 10, save_path: Optional[str] = None):
        """
        Plot feature importance for the best model (if available)
        
        Args:
            top_n: Number of top features to plot - limits display to most important features
            save_path: Path to save the plot (optional) - for persistence
        """
        if not self._pipeline_executed or self.best_model is None:
            self.logger.warning("No best model available for feature importance")
            return  # Graceful exit if requirements not met
        
        feature_names = self._results['feature_names']
        # Get feature importance data from model factory
        importance_df = self._model_factory.get_feature_importance(self.best_model, feature_names)
        
        if importance_df.empty:
            self.logger.info("Best model doesn't support feature importance visualization")
            return  # Some models (like SVM) don't have feature importance
        
        # Create visualization using Strategy-like pattern
        self._create_feature_importance_plot(importance_df, top_n, save_path)
    
    def _create_feature_importance_plot(self, importance_df: pd.DataFrame, top_n: int, save_path: Optional[str]):
        """Private method for creating feature importance plot - visualization logic"""
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)  # Get top N most important features
        
        # Polymorphic styling based on model type - different colors for different models
        palette = self._get_plot_palette(self.best_model_name)
        
        # Create horizontal bar plot - most important features at top
        sns.barplot(data=top_features, x='importance', y='feature', palette=palette)
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()  # Improve layout
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High quality save
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()  # Display the plot
    
    def _get_plot_palette(self, model_name: str) -> str:
        """Strategy-like method for different plot styles - different colors for different models"""
        palettes = {
            'random_forest': 'viridis',        # Green-blue palette for tree-based
            'gradient_boosting': 'plasma',     # Purple-yellow for boosting
            'logistic_regression': 'coolwarm', # Blue-red for linear models
            'svm': 'magma'                     # Black-red-purple for SVM
        }
        return palettes.get(model_name, 'viridis')  # Default palette
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of all models' performance - visual model selection aid
        """
        if not self._pipeline_executed:
            self.logger.warning("No results available. Run pipeline first.")
            return
        
        report_df = self.generate_report()
        self._create_model_comparison_plot(report_df, save_path)
    
    def _create_model_comparison_plot(self, report_df: pd.DataFrame, save_path: Optional[str]):
        """Private method for creating model comparison plot - visualization logic"""
        # Select metrics to include in the comparison plot
        metrics_to_plot = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score']
        # Transform data from wide to long format for seaborn
        plot_data = report_df[['Model'] + metrics_to_plot].melt(
            id_vars=['Model'], var_name='Metric', value_name='Score'
        )
        
        plt.figure(figsize=(14, 8))
        # Create grouped bar plot - models on x-axis, metrics as different colored bars
        sns.barplot(data=plot_data, x='Model', y='Score', hue='Metric', palette='Set2')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)  # Rotate model names for better readability
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed results including all metrics and configurations
        Implementation of abstract method from BaseTrainer
        """
        if not self._pipeline_executed:
            self.logger.warning("No results available. Run pipeline first.")
            return {}  # Return empty dict instead of raising error
        
        # Comprehensive results structure for analysis and reporting
        detailed_results = {
            'best_model_info': {
                'name': self.best_model_name,
                'cv_score': self._results['best_score'],  # Cross-validation score
                'type': type(self.best_model).__name__    # Python class name
            },
            'data_info': self._results['data_info'],
            'preprocessing_info': self._results['preprocessing_info'],
            'all_models_performance': self.generate_report().to_dict('records'),  # List of dicts
            'pipeline_status': 'completed',
            'execution_timestamp': pd.Timestamp.now().isoformat()  # When pipeline was run
        }
        
        return detailed_results
    
    def save_pipeline_results(self, filepath: str = 'pipeline_results.json'):
        """
        Save pipeline results to JSON file - for persistence and sharing results
        """
        try:
            results_to_save = self.get_detailed_results()
            
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)  
            
            self.logger.info(f"Pipeline results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving pipeline results: {str(e)}")

# Inheritance example - Specialized trainer for medical data
class MedicalModelTrainer(ModelTrainer):
    """Specialized trainer for medical data with additional validation"""
    
    def __init__(self, data_path: str, medical_validation: bool = True):
        # Initialize parent class first
        super().__init__(data_path)
        self.medical_validation = medical_validation  # Flag for medical-specific checks
        self._medical_metrics = ['sensitivity', 'specificity', 'auc_roc']  # Medical performance metrics
    
    @property
    def medical_metrics(self) -> list:
        """Get medical-specific metrics - important for medical model evaluation"""
        return self._medical_metrics.copy()  # Return copy to prevent external modification
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Override with medical-specific validation - extends parent functionality"""
        if self.medical_validation:
            self.logger.info("Running medical data specific validation")
            self._validate_medical_data_quality()  # Additional medical checks
        
        # Call parent class pipeline execution
        return super().run_pipeline()
    
    def _validate_medical_data_quality(self):
        """Medical-specific data quality checks - ensures data meets medical standards"""
        # Implement medical data validation logic
        self.logger.info("Performing medical data quality checks...")
        # Check for data completeness, value ranges, etc. 
        
        # Actual implementation would include checks like:
        # - Valid value ranges for medical measurements
        # - Required fields for medical compliance
        # - Data provenance and quality indicators
        pass