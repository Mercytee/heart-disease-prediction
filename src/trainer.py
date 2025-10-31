import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .model_factory import ModelFactory

class ModelTrainer:
    """
    Main training class that orchestrates the entire ML pipeline using OOP
    Coordinates data loading, preprocessing, model training, and evaluation
    """
    
    def __init__(self, data_path: str):  # FIXED: __init__ not _init_
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.model_factory = ModelFactory()
        self.logger = self._setup_logger()
        self.results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training operations"""
        logger = logging.getLogger(__name__)  # FIXED: __name__ not _name_
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete ML pipeline from data loading to model evaluation
        
        Returns:
            dict: Complete pipeline results
        """
        self.logger.info("Starting complete ML pipeline")
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading data")
            df = self.data_loader.load_data()
            
            # Get data information
            data_info = self.data_loader.get_data_info(df)
            self.logger.info(f"Data loaded: {data_info['shape'][0]} samples, {data_info['shape'][1]} features")
            
            # Validate data
            if not self.data_loader.validate_data(df):
                raise ValueError("Data validation failed")
            
            # Step 2: Preprocess data
            self.logger.info("Step 2: Preprocessing data")
            X, y = self.preprocessor.preprocess_data(df)
            
            # Step 3: Split data
            self.logger.info("Step 3: Splitting data")
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
            
            # Step 4: Train models
            self.logger.info("Step 4: Training models")
            training_results = self.model_factory.train_models(X_train, y_train)
            
            # Step 5: Evaluate models
            self.logger.info("Step 5: Evaluating models")
            evaluation_results = self.model_factory.evaluate_all_models(X_test, y_test)
            
            # Store complete results
            self.results = {
                'data_info': data_info,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'best_model': self.model_factory.best_model,
                'best_model_name': self.model_factory.best_model_name,
                'best_score': self.model_factory.best_score,
                'feature_names': self.preprocessor.feature_names,
                'preprocessing_info': self.preprocessor.get_preprocessing_info()
            }
            
            self.logger.info("ML pipeline completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate comprehensive report of model performances
        
        Returns:
            pd.DataFrame: Model performance comparison
        """
        if not self.results:
            self.logger.warning("No results available. Run pipeline first.")
            return pd.DataFrame()
        
        report_df = self.model_factory.get_model_comparison(self.results['evaluation_results'])
        
        self.logger.info("Performance report generated")
        return report_df
    
    def plot_feature_importance(self, top_n: int = 10, save_path: Optional[str] = None):
        """
        Plot feature importance for the best model (if available)
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot (optional)
        """
        if not self.results or self.results['best_model'] is None:
            self.logger.warning("No best model available for feature importance")
            return
        
        best_model = self.results['best_model']
        feature_names = self.results['feature_names']
        
        importance_df = self.model_factory.get_feature_importance(best_model, feature_names)
        
        if importance_df.empty:
            self.logger.info("Best model doesn't support feature importance visualization")
            return
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {self.results["best_model_name"]}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of all models' performance
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.results:
            self.logger.warning("No results available. Run pipeline first.")
            return
        
        report_df = self.generate_report()
        
        # Prepare data for plotting
        metrics_to_plot = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score']
        plot_data = report_df[['Model'] + metrics_to_plot].melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        # Create comparison plot
        plt.figure(figsize=(14, 8))
        sns.barplot(data=plot_data, x='Model', y='Score', hue='Metric', palette='Set2')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed results including all metrics and configurations
        
        Returns:
            dict: Detailed pipeline results
        """
        if not self.results:
            self.logger.warning("No results available. Run pipeline first.")
            return {}
        
        detailed_results = {
            'best_model_info': {
                'name': self.results['best_model_name'],
                'cv_score': self.results['best_score'],
                'type': type(self.results['best_model']).__name__  # FIXED: __name__ not _name_
            },
            'data_info': self.results['data_info'],
            'preprocessing_info': self.results['preprocessing_info'],
            'all_models_performance': self.generate_report().to_dict('records')
        }
        
        return detailed_results
    
    def save_pipeline_results(self, filepath: str = 'pipeline_results.json'):
        """
        Save pipeline results to JSON file
        
        Args:
            filepath: Path to save results
        """
        import json
        
        try:
            # Convert results to serializable format
            results_to_save = self.get_detailed_results()
            
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            self.logger.info(f"Pipeline results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving pipeline results: {str(e)}")