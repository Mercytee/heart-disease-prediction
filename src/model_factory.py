from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import joblib
import pandas as pd

class ModelFactory:
    """
    Factory class for creating and managing multiple ML models with OOP design
    Handles model creation, training, evaluation, and persistence
    """
    
    def __init__(self):  # FIXED: __init__ not _init_
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.training_results = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for model operations"""
        logger = logging.getLogger(__name__)  # FIXED: __name__ not _name_
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create multiple classification models for heart disease prediction
        
        Returns:
            dict: Dictionary of model configurations
        """
        self.logger.info("Creating machine learning models")
        
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
        
        self.logger.info(f"Created {len(self.models)} different models")
        return self.models
    
    def train_models(self, X_train, y_train, cv: int = 5) -> Dict[str, Any]:
        """
        Train all models using cross-validation and hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            
        Returns:
            dict: Training results for all models
        """
        self.logger.info("Starting model training with cross-validation")
        
        if not self.models:
            self.create_models()
        
        self.training_results = {}
        
        for name, model_info in self.models.items():
            self.logger.info(f"Training {name}")
            
            try:
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store results
                self.training_results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_,
                    'grid_search': grid_search
                }
                
                self.logger.info(f"{name} - Best score: {grid_search.best_score_:.4f}, Best params: {grid_search.best_params_}")
                
                # Update best model
                if grid_search.best_score_ > self.best_score:
                    self.best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = name
                    
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.logger.info(f"Best model: {self.best_model_name} with score: {self.best_score:.4f}")
        return self.training_results
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance on test set
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating model performance")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['true_negative'], metrics['false_positive'], metrics['false_negative'], metrics['true_positive'] = cm.ravel()
        
        self.logger.info(f"Model evaluation completed - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def evaluate_all_models(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation results for all models
        """
        self.logger.info("Evaluating all models on test set")
        
        evaluation_results = {}
        
        for name, result in self.training_results.items():
            model = result['model']
            metrics = self.evaluate_model(model, X_test, y_test)
            evaluation_results[name] = {
                'training_score': result['best_score'],
                'test_metrics': metrics,
                'model': model,
                'best_params': result['best_params']
            }
            
            self.logger.info(f"{name} - Test Accuracy: {metrics['accuracy']:.4f}")
        
        return evaluation_results
    
    def get_model_comparison(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Create comparison DataFrame of all models
        
        Args:
            evaluation_results: Results from evaluate_all_models
            
        Returns:
            pd.DataFrame: Model comparison table
        """
        comparison_data = []
        
        for name, result in evaluation_results.items():
            row = {
                'Model': name,
                'CV Score': result['training_score'],
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test Precision': result['test_metrics']['precision'],
                'Test Recall': result['test_metrics']['recall'],
                'Test F1-Score': result['test_metrics']['f1_score'],
                'Best Parameters': str(result['best_params'])
            }
            
            if 'roc_auc' in result['test_metrics']:
                row['ROC AUC'] = result['test_metrics']['roc_auc']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Test Accuracy', ascending=False)
    
    def save_model(self, model, filepath: str):
        """
        Save trained model to file
        
        Args:
            model: Model to save
            filepath: Path to save the model
        """
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model or None if error
        """
        try:
            model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from model if available
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.info("Model does not support feature importance")
            return pd.DataFrame()