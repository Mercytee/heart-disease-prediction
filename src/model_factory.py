from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple  # ADDED List import
import joblib
import pandas as pd
from .base_classes import BaseModel, ModelStrategy

# Strategy Pattern Implementations
class RandomForestStrategy(ModelStrategy):
    """Concrete strategy for Random Forest"""
    
    def create_model(self) -> Any:
        return RandomForestClassifier(random_state=42)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

class LogisticRegressionStrategy(ModelStrategy):
    """Concrete strategy for Logistic Regression"""
    
    def create_model(self) -> Any:
        return LogisticRegression(random_state=42, max_iter=1000)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }

class SVMStrategy(ModelStrategy):
    """Concrete strategy for SVM"""
    
    def create_model(self) -> Any:
        return SVC(random_state=42, probability=True)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }

class GradientBoostingStrategy(ModelStrategy):
    """Concrete strategy for Gradient Boosting"""
    
    def create_model(self) -> Any:
        return GradientBoostingClassifier(random_state=42)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }

class DecisionTreeStrategy(ModelStrategy):
    """Concrete strategy for Decision Tree"""
    
    def create_model(self) -> Any:
        return DecisionTreeClassifier(random_state=42)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

class ModelFactory(BaseModel):
    """
    Enhanced Factory class with Strategy Pattern and Property Decorators
    """
    
    def __init__(self):
        self._models = {}
        self._best_model = None
        self._best_score = 0
        self._best_model_name = None
        self.training_results = {}
        self.logger = self._setup_logger()
        self._strategies = self._initialize_strategies()
        
    # Property decorators for encapsulation
    @property
    def best_model(self):
        """Getter for best model with validation"""
        if self._best_model is None:
            raise ValueError("No best model available. Train models first.")
        return self._best_model
    
    @best_model.setter
    def best_model(self, value):
        """Setter for best model with validation"""
        if value is None:
            raise ValueError("Best model cannot be None")
        self._best_model = value
    
    @property
    def best_score(self) -> float:
        """Getter for best score"""
        return self._best_score
    
    @property
    def best_model_name(self) -> Optional[str]:
        """Getter for best model name"""
        return self._best_model_name
    
    @property
    def models(self) -> Dict[str, Any]:
        """Getter for models dictionary"""
        return self._models.copy()
    
    def _initialize_strategies(self) -> Dict[str, ModelStrategy]:
        """Initialize all model strategies - POLYMORPHISM in action"""
        return {
            'random_forest': RandomForestStrategy(),
            'logistic_regression': LogisticRegressionStrategy(),
            'svm': SVMStrategy(),
            'gradient_boosting': GradientBoostingStrategy(),
            'decision_tree': DecisionTreeStrategy()
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_models(self) -> Dict[str, Any]:
        """Create models using Strategy Pattern - POLYMORPHISM"""
        self.logger.info("Creating machine learning models using Strategy Pattern")
        
        self._models = {}
        for name, strategy in self._strategies.items():
            self._models[name] = {
                'model': strategy.create_model(),  # Polymorphic call
                'params': strategy.get_hyperparameters()  # Polymorphic call
            }
        
        self.logger.info(f"Created {len(self._models)} different models using strategies")
        return self._models
    
    def train(self, X_train, y_train, cv: int = 5) -> Dict[str, Any]:
        """Implementation of abstract method from BaseModel"""
        self.logger.info("Starting model training with cross-validation")
        
        if not self._models:
            self.create_models()
        
        self.training_results = {}
        
        for name, model_info in self._models.items():
            self.logger.info(f"Training {name}")
            
            try:
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                self.training_results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
                # Update best model using property setter
                if grid_search.best_score_ > self._best_score:
                    self._best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_  # Using setter
                    self._best_model_name = name
                    
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.logger.info(f"Best model: {self.best_model_name} with score: {self.best_score:.4f}")
        return self.training_results
    
    def predict(self, X) -> np.ndarray:
        """Implementation of abstract method from BaseModel"""
        if self._best_model is None:
            raise ValueError("No model trained for prediction")
        return self._best_model.predict(X)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Implementation of abstract method from BaseModel"""
        if self._best_model is None:
            raise ValueError("No model trained for evaluation")
        
        return self.evaluate_model(self._best_model, X_test, y_test)
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """Enhanced evaluation with property-based results"""
        self.logger.info("Evaluating model performance")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except Exception:
                metrics['roc_auc'] = 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
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