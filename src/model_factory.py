import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.exceptions import NotFittedError
import logging
import joblib
from typing import Dict, Any, List, Tuple, Optional

# ADD XGBoost import
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

class ModelFactory:
    def __init__(self):
        print("DEBUG: ModelFactory _init_ called")  # Debug line
        # Initialize all attributes first
        self.models = {}
        self._best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.best_params = None
        
        # Setup logger - SIMPLIFIED APPROACH
        self.logger = logging.getLogger('model_factory')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info("ModelFactory initialized successfully")
    
    def create_models(self) -> Dict[str, Any]:
        """Create multiple machine learning models"""
        self.logger.info("Creating machine learning models")
        
        # Base models configuration
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [3, 4]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            self.models['xgboost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1, 0.05]
                }
            }
        
        self.logger.info(f"Created {len(self.models)} models")
        return self.models
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, use_grid_search: bool = True) -> Dict[str, Any]:
        """Train all models with optional grid search"""
        self.logger.info("Training all models")
        
        if not self.models:
            self.create_models()
        
        training_results = {}
        
        for name, model_info in self.models.items():
            self.logger.info(f"Training {name}")
            
            try:
                if use_grid_search and 'params' in model_info:
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(
                        model_info['model'], 
                        model_info['params'], 
                        cv=5, 
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                else:
                    # Simple training
                    best_model = model_info['model']
                    best_model.fit(X_train, y_train)
                    best_params = {}
                    best_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').mean()
                
                # ✅ CRITICAL FIX: Update the model in self.models with the trained one
                self.models[name]['model'] = best_model
                self.models[name]['best_params'] = best_params
                self.models[name]['cv_score'] = best_score
                
                # Store results
                training_results[name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'cv_mean': best_score,
                    'best_estimator': best_model
                }
                
                # Update best model
                if best_score > self.best_score:
                    self.best_score = best_score
                    self._best_model = best_model
                    self.best_model_name = name
                    self.best_params = best_params
                    
                self.logger.info(f"{name} - Best CV Accuracy: {best_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        if self._best_model is None:
            self.logger.error("No models were successfully trained")
        else:
            self.logger.info(f"Best model: {self.best_model_name} with score: {self.best_score:.4f}")
        
        return training_results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate a single model"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None:
                # Calculate ROC AUC for binary classification
                if y_pred_proba.shape[1] == 2:  # binary classification
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all trained models"""
        self.logger.info("Evaluating all models on test set")
        
        evaluation_results = {}
        
        for name, model_info in self.models.items():
            if 'model' in model_info and hasattr(model_info['model'], 'predict'):
                # Check if model is fitted
                if hasattr(model_info['model'], 'fit'):
                    try:
                        # Try to get predictions to check if fitted
                        _ = model_info['model'].predict(X_test.iloc[:1])  # Small test
                        is_fitted = True
                    except (NotFittedError, Exception):
                        is_fitted = False
                    
                    if not is_fitted:
                        self.logger.warning(f"Model {name} is not fitted, skipping evaluation")
                        continue
                
                self.logger.info(f"Evaluating model: {name}")
                metrics = self.evaluate_model(model_info['model'], X_test, y_test)
                if metrics:  # Only add if evaluation succeeded
                    evaluation_results[name] = metrics
                    self.logger.info(f"{name} - Test Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return evaluation_results
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from model if available"""
        try:
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
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def get_model_comparison(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison table of all models"""
        comparison_data = []
        
        for model_name, metrics in evaluation_results.items():
            # Get CV score from stored models if available
            cv_score = self.models.get(model_name, {}).get('cv_score', self.best_score if model_name == self.best_model_name else 0)
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'CV Score': f"{cv_score:.4f}",
                'Test Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Test Precision': f"{metrics.get('precision', 0):.4f}",
                'Test Recall': f"{metrics.get('recall', 0):.4f}",
                'Test F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                'ROC AUC': f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else 'N/A',
                'Best Parameters': str(self.models.get(model_name, {}).get('best_params', 'N/A'))
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_model(self, model, filepath: str):
        """Save model to file"""
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @property
    def best_model(self):
        """Getter for best model with validation"""
        if self._best_model is None:
            raise ValueError("No best model available. Train models first.")
        return self._best_model
    
    @best_model.setter
    def best_model(self, value):
        """Setter for best model"""
        self._best_model = value