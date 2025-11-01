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
        print("DEBUG: ModelFactory _init_ called")  # Debug line for initialization tracking
        # Initialize all attributes first
        self.models = {}  # Dictionary to store all model configurations
        self._best_model = None  # Private variable for best model
        self.best_model_name = None  # Name of the best performing model
        self.best_score = 0  # Best cross-validation score
        self.best_params = None  # Best hyperparameters found
        
        # Setup logger - SIMPLIFIED APPROACH
        self.logger = logging.getLogger('model_factory')  # Create logger instance
        if not self.logger.handlers:  # Check if handlers already exist
            handler = logging.StreamHandler()  # Create console handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Log format
            handler.setFormatter(formatter)  # Apply format to handler
            self.logger.addHandler(handler)  # Add handler to logger
            self.logger.setLevel(logging.INFO)  # Set logging level
        
        self.logger.info("ModelFactory initialized successfully")  # Log initialization
    
    def create_models(self) -> Dict[str, Any]:
        """Create multiple machine learning models"""
        self.logger.info("Creating machine learning models")  # Log model creation start
        
        # Base models configuration - dictionary of model configurations
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),  # Random Forest classifier
                'params': {  # Hyperparameter grid for tuning
                    'n_estimators': [100, 200],  # Number of trees
                    'max_depth': [10, 15, None],  # Maximum tree depth
                    'min_samples_split': [2, 5]  # Minimum samples to split
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),  # Logistic regression
                'params': {  # Hyperparameter grid
                    'C': [0.1, 1, 10],  # Regularization strength
                    'penalty': ['l2'],  # Regularization type
                    'solver': ['liblinear']  # Optimization algorithm
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),  # Support Vector Machine
                'params': {  # Hyperparameter grid
                    'C': [0.1, 1, 10],  # Regularization parameter
                    'kernel': ['linear', 'rbf'],  # Kernel type
                    'gamma': ['scale']  # Kernel coefficient
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),  # Gradient Boosting
                'params': {  # Hyperparameter grid
                    'n_estimators': [100, 200],  # Number of boosting stages
                    'learning_rate': [0.1, 0.05],  # Learning rate
                    'max_depth': [3, 4]  # Maximum tree depth
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),  # Decision Tree
                'params': {  # Hyperparameter grid
                    'max_depth': [10, 15, None],  # Maximum tree depth
                    'min_samples_split': [2, 5, 10]  # Minimum samples to split
                }
            }
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:  # Check if XGBoost is installed
            self.models['xgboost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),  # XGBoost classifier
                'params': {  # Hyperparameter grid
                    'n_estimators': [100, 200],  # Number of trees
                    'max_depth': [3, 5],  # Maximum tree depth
                    'learning_rate': [0.1, 0.05]  # Learning rate
                }
            }
        
        self.logger.info(f"Created {len(self.models)} models")  # Log total models created
        return self.models  # Return models dictionary
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, use_grid_search: bool = True) -> Dict[str, Any]:
        """Train all models with optional grid search"""
        self.logger.info("Training all models")  # Log training start
        
        if not self.models:  # Check if models are created
            self.create_models()  # Create models if they don't exist
        
        training_results = {}  # Dictionary to store training results
        
        for name, model_info in self.models.items():  # Iterate through all models
            self.logger.info(f"Training {name}")  # Log current model being trained
            
            try:
                if use_grid_search and 'params' in model_info:  # Check if grid search should be used
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(
                        model_info['model'],  # Base model
                        model_info['params'],  # Parameter grid
                        cv=5,  # 5-fold cross-validation
                        scoring='accuracy',  # Scoring metric
                        n_jobs=-1,  # Use all available cores
                        verbose=0  # No verbose output
                    )
                    grid_search.fit(X_train, y_train)  # Perform grid search
                    
                    best_model = grid_search.best_estimator_  # Get best model from grid search
                    best_params = grid_search.best_params_  # Get best parameters
                    best_score = grid_search.best_score_  # Get best cross-validation score
                    
                else:
                    # Simple training without grid search
                    best_model = model_info['model']  # Use base model
                    best_model.fit(X_train, y_train)  # Train the model
                    best_params = {}  # No parameters to store
                    best_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').mean()  # Calculate CV score
                
                # ✅ CRITICAL FIX: Update the model in self.models with the trained one
                self.models[name]['model'] = best_model  # Store trained model
                self.models[name]['best_params'] = best_params  # Store best parameters
                self.models[name]['cv_score'] = best_score  # Store cross-validation score
                
                # Store results in training_results dictionary
                training_results[name] = {
                    'model': best_model,  # Trained model
                    'best_params': best_params,  # Best parameters
                    'cv_mean': best_score,  # Cross-validation mean score
                    'best_estimator': best_model  # Best estimator (same as model)
                }
                
                # Update best model tracking
                if best_score > self.best_score:  # Check if this model is better
                    self.best_score = best_score  # Update best score
                    self._best_model = best_model  # Update best model
                    self.best_model_name = name  # Update best model name
                    self.best_params = best_params  # Update best parameters
                    
                self.logger.info(f"{name} - Best CV Accuracy: {best_score:.4f}")  # Log model performance
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")  # Log training error
                continue  # Continue with next model
        
        if self._best_model is None:  # Check if any model was successfully trained
            self.logger.error("No models were successfully trained")  # Log error
        else:
            self.logger.info(f"Best model: {self.best_model_name} with score: {self.best_score:.4f}")  # Log best model
        
        return training_results  # Return training results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate a single model"""
        try:
            y_pred = model.predict(X_test)  # Make predictions
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None  # Get probabilities if available
            
            metrics = {  # Calculate evaluation metrics
                'accuracy': accuracy_score(y_test, y_pred),  # Accuracy score
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),  # Precision score
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),  # Recall score
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),  # F1 score
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()  # Confusion matrix as list
            }
            
            if y_pred_proba is not None:  # Check if probability predictions are available
                # Calculate ROC AUC for binary classification
                if y_pred_proba.shape[1] == 2:  # binary classification check
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])  # Binary ROC AUC
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # Multi-class ROC AUC
            
            return metrics  # Return calculated metrics
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")  # Log evaluation error
            return {}  # Return empty dictionary on error
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all trained models"""
        self.logger.info("Evaluating all models on test set")  # Log evaluation start
        
        evaluation_results = {}  # Dictionary to store evaluation results
        
        for name, model_info in self.models.items():  # Iterate through all models
            if 'model' in model_info and hasattr(model_info['model'], 'predict'):  # Check if model has predict method
                # Check if model is fitted
                if hasattr(model_info['model'], 'fit'):  # Check if model has fit method
                    try:
                        # Try to get predictions to check if fitted
                        _ = model_info['model'].predict(X_test.iloc[:1])  # Small test prediction
                        is_fitted = True  # Model is fitted
                    except (NotFittedError, Exception):
                        is_fitted = False  # Model is not fitted
                    
                    if not is_fitted:  # Skip unfitted models
                        self.logger.warning(f"Model {name} is not fitted, skipping evaluation")
                        continue
                
                self.logger.info(f"Evaluating model: {name}")  # Log current model evaluation
                metrics = self.evaluate_model(model_info['model'], X_test, y_test)  # Evaluate model
                if metrics:  # Only add if evaluation succeeded
                    evaluation_results[name] = metrics  # Store metrics
                    self.logger.info(f"{name} - Test Accuracy: {metrics.get('accuracy', 0):.4f}")  # Log accuracy
        
        return evaluation_results  # Return evaluation results
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from model if available"""
        try:
            if hasattr(model, 'feature_importances_'):  # Check if model has feature importance
                importances = model.feature_importances_  # Get feature importances
                importance_df = pd.DataFrame({  # Create DataFrame
                    'feature': feature_names,  # Feature names
                    'importance': importances  # Importance scores
                }).sort_values('importance', ascending=False)  # Sort by importance
                return importance_df  # Return sorted DataFrame
            else:
                self.logger.info("Model does not support feature importance")  # Log unsupported feature
                return pd.DataFrame()  # Return empty DataFrame
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")  # Log error
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_model_comparison(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison table of all models"""
        comparison_data = []  # List to store comparison data
        
        for model_name, metrics in evaluation_results.items():  # Iterate through evaluation results
            # Get CV score from stored models if available
            cv_score = self.models.get(model_name, {}).get('cv_score', self.best_score if model_name == self.best_model_name else 0)
            
            comparison_data.append({  # Add model comparison data
                'Model': model_name.replace('_', ' ').title(),  # Format model name
                'CV Score': f"{cv_score:.4f}",  # Cross-validation score
                'Test Accuracy': f"{metrics.get('accuracy', 0):.4f}",  # Test accuracy
                'Test Precision': f"{metrics.get('precision', 0):.4f}",  # Test precision
                'Test Recall': f"{metrics.get('recall', 0):.4f}",  # Test recall
                'Test F1-Score': f"{metrics.get('f1_score', 0):.4f}",  # Test F1-score
                'ROC AUC': f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else 'N/A',  # ROC AUC if available
                'Best Parameters': str(self.models.get(model_name, {}).get('best_params', 'N/A'))  # Best parameters
            })
        
        return pd.DataFrame(comparison_data)  # Return comparison DataFrame
    
    def save_model(self, model, filepath: str):
        """Save model to file"""
        try:
            joblib.dump(model, filepath)  # Save model using joblib
            self.logger.info(f"Model saved to {filepath}")  # Log successful save
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")  # Log save error
            raise  # Re-raise exception
    
    @property
    def best_model(self):
        """Getter for best model with validation"""
        if self._best_model is None:  # Check if best model exists
            raise ValueError("No best model available. Train models first.")  # Raise error if not
        return self._best_model  # Return best model
    
    @best_model.setter
    def best_model(self, value):
        """Setter for best model"""
        self._best_model = value  # Set best model value