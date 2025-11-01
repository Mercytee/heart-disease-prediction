import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import joblib
from abc import ABC, abstractmethod
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor

class BasePredictor(ABC):
    """Abstract base class for all predictors"""
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, preprocess: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass  # Abstract method for batch predictions
    
    @abstractmethod
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        pass  # Abstract method for single prediction
    
    @abstractmethod
    def get_prediction_interpretation(self, prediction_result: Dict[str, Any]) -> str:
        pass  # Abstract method for prediction interpretation

class Predictor(BasePredictor):
    """
    Enhanced Prediction class with OOP features
    Uses trained models to make predictions with proper preprocessing
    """
    
    def __init__(self, model_path: str, preprocessor: Optional[DataPreprocessor] = None):
        self._model_path = model_path  # Path to saved model file
        self._model = self._load_model(model_path)  # Load the trained model
        self._preprocessor = preprocessor  # Optional preprocessor for data transformation
        self.logger = self._setup_logger()  # Initialize logging
        self._prediction_history = []  # Store history of predictions made
        
    # Property decorators for controlled access to attributes
    @property
    def model_path(self) -> str:
        """Get model path - read-only property"""
        return self._model_path
    
    @property
    def model(self):
        """Get model with access control - returns the actual model object"""
        return self._model
    
    @property
    def model_type(self) -> str:
        """Get model type name - returns the class name of the model"""
        return type(self._model).__name__
    
    @property
    def preprocessor_available(self) -> bool:
        """Check if preprocessor is available - returns True if preprocessor exists"""
        return self._preprocessor is not None
    
    @property
    def prediction_history(self) -> List[Dict[str, Any]]:
        """Get prediction history - returns copy to prevent external modification"""
        return self._prediction_history.copy()
    
    @property
    def total_predictions_made(self) -> int:
        """Get total number of predictions made - calculated from history"""
        return len(self._prediction_history)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for prediction operations - internal method"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # Avoid duplicate handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_model(self, model_path: str):
        """Load trained model from file with enhanced error handling"""
        try:
            model = joblib.load(model_path)  # Load model using joblib
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Model type: {type(model).__name__}")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {model_path}")
            raise  # Re-raise exception
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise  # Re-raise exception
    
    def predict(self, data: pd.DataFrame, preprocess: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on new data
        Implementation of abstract method from BasePredictor
        """
        try:
            self.logger.info(f"Making predictions on data with shape: {data.shape}")
            
            # Preprocess data if requested and preprocessor available
            X_processed = self._preprocess_prediction_data(data, preprocess)
            
            # Make predictions using the loaded model
            predictions, probabilities = self._make_predictions(X_processed)
            
            # Log prediction results for monitoring
            self._log_prediction_summary(predictions)
            
            # Store prediction in history for tracking
            self._store_prediction_in_history(data.shape[0], 'batch')
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise  # Re-raise exception
    
    def _preprocess_prediction_data(self, data: pd.DataFrame, preprocess: bool) -> pd.DataFrame:
        """Private method for preprocessing prediction data - internal implementation"""
        if preprocess and self.preprocessor_available:  # Check if preprocessing is needed and possible
            self.logger.info("Preprocessing data before prediction")
            # Create a mock target column for preprocessing compatibility
            data_with_target = data.copy()  # Create copy to avoid modifying original
            if 'target' not in data_with_target.columns:  # Add dummy target if missing
                data_with_target['target'] = 0  # Dummy target value
            X_processed, _ = self._preprocessor.preprocess_data(data_with_target)  # Preprocess data
            return X_processed
        else:
            return data  # Return original data if no preprocessing
    
    def _make_predictions(self, X_processed: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Private method for making actual predictions - core prediction logic"""
        predictions = self._model.predict(X_processed)  # Get class predictions
        prediction_proba = None  # Initialize probability as None
        
        if hasattr(self._model, 'predict_proba'):  # Check if model supports probability predictions
            prediction_proba = self._model.predict_proba(X_processed)  # Get probability scores
        
        return predictions, prediction_proba
    
    def _log_prediction_summary(self, predictions: np.ndarray):
        """Private method for logging prediction summary - monitoring and debugging"""
        prediction_counts = pd.Series(predictions).value_counts().to_dict()  # Count predictions by class
        self.logger.info(f"Predictions completed: {len(predictions)} samples")
        self.logger.info(f"Class distribution: {prediction_counts}")  # Log class distribution
    
    def _store_prediction_in_history(self, sample_count: int, prediction_type: str):
        """Private method for storing prediction in history - tracking usage"""
        history_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),  # Current timestamp
            'samples': sample_count,  # Number of samples predicted
            'type': prediction_type,  # Type of prediction (batch/single)
            'model_type': self.model_type  # Type of model used
        }
        self._prediction_history.append(history_entry)  # Add to history
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        Implementation of abstract method from BasePredictor
        """
        try:
            # Convert single sample to DataFrame for compatibility
            single_df = pd.DataFrame([features])  # Create DataFrame from single sample
            
            predictions, probabilities = self.predict(single_df)  # Use batch prediction method
            
            result = self._create_single_prediction_result(predictions, probabilities)  # Format result
            
            # Store single prediction in history
            self._store_prediction_in_history(1, 'single')
            
            self.logger.info(f"Single prediction: {result}")  # Log the result
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise  # Re-raise exception
    
    def _create_single_prediction_result(self, predictions: np.ndarray, 
                                       probabilities: Optional[np.ndarray]) -> Dict[str, Any]:
        """Private method for creating single prediction result - result formatting"""
        result = {
            'prediction': int(predictions[0]),  # Convert prediction to integer
            'prediction_label': 'Heart Disease' if predictions[0] == 1 else 'No Heart Disease',  # Human-readable label
            'timestamp': pd.Timestamp.now().isoformat()  # Prediction timestamp
        }
        
        if probabilities is not None:  # If probability scores available
            result.update({
                'probability_no_disease': float(probabilities[0][0]),  # Probability for class 0
                'probability_disease': float(probabilities[0][1]),  # Probability for class 1
                'confidence': max(probabilities[0])  # Highest probability (confidence)
            })
        
        return result
    
    def get_prediction_interpretation(self, prediction_result: Dict[str, Any]) -> str:
        """
        Provide interpretation of prediction results
        Implementation of abstract method from BasePredictor
        """
        prediction = prediction_result['prediction']  # Get the prediction (0 or 1)
        confidence = prediction_result.get('confidence', 0.0)  # Get confidence, default to 0
        
        # Use Strategy-like pattern for different interpretation levels
        if prediction == 1:  # Positive prediction (heart disease)
            return self._get_positive_interpretation(confidence)
        else:  # Negative prediction (no heart disease)
            return self._get_negative_interpretation(confidence)
    
    def _get_positive_interpretation(self, confidence: float) -> str:
        """Private method for positive prediction interpretation - medical context"""
        interpretation = f"High risk of heart disease (confidence: {confidence:.2%})"  # Base interpretation
        
        if confidence > 0.8:  # High confidence
            interpretation += ". Strong recommendation to consult a cardiologist."
        elif confidence > 0.6:  # Medium confidence
            interpretation += ". Medical evaluation recommended."
        else:  # Low confidence
            interpretation += ". Further tests suggested for confirmation."
        
        return interpretation
    
    def _get_negative_interpretation(self, confidence: float) -> str:
        """Private method for negative prediction interpretation - medical context"""
        interpretation = f"Low risk of heart disease (confidence: {confidence:.2%})"  # Base interpretation
        
        if confidence < 0.7:  # Low confidence in negative prediction
            interpretation += ". Consider additional tests for complete assessment."
        else:  # High confidence in negative prediction
            interpretation += ". Continue with regular health monitoring."
        
        return interpretation
    
    def batch_predict(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on a batch of data from file
        """
        try:
            # Load data from file using DataLoader
            data_loader = DataLoader(file_path)
            data = data_loader.load_data()
            
            # Make predictions using the main predict method
            predictions, probabilities = self.predict(data)
            
            # Create comprehensive results DataFrame
            results_df = self._create_batch_results(data, predictions, probabilities, output_path)
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise  # Re-raise exception
    
    def _create_batch_results(self, data: pd.DataFrame, predictions: np.ndarray, 
                            probabilities: Optional[np.ndarray], output_path: Optional[str]) -> pd.DataFrame:
        """Private method for creating batch prediction results - result formatting"""
        results_df = data.copy()  # Start with original data
        results_df['prediction'] = predictions  # Add predictions column
        results_df['prediction_label'] = results_df['prediction'].map(  # Add human-readable labels
            {0: 'No Heart Disease', 1: 'Heart Disease'}
        )
        
        if probabilities is not None:  # If probability scores available
            results_df['probability_no_disease'] = probabilities[:, 0]  # Class 0 probabilities
            results_df['probability_disease'] = probabilities[:, 1]  # Class 1 probabilities
            results_df['confidence'] = np.max(probabilities, axis=1)  # Maximum probability per sample
            results_df['risk_level'] = results_df['confidence'].apply(  # Categorical risk level
                lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
            )
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)  # Save to CSV without index
            self.logger.info(f"Batch predictions saved to {output_path}")
        
        self.logger.info(f"Batch predictions completed for {len(results_df)} samples")
        return results_df
    
    def clear_prediction_history(self):
        """Clear prediction history - reset tracking"""
        self._prediction_history.clear()  # Clear the history list
        self.logger.info("Prediction history cleared")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics - usage analytics"""
        if not self._prediction_history:  # Check if history is empty
            return {}  # Return empty dict if no predictions
        
        total_predictions = sum(entry['samples'] for entry in self._prediction_history)  # Total samples predicted
        batch_predictions = sum(1 for entry in self._prediction_history if entry['type'] == 'batch')  # Batch prediction count
        single_predictions = sum(1 for entry in self._prediction_history if entry['type'] == 'single')  # Single prediction count
        
        return {
            'total_predictions_made': total_predictions,
            'batch_predictions_count': batch_predictions,
            'single_predictions_count': single_predictions,
            'first_prediction_time': self._prediction_history[0]['timestamp'] if self._prediction_history else None,  # First prediction timestamp
            'last_prediction_time': self._prediction_history[-1]['timestamp'] if self._prediction_history else None  # Last prediction timestamp
        }

# Specialized predictor for medical applications
class MedicalPredictor(Predictor):
    """Specialized predictor for medical applications with enhanced features"""
    
    def __init__(self, model_path: str, preprocessor: Optional[DataPreprocessor] = None):
        super().__init__(model_path, preprocessor)  # Initialize parent class
        self._risk_thresholds = {'low': 0.3, 'medium': 0.6, 'high': 0.8}  # Custom risk thresholds
    
    @property
    def risk_thresholds(self) -> Dict[str, float]:
        """Get risk thresholds - read-only property"""
        return self._risk_thresholds.copy()  # Return copy to prevent modification
    
    def get_detailed_medical_interpretation(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed medical interpretation - enhanced medical context"""
        base_interpretation = self.get_prediction_interpretation(prediction_result)  # Get base interpretation
        
        detailed_info = {
            'interpretation': base_interpretation,  # Base interpretation from parent
            'confidence_level': self._get_confidence_level(prediction_result.get('confidence', 0)),  # Confidence category
            'recommended_actions': self._get_recommended_actions(prediction_result),  # Actionable recommendations
            'risk_category': self._get_risk_category(prediction_result)  # Risk classification
        }
        
        return detailed_info
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description - categorical confidence"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        else:
            return "Low"
    
    def _get_recommended_actions(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Get recommended medical actions - clinical guidance"""
        actions = []  # Initialize empty actions list
        prediction = prediction_result['prediction']  # Get prediction (0 or 1)
        confidence = prediction_result.get('confidence', 0)  # Get confidence score
        
        if prediction == 1:  # Positive prediction (heart disease)
            actions.append("Consult with healthcare provider")  # Always recommend consultation
            if confidence > 0.7:  # High confidence positive
                actions.append("Consider cardiology referral")  # Specialist referral
                actions.append("Schedule follow-up tests")  # Additional testing
            actions.append("Review lifestyle factors")  # Lifestyle recommendations
        else:  # Negative prediction (no heart disease)
            actions.append("Continue regular health monitoring")  # Standard monitoring
            if confidence < 0.6:  # Low confidence negative
                actions.append("Consider preventive screening")  # Additional screening
        
        return actions
    
    def _get_risk_category(self, prediction_result: Dict[str, Any]) -> str:
        """Get risk category based on prediction and confidence - risk stratification"""
        prediction = prediction_result['prediction']  # Get prediction (0 or 1)
        confidence = prediction_result.get('confidence', 0)  # Get confidence score
        
        if prediction == 1:  # Positive prediction
            if confidence >= self._risk_thresholds['high']:  # High confidence positive
                return "High Risk"
            elif confidence >= self._risk_thresholds['medium']:  # Medium confidence positive
                return "Medium Risk"
            else:  # Low confidence positive
                return "Low Risk"
        else:  # Negative prediction
            if confidence >= self._risk_thresholds['high']:  # High confidence negative
                return "Very Low Risk"
            else:  # Lower confidence negative
                return "Low Risk"