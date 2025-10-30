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
        pass
    
    @abstractmethod
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_prediction_interpretation(self, prediction_result: Dict[str, Any]) -> str:
        pass

class Predictor(BasePredictor):
    """
    Enhanced Prediction class with advanced OOP features
    Uses trained models to make predictions with proper preprocessing
    """
    
    def __init__(self, model_path: str, preprocessor: Optional[DataPreprocessor] = None):
        self._model_path = model_path
        self._model = self._load_model(model_path)
        self._preprocessor = preprocessor
        self.logger = self._setup_logger()
        self._prediction_history = []
        
    # Property decorators
    @property
    def model_path(self) -> str:
        """Get model path"""
        return self._model_path
    
    @property
    def model(self):
        """Get model with access control"""
        return self._model
    
    @property
    def model_type(self) -> str:
        """Get model type name"""
        return type(self._model).__name__
    
    @property
    def preprocessor_available(self) -> bool:
        """Check if preprocessor is available"""
        return self._preprocessor is not None
    
    @property
    def prediction_history(self) -> List[Dict[str, Any]]:
        """Get prediction history"""
        return self._prediction_history.copy()
    
    @property
    def total_predictions_made(self) -> int:
        """Get total number of predictions made"""
        return len(self._prediction_history)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for prediction operations"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_model(self, model_path: str):
        """Load trained model from file with enhanced error handling"""
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Model type: {type(model).__name__}")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, preprocess: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on new data
        Implementation of abstract method from BasePredictor
        """
        try:
            self.logger.info(f"Making predictions on data with shape: {data.shape}")
            
            # Preprocess data if requested and preprocessor available
            X_processed = self._preprocess_prediction_data(data, preprocess)
            
            # Make predictions
            predictions, probabilities = self._make_predictions(X_processed)
            
            # Log prediction results
            self._log_prediction_summary(predictions)
            
            # Store in history
            self._store_prediction_in_history(data.shape[0], 'batch')
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _preprocess_prediction_data(self, data: pd.DataFrame, preprocess: bool) -> pd.DataFrame:
        """Private method for preprocessing prediction data"""
        if preprocess and self.preprocessor_available:
            self.logger.info("Preprocessing data before prediction")
            # Create a mock target column for preprocessing compatibility
            data_with_target = data.copy()
            if 'target' not in data_with_target.columns:
                data_with_target['target'] = 0  # Dummy target
            X_processed, _ = self._preprocessor.preprocess_data(data_with_target)
            return X_processed
        else:
            return data
    
    def _make_predictions(self, X_processed: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Private method for making actual predictions"""
        predictions = self._model.predict(X_processed)
        prediction_proba = None
        
        if hasattr(self._model, 'predict_proba'):
            prediction_proba = self._model.predict_proba(X_processed)
        
        return predictions, prediction_proba
    
    def _log_prediction_summary(self, predictions: np.ndarray):
        """Private method for logging prediction summary"""
        prediction_counts = pd.Series(predictions).value_counts().to_dict()
        self.logger.info(f"Predictions completed: {len(predictions)} samples")
        self.logger.info(f"Class distribution: {prediction_counts}")
    
    def _store_prediction_in_history(self, sample_count: int, prediction_type: str):
        """Private method for storing prediction in history"""
        history_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'samples': sample_count,
            'type': prediction_type,
            'model_type': self.model_type
        }
        self._prediction_history.append(history_entry)
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        Implementation of abstract method from BasePredictor
        """
        try:
            # Convert single sample to DataFrame
            single_df = pd.DataFrame([features])
            
            predictions, probabilities = self.predict(single_df)
            
            result = self._create_single_prediction_result(predictions, probabilities)
            
            # Store single prediction in history
            self._store_prediction_in_history(1, 'single')
            
            self.logger.info(f"Single prediction: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise
    
    def _create_single_prediction_result(self, predictions: np.ndarray, 
                                       probabilities: Optional[np.ndarray]) -> Dict[str, Any]:
        """Private method for creating single prediction result"""
        result = {
            'prediction': int(predictions[0]),
            'prediction_label': 'Heart Disease' if predictions[0] == 1 else 'No Heart Disease',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if probabilities is not None:
            result.update({
                'probability_no_disease': float(probabilities[0][0]),
                'probability_disease': float(probabilities[0][1]),
                'confidence': max(probabilities[0])
            })
        
        return result
    
    def get_prediction_interpretation(self, prediction_result: Dict[str, Any]) -> str:
        """
        Provide interpretation of prediction results
        Implementation of abstract method from BasePredictor
        """
        prediction = prediction_result['prediction']
        confidence = prediction_result.get('confidence', 0.0)
        
        # Use Strategy-like pattern for different interpretation levels
        if prediction == 1:
            return self._get_positive_interpretation(confidence)
        else:
            return self._get_negative_interpretation(confidence)
    
    def _get_positive_interpretation(self, confidence: float) -> str:
        """Private method for positive prediction interpretation"""
        interpretation = f"High risk of heart disease (confidence: {confidence:.2%})"
        
        if confidence > 0.8:
            interpretation += ". Strong recommendation to consult a cardiologist."
        elif confidence > 0.6:
            interpretation += ". Medical evaluation recommended."
        else:
            interpretation += ". Further tests suggested for confirmation."
        
        return interpretation
    
    def _get_negative_interpretation(self, confidence: float) -> str:
        """Private method for negative prediction interpretation"""
        interpretation = f"Low risk of heart disease (confidence: {confidence:.2%})"
        
        if confidence < 0.7:
            interpretation += ". Consider additional tests for complete assessment."
        else:
            interpretation += ". Continue with regular health monitoring."
        
        return interpretation
    
    def batch_predict(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on a batch of data from file
        """
        try:
            # Load data
            data_loader = DataLoader(file_path)
            data = data_loader.load_data()
            
            # Make predictions
            predictions, probabilities = self.predict(data)
            
            # Create comprehensive results
            results_df = self._create_batch_results(data, predictions, probabilities, output_path)
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def _create_batch_results(self, data: pd.DataFrame, predictions: np.ndarray, 
                            probabilities: Optional[np.ndarray], output_path: Optional[str]) -> pd.DataFrame:
        """Private method for creating batch prediction results"""
        results_df = data.copy()
        results_df['prediction'] = predictions
        results_df['prediction_label'] = results_df['prediction'].map(
            {0: 'No Heart Disease', 1: 'Heart Disease'}
        )
        
        if probabilities is not None:
            results_df['probability_no_disease'] = probabilities[:, 0]
            results_df['probability_disease'] = probabilities[:, 1]
            results_df['confidence'] = np.max(probabilities, axis=1)
            results_df['risk_level'] = results_df['confidence'].apply(
                lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
            )
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Batch predictions saved to {output_path}")
        
        self.logger.info(f"Batch predictions completed for {len(results_df)} samples")
        return results_df
    
    def clear_prediction_history(self):
        """Clear prediction history"""
        self._prediction_history.clear()
        self.logger.info("Prediction history cleared")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if not self._prediction_history:
            return {}
        
        total_predictions = sum(entry['samples'] for entry in self._prediction_history)
        batch_predictions = sum(1 for entry in self._prediction_history if entry['type'] == 'batch')
        single_predictions = sum(1 for entry in self._prediction_history if entry['type'] == 'single')
        
        return {
            'total_predictions_made': total_predictions,
            'batch_predictions_count': batch_predictions,
            'single_predictions_count': single_predictions,
            'first_prediction_time': self._prediction_history[0]['timestamp'] if self._prediction_history else None,
            'last_prediction_time': self._prediction_history[-1]['timestamp'] if self._prediction_history else None
        }

# Specialized predictor for medical applications
class MedicalPredictor(Predictor):
    """Specialized predictor for medical applications with enhanced features"""
    
    def __init__(self, model_path: str, preprocessor: Optional[DataPreprocessor] = None):
        super().__init__(model_path, preprocessor)
        self._risk_thresholds = {'low': 0.3, 'medium': 0.6, 'high': 0.8}
    
    @property
    def risk_thresholds(self) -> Dict[str, float]:
        """Get risk thresholds"""
        return self._risk_thresholds.copy()
    
    def get_detailed_medical_interpretation(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed medical interpretation"""
        base_interpretation = self.get_prediction_interpretation(prediction_result)
        
        detailed_info = {
            'interpretation': base_interpretation,
            'confidence_level': self._get_confidence_level(prediction_result.get('confidence', 0)),
            'recommended_actions': self._get_recommended_actions(prediction_result),
            'risk_category': self._get_risk_category(prediction_result)
        }
        
        return detailed_info
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        else:
            return "Low"
    
    def _get_recommended_actions(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Get recommended medical actions"""
        actions = []
        prediction = prediction_result['prediction']
        confidence = prediction_result.get('confidence', 0)
        
        if prediction == 1:  # Positive prediction
            actions.append("Consult with healthcare provider")
            if confidence > 0.7:
                actions.append("Consider cardiology referral")
                actions.append("Schedule follow-up tests")
            actions.append("Review lifestyle factors")
        else:  # Negative prediction
            actions.append("Continue regular health monitoring")
            if confidence < 0.6:
                actions.append("Consider preventive screening")
        
        return actions
    
    def _get_risk_category(self, prediction_result: Dict[str, Any]) -> str:
        """Get risk category based on prediction and confidence"""
        prediction = prediction_result['prediction']
        confidence = prediction_result.get('confidence', 0)
        
        if prediction == 1:
            if confidence >= self._risk_thresholds['high']:
                return "High Risk"
            elif confidence >= self._risk_thresholds['medium']:
                return "Medium Risk"
            else:
                return "Low Risk"
        else:
            if confidence >= self._risk_thresholds['high']:
                return "Very Low Risk"
            else:
                return "Low Risk"