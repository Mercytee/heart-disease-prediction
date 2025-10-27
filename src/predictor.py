import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import joblib
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor

class Predictor:
    """
    Prediction class for making heart disease predictions on new data
    Uses trained models to make predictions with proper preprocessing
    """
    
    def _init_(self, model_path: str, preprocessor: Optional[DataPreprocessor] = None):
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.preprocessor = preprocessor
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for prediction operations"""
        logger = logging.getLogger(_name_)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Model type: {type(model)._name_}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, preprocess: bool = True) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            data: New data for prediction
            preprocess: Whether to preprocess data before prediction
            
        Returns:
            np.ndarray: Predictions (0 = no disease, 1 = disease)
        """
        try:
            self.logger.info(f"Making predictions on data with shape: {data.shape}")
            
            if preprocess and self.preprocessor:
                self.logger.info("Preprocessing data before prediction")
                # Note: We don't have target column for prediction data
                X_processed, _ = self.preprocessor.preprocess_data(data)
            else:
                X_processed = data
            
            predictions = self.model.predict(X_processed)
            prediction_proba = self.model.predict_proba(X_processed) if hasattr(self.model, 'predict_proba') else None
            
            self.logger.info(f"Predictions completed: {len(predictions)} samples")
            self.logger.info(f"Class distribution: {pd.Series(predictions).value_counts().to_dict()}")
            
            return predictions, prediction_proba
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            dict: Prediction results with probabilities
        """
        try:
            # Convert single sample to DataFrame
            single_df = pd.DataFrame([features])
            
            predictions, probabilities = self.predict(single_df)
            
            result = {
                'prediction': int(predictions[0]),
                'prediction_label': 'Heart Disease' if predictions[0] == 1 else 'No Heart Disease'
            }
            
            if probabilities is not None:
                result['probability_no_disease'] = float(probabilities[0][0])
                result['probability_disease'] = float(probabilities[0][1])
                result['confidence'] = max(result['probability_no_disease'], result['probability_disease'])
            
            self.logger.info(f"Single prediction: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise
    
    def get_prediction_interpretation(self, prediction_result: Dict[str, Any]) -> str:
        """
        Provide interpretation of prediction results
        
        Args:
            prediction_result: Result from predict_single method
            
        Returns:
            str: Interpretation text
        """
        prediction = prediction_result['prediction']
        confidence = prediction_result.get('confidence', 0.0)
        
        if prediction == 1:
            interpretation = f"High risk of heart disease (confidence: {confidence:.2%})"
            if confidence > 0.8:
                interpretation += ". Consider consulting a cardiologist."
            else:
                interpretation += ". Further medical evaluation recommended."
        else:
            interpretation = f"Low risk of heart disease (confidence: {confidence:.2%})"
            if confidence < 0.7:
                interpretation += ". Consider additional tests for confirmation."
        
        return interpretation
    
    def batch_predict(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on a batch of data from file
        
        Args:
            file_path: Path to data file
            output_path: Path to save predictions (optional)
            
        Returns:
            pd.DataFrame: Predictions with original data
        """
        try:
            # Load data
            data_loader = DataLoader(file_path)
            data = data_loader.load_data()
            
            # Make predictions
            predictions, probabilities = self.predict(data)
            
            # Create results DataFrame
            results_df = data.copy()
            results_df['prediction'] = predictions
            results_df['prediction_label'] = results_df['prediction'].map({0: 'No Heart Disease', 1: 'Heart Disease'})
            
            if probabilities is not None:
                results_df['probability_no_disease'] = probabilities[:, 0]
                results_df['probability_disease'] = probabilities[:, 1]
                results_df['confidence'] = np.max(probabilities, axis=1)
            
            # Save results if output path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                self.logger.info(f"Batch predictions saved to {output_path}")
            
            self.logger.info(f"Batch predictions completed for {len(results_df)} samples")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise