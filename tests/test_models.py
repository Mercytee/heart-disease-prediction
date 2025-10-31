import unittest
import pandas as pd
import numpy as np
import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_factory import ModelFactory

class TestHeartDiseaseModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        warnings.filterwarnings("ignore")
        
        # Create realistic sample data that matches Cleveland dataset structure
        self.sample_data = pd.DataFrame({
            'age': [52, 63, 45, 67, 58],
            'sex': [1, 0, 1, 0, 1],
            'cp': [0, 2, 1, 3, 0],
            'trestbps': [125, 140, 138, 160, 120],
            'chol': [212, 268, 200, 286, 250],
            'fbs': [0, 0, 1, 0, 0],
            'restecg': [1, 0, 1, 0, 1],
            'thalach': [168, 160, 130, 108, 150],
            'exang': [0, 0, 1, 1, 0],
            'oldpeak': [1.0, 3.6, 1.5, 1.5, 2.0],
            'slope': [2, 2, 1, 1, 2],
            'ca': [2, 2, 0, 3, 1],
            'thal': [3, 3, 2, 2, 3],
            'num': [0, 1, 0, 1, 0]
        })
    
    def test_data_loader_initialization(self):
        """Test data loader initialization"""
        loader = DataLoader('dummy_path.csv')
        self.assertIsNotNone(loader)
        self.assertEqual(len(loader.column_names), 14)
        self.assertIn('age', loader.column_names)
        self.assertIn('num', loader.column_names)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        self.assertIsNotNone(preprocessor)
        self.assertFalse(preprocessor.is_fitted)
    
    def test_model_factory_creation(self):
        """Test model factory creates all models"""
        factory = ModelFactory()
        models = factory.create_models()
        self.assertEqual(len(models), 5)
        self.assertIn('random_forest', models)
        self.assertIn('logistic_regression', models)
        self.assertIn('gradient_boosting', models)
        self.assertIn('svm', models)
        self.assertIn('decision_tree', models)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(self.sample_data)
        
        self.assertEqual(X.shape[0], 5)  # Same number of samples
        self.assertEqual(len(y), 5)
        self.assertTrue(preprocessor.is_fitted)
        # Check that target is binary
        self.assertTrue(set(y.unique()).issubset({0, 1}))
    
    def test_data_splitting(self):
        """Test data splitting functionality"""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(self.sample_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.4)
        
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 5)
        self.assertEqual(len(y_train) + len(y_test), 5)
        self.assertGreater(X_train.shape[0], 0)
        self.assertGreater(X_test.shape[0], 0)
    
    def test_model_training_structure(self):
        """Test that models can be created and have correct structure"""
        factory = ModelFactory()
        models = factory.create_models()
        
        for name, model_info in models.items():
            self.assertIn('model', model_info)
            self.assertIn('params', model_info)
            self.assertIsNotNone(model_info['model'])
            self.assertIsInstance(model_info['params'], dict)
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        factory = ModelFactory()
        
        # Test with sample data
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(self.sample_data)
        
        # Train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test feature importance
        importance_df = factory.get_feature_importance(model, preprocessor.feature_names)
        
        if not importance_df.empty:
            self.assertEqual(len(importance_df), len(preprocessor.feature_names))
            self.assertIn('feature', importance_df.columns)
            self.assertIn('importance', importance_df.columns)
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation"""
        factory = ModelFactory()
        
        # Create simple test case
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 0]
        
        # We'll test this indirectly by creating a mock model
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="constant", constant=0)
        X_dummy = np.random.randn(5, 5)
        model.fit(X_dummy, y_true)
        
        metrics = factory.evaluate_model(model, X_dummy, y_true)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_complete_pipeline_with_sample_data(self):
        """Test complete pipeline integration with sample data"""
        from src.trainer import ModelTrainer
        
        # Create a more substantial sample dataset
        np.random.seed(42)
        n_samples = 50
        
        sample_data = pd.DataFrame({
            'age': np.random.randint(30, 70, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(100, 180, n_samples),
            'chol': np.random.randint(150, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(80, 180, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 4, n_samples),
            'slope': np.random.randint(1, 4, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(2, 4, n_samples),
            'num': np.random.randint(0, 2, n_samples)  # Binary for testing
        })
        
        # Save sample data temporarily
        sample_data.to_csv('tests/sample_heart_data.data', index=False)
        
        try:
            # Test pipeline
            trainer = ModelTrainer('tests/sample_heart_data.data')
            results = trainer.run_pipeline()
            
            # Verify results structure
            self.assertIn('best_model', results)
            self.assertIn('evaluation_results', results)
            self.assertIn('training_results', results)
            self.assertIsInstance(results['best_score'], float)
            
        finally:
            # Clean up
            if os.path.exists('tests/sample_heart_data.data'):
                os.remove('tests/sample_heart_data.data')

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 