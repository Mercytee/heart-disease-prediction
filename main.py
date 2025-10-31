import logging
import pandas as pd
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer import ModelTrainer

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('heart_disease_analysis.log'),
            logging.StreamHandler()
        ]
    )

def download_dataset():
    """Download the heart disease dataset if not exists"""
    data_path = 'data/cleveland.data'
    
    if os.path.exists(data_path):
        print("✅ Dataset already exists")
        return True
        
    print("📥 Downloading dataset...")
    try:
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # UCI dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        # Column names as per UCI documentation
        column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        
        # Download and load the data
        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # Save to local file
        df.to_csv(data_path, index=False)
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please download manually from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
        print("And save as 'data/cleveland.data'")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from src.data_loader import DataLoader
        from src.preprocessor import DataPreprocessor
        from src.model_factory import ModelFactory
        from src.trainer import ModelTrainer
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages:")
        print("pip install pandas numpy scikit-learn matplotlib seaborn joblib")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """
    Main function to run the heart disease prediction pipeline
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("🫀 HEART DISEASE PREDICTION SYSTEM")
    print("=" * 60)
    
    try:
        # Test imports first
        if not test_imports():
            return
        
        # Download dataset if needed
        if not download_dataset():
            return
        
        logger.info("Starting Heart Disease Prediction System")
        
        # Initialize the trainer
        data_path = 'data/cleveland.data'
        print(f"📁 Using dataset: {data_path}")
        
        trainer = ModelTrainer(data_path)
        
        # Run complete pipeline
        print("\n🚀 Running complete ML pipeline...")
        print("1. Loading data...")
        print("2. Preprocessing...") 
        print("3. Training models...")
        print("4. Evaluating models...")
        
        results = trainer.run_pipeline()
        
        # Generate and display report
        print("\n" + "=" * 60)
        print("📊 HEART DISEASE PREDICTION MODEL PERFORMANCE REPORT")
        print("=" * 60)
        
        report = trainer.generate_report()
        print(report.to_string(index=False))
        
        # Display best model
        best_model_name = type(results['best_model']).__name__
        best_score = results['best_score']
        print(f"\n🎯 BEST MODEL: {best_model_name}")
        print(f"📈 Cross-Validation Score: {best_score:.4f}")
        
        # Show feature importance
        print(f"\n🔍 Top 5 Most Important Features:")
        try:
            trainer.plot_feature_importance(top_n=5)
        except Exception as e:
            print(f"Note: Feature importance visualization not available: {e}")
        
        # Save best model
        model_path = 'best_heart_disease_model.pkl'
        trainer.model_factory.save_model(results['best_model'], model_path)
        print(f"💾 Best model saved to: {model_path}")
        
        # Save results
        results_path = 'pipeline_results.json'
        trainer.save_pipeline_results(results_path)
        print(f"📄 Pipeline results saved to: {results_path}")
        
        print("\n" + "=" * 60)
        print("✅ HEART DISEASE PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        logger.info("Heart Disease Prediction System completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
        print("Please make sure the dataset exists at 'data/cleveland.data'")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def quick_test():
    """Quick test function to verify basic functionality"""
    print("\n🔧 Running quick test...")
    try:
        from src.data_loader import DataLoader
        from src.preprocessor import DataPreprocessor
        
        # Test data loading
        loader = DataLoader('data/cleveland.data')
        df = loader.load_data()
        print(f"✅ Data loading: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(df)
        print(f"✅ Data preprocessing: {X.shape[1]} features after processing")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if user wants quick test
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        quick_test()
    else:
        main()