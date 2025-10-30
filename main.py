import logging
import pandas as pd
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

def main():
    """
    Main function to run the heart disease prediction pipeline
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Heart Disease Prediction System")
    
    try:
        # Initialize the trainer
        data_path = 'data/cleveland.data'
        trainer = ModelTrainer(data_path)
        
        # Run complete pipeline
        results = trainer.run_pipeline()
        
        # Generate and display report
        report = trainer.generate_report()
        print("\n" + "="*60)
        print("HEART DISEASE PREDICTION MODEL PERFORMANCE REPORT")
        print("="*60)
        print(report.to_string(index=False))
        
        # Display best model
        best_model_name = type(results['best_model']).__name__
        best_score = results['best_score']
        print(f"\n🎯 BEST MODEL: {best_model_name} (CV Score: {best_score:.4f})")
        
        # Plot feature importance
        trainer.plot_feature_importance()
        
        # Save best model
        trainer.model_factory.save_model(
            results['best_model'], 
            'best_heart_disease_model.pkl'
        )
        
        logger.info("Heart Disease Prediction System completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()