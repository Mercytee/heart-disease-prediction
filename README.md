#  Heart Disease Prediction System

A machine learning pipeline for predicting heart disease using clinical data. This system implements multiple ML models with hyperparameter tuning and provides comprehensive performance analysis.

## Project Overview

This project implements a complete machine learning pipeline for binary classification of heart disease presence using the Cleveland Heart Disease dataset. The system includes data preprocessing, feature engineering, model training with hyperparameter optimization, and performance evaluation.

##  Features

- **Data Preprocessing**: Automated handling of missing values, categorical encoding, and feature scaling
- **Multiple ML Models**: Random Forest, SVM, Logistic Regression, Gradient Boosting, Decision Tree, XGBoost
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Feature Engineering**: Medical domain-specific feature creation
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Persistence**: Save/load best performing model
- **Detailed Logging**: Complete pipeline execution tracking

##  Dataset

The system uses the Cleveland Heart Disease Dataset with 14 clinical features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Gender (1=male, 0=female) | Categorical |
| cp | Chest pain type | Categorical |
| trestbps | Resting blood pressure | Numerical |
| chol | Serum cholesterol | Numerical |
| fbs | Fasting blood sugar | Categorical |
| restecg | Resting electrocardiographic results | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Categorical |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of the peak exercise ST segment | Categorical |
| ca | Number of major vessels colored by fluoroscopy | Categorical |
| thal | Thalassemia type | Categorical |
| num | Target variable (0=no disease, 1-4=disease) | Target |

## Project Structure

```
heart-disease-prediction/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and validation
│   ├── preprocessor.py         # Data preprocessing pipeline
│   ├── model_factory.py        # Model creation and training
│   ├── trainer.py              # Main training pipeline
│   └── base_classes.py         # Abstract base classes
├── data/
│   └── cleveland.data          # Heart disease dataset
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_experimentation.ipynb
├── tests/
│   └── test_models.py
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
└── README.md
```

##Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mercytee/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

### Run Complete Pipeline
```bash
python main.py
```

This executes the full ML pipeline:
1. Data loading and validation
2. Data preprocessing and feature engineering
3. Model training with hyperparameter tuning
4. Model evaluation and comparison
5. Best model selection and saving

### Expected Output
```
============================================================
HEART DISEASE PREDICTION SYSTEM
============================================================
All imports successful!
Dataset already exists
Using dataset: data/cleveland.data

Running complete ML pipeline...
1. Loading data...
2. Preprocessing...
3. Training models...
4. Evaluating models...

HEART DISEASE PREDICTION MODEL PERFORMANCE REPORT
============================================================
              Model  CV Score  Test Accuracy  Test Precision  Test Recall  Test F1-Score
                svm    0.8435         0.8361          0.8436       0.8361         0.8362
logistic_regression    0.8270         0.8197          0.8308       0.8197         0.8197
      random_forest    0.8107         0.8033          0.8436       0.8033         0.8006

 BEST MODEL: SVC
Cross-Validation Score: 0.8435
Best model saved to: best_heart_disease_model.pkl
```

## Model Performance

Typical performance metrics (may vary based on data split):

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|-------|-------------|---------------|-----------|--------|----------|
| SVM | 84.35% | 83.61% | 84.36% | 83.61% | 83.62% |
| Logistic Regression | 82.70% | 81.97% | 83.08% | 81.97% | 81.97% |
| Random Forest | 81.07% | 80.33% | 84.36% | 80.33% | 80.06% |
| Gradient Boosting | 79.02% | 77.05% | 78.48% | 77.05% | 77.01% |
| XGBoost | 80.65% | 77.05% | 78.48% | 77.05% | 77.01% |

##  Configuration

### Hyperparameter Tuning
Models are optimized using GridSearchCV with the following parameter ranges:

- **Random Forest**: n_estimators=[100,200], max_depth=[10,15,None]
- **SVM**: C=[0.1,1,10], kernel=['linear','rbf']
- **Logistic Regression**: C=[0.1,1,10], penalty=['l2']
- **Gradient Boosting**: n_estimators=[100,200], learning_rate=[0.1,0.05]

### Preprocessing Settings
- Missing values: Median imputation for numerical, mode for categorical
- Categorical encoding: Label encoding
- Feature scaling: StandardScaler for numerical features
- Test split: 20% for evaluation

##  Testing

Run the test suite:
```bash
python -m pytest tests/
```

##  Logs

The system generates detailed logs:
- `heart_disease_analysis.log` - Complete pipeline execution log
- `pipeline_results.json` - Structured results and metrics

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Authors

- **Mercy Ngwenya** - [Mercytee](https://github.com/Mercytee)
- **Mediator Nhongo** - [MediatorNhongo](https://github.com/nhongomediator-blip)

##  Acknowledgments

- Cleveland Heart Disease Dataset providers
- Scikit-learn, pandas, and numpy communities
- Open-source ML libraries and tools
