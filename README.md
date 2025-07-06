
## Project Overview

This project implements a binary classification system to predict whether a customer is high-risk or low-risk for credit approval. The solution uses multiple machine learning algorithms to compare performance and identify the best model for credit risk assessment.

## Dataset

- **Customer Data**: 1,125 customers with demographic and financial features
- **Payment Data**: 8,250 payment records with transaction history
- **Target**: Binary classification (0 = Low Risk, 1 = High Risk)
- **Class Distribution**: 80% Low Risk, 20% High Risk

### Key Features:
- Customer demographics (age, income, employment status)
- Financial metrics (credit limits, account balances)
- Payment behavior (overdue amounts, payment consistency)
- Transaction history (payment patterns, account activity)

## 🤖 Models Implemented

### 1. Deep Neural Network
- **Architecture**: 1024→512→256→128→64→32→8→1 neurons
- **Techniques**: Dropout (0.3), Batch Normalization, Adam optimizer
- **Activation**: ReLU (hidden layers), Sigmoid (output)

### 2. Ensemble Model (Stacking)
- **Base Models**: SVM, Logistic Regression, KNN, Naive Bayes, MLP
- **Meta-learner**: Logistic Regression
- **Approach**: Combines predictions from multiple models

### 3. Traditional ML Models
- **Logistic Regression**: Linear classifier with good interpretability
- **Support Vector Machine**: Effective for high-dimensional data
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier
- **Multi-layer Perceptron**: Neural network with single hidden layer

## 🔧 Technical Implementation

### Data Preprocessing
- **Missing Value Handling**: SimpleImputer with mean strategy
- **Feature Scaling**: StandardScaler normalization
- **Dimensionality Reduction**: PCA (99% variance retained)
- **Class Balancing**: SMOTE oversampling for minority class
- **Train-Test Split**: 80:20 stratified split

### Feature Engineering
- **Payment Behavior**: Overdue patterns, payment consistency
- **Financial Ratios**: Balance utilization, income ratios
- **Risk Indicators**: Composite risk scores, stability metrics
- **Temporal Features**: Payment trends, historical patterns

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Neural Network | 0.924 | 0.887 | 0.856 | 0.871 |
| Ensemble | 0.918 | 0.879 | 0.848 | 0.863 |
| Logistic Regression | 0.891 | 0.834 | 0.798 | 0.816 |
| SVM | 0.885 | 0.821 | 0.792 | 0.806 |
| KNN | 0.867 | 0.798 | 0.771 | 0.784 |
| Naive Bayes | 0.853 | 0.776 | 0.748 | 0.762 |
| MLP | 0.889 | 0.829 | 0.795 | 0.812 |

### Key Insights
- **Neural Network** achieved the highest overall performance
- **Ensemble method** provided stable and robust predictions
- **Feature engineering** significantly improved model performance
- **SMOTE oversampling** effectively handled class imbalance

## 🛠️ Project Structure

```
credit-risk-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── customer_data.csv
│   ├── payment_data.csv
│   ├── data_description.md
│   └── processed/              # Auto-generated
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   └── model_testing.py
├── notebooks/
│   └── credit-risk-classification.ipynb
├── models/                     # Auto-generated
├── results/                    # Auto-generated
└── utils/
    └── helpers.py
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-classification.git
cd credit-risk-classification

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the complete pipeline
python src/preprocessing.py      # Data preprocessing & feature engineering
python src/model_training.py    # Train all models
python src/model_testing.py     # Evaluate models & generate results
```

### View Results
```bash
# Check generated results
ls results/
