import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import pickle
import os

print("Starting preprocessing...")

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Loading the input data 
print("Loading data...")
payments = pd.read_csv("data/payment_data.csv")
payments = payments.set_index("id")
customers = pd.read_csv("data/customer_data.csv", encoding='utf-8')
customers = customers.set_index("id")
customer_data = customers.join(payments)
print(f"Data loaded: {len(customer_data)} records")

# Check null values 
print("\nNull values before cleaning:")
print(customer_data.isnull().sum())

# Simple Feature Engineering (before dropping columns)
print("\nCreating new features...")

# 1. Payment behavior features (grouped by customer)
payment_features = customer_data.groupby(customer_data.index).agg({
    'OVD_sum': ['sum', 'max', 'mean'],
    'new_balance': ['mean', 'std'],
    'pay_normal': ['mean', 'std'],
    'OVD_t1': ['max', 'mean'],
    'OVD_t2': ['max', 'mean'],
    'OVD_t3': ['max', 'mean']
}).fillna(0)

# Flatten column names
payment_features.columns = [
    'total_overdue', 'max_overdue', 'avg_overdue',
    'avg_balance', 'balance_volatility', 'avg_pay_normal', 'pay_consistency',
    'max_ovd_t1', 'avg_ovd_t1', 'max_ovd_t2', 'avg_ovd_t2', 'max_ovd_t3', 'avg_ovd_t3'
]

# 2. Create risk indicators
print("Creating risk indicators...")
payment_features['has_overdue'] = (payment_features['total_overdue'] > 0).astype(int)
payment_features['high_overdue'] = (payment_features['max_overdue'] > 30).astype(int)
payment_features['severe_overdue'] = (payment_features['max_overdue'] > 100).astype(int)

# 3. Financial stability indicators
payment_features['balance_risk_score'] = payment_features['avg_balance'] / (payment_features['avg_balance'].median() + 1)
payment_features['payment_volatility'] = payment_features['pay_consistency'] / (payment_features['avg_pay_normal'] + 1)

# 4. Get one record per customer for joining
customer_base = customer_data.groupby(customer_data.index).first()

# 5. Join payment features back to customer data
customer_data_enhanced = customer_base.join(payment_features)

# 6. Create interaction features (using original + new features)
print("Creating interaction features...")
customer_data_enhanced['fea_4_to_balance_ratio'] = customer_data_enhanced['fea_4'] / (customer_data_enhanced['avg_balance'] + 1)
customer_data_enhanced['overdue_to_income_ratio'] = customer_data_enhanced['total_overdue'] / (customer_data_enhanced['fea_4'] + 1)
customer_data_enhanced['balance_to_income_ratio'] = customer_data_enhanced['avg_balance'] / (customer_data_enhanced['fea_4'] + 1)

# 7. Risk score (composite feature)
customer_data_enhanced['composite_risk_score'] = (
    customer_data_enhanced['has_overdue'] * 0.3 +
    customer_data_enhanced['high_overdue'] * 0.3 +
    customer_data_enhanced['overdue_to_income_ratio'] * 0.2 +
    customer_data_enhanced['payment_volatility'] * 0.2
)

print(f"Original features: {len(customer_base.columns)}")
print(f"New features created: {len(payment_features.columns)}")
print(f"Total features now: {len(customer_data_enhanced.columns)}")

# Use enhanced data for rest of pipeline
customer_data = customer_data_enhanced

# Dropping unwanted features 
print("\nDropping unwanted original features...")
columns_to_drop = ["fea_2", "prod_limit", "report_date", "update_date", "prod_code"]
existing_cols_to_drop = [col for col in columns_to_drop if col in customer_data.columns]
customer_data = customer_data.drop(columns=existing_cols_to_drop)

# Remove rows with missing values 
customer_data.dropna(inplace=True)
print(f"After cleaning: {len(customer_data)} records")

# Check target distribution
print(f"\nTarget distribution:")
print(customer_data['label'].value_counts())
print(f"High risk customers: {customer_data['label'].sum()} ({customer_data['label'].mean():.2%})")

# Separate features and target (like your notebook)
print("\nPreparing features and target...")
Y = customer_data['label']
X = customer_data.drop(['label'], axis=1)

print(f"Feature names: {list(X.columns)}")
print(f"Features shape: {X.shape}")

# Handle missing values with SimpleImputer (like your notebook)
imputer = SimpleImputer(missing_values=0, strategy='mean')
X = imputer.fit_transform(X.to_numpy())

print(f"After imputation: {X.shape}")

# Splitting dataset (exactly like your notebook)
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Feature Extraction - Scaling (like your notebook)
print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA (like your notebook)
print("\nApplying PCA...")
pca = PCA(random_state=42, n_components=0.99)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(f"PCA reduced features to: {X_train.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Handle class imbalance with SMOTE (like your notebook)
print("\nApplying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:")
print(f"Training samples: {len(X_train)}")
print(f"Class distribution: {np.bincount(y_train)}")

# Save processed data
print("\nSaving processed data...")
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

# Save preprocessing objects
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('models/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('models/smote.pkl', 'wb') as f:
    pickle.dump(smote, f)

print("\n Preprocessing complete!")
print(f"Final training shape: {X_train.shape}")
print(f"Final test shape: {X_test.shape}")
print(f"Features extracted: {X_train.shape[1]} principal components")

