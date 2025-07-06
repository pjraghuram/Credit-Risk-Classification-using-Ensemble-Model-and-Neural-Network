import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential, layers, optimizers
import tensorflow as tf
import pickle
import os

print("Starting model training...")

# Load processed data
print("Loading processed data...")
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training class distribution: {np.bincount(y_train)}")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Deep Neural Network 
print("\n Training Deep Neural Network...")
model = Sequential([
    layers.Dense(1024, input_shape=[X_train.shape[1]], activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train 
opt = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy', 'mae'])
history = model.fit(X_train, y_train, batch_size=243, epochs=100, verbose=1)

# Save neural network
model.save('models/neural_network_model.h5')
print("Neural network trained and saved!")

# 2. Ensemble Model 
print("\n Training Ensemble Model...")
estimators = [
    ('SVC', LinearSVC(random_state=42, max_iter=2000)),
    ('LR', LogisticRegression(random_state=43, max_iter=1000)),
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('MLPC', MLPClassifier(random_state=42, max_iter=500))
]

ensemble = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
)
ensemble.fit(X_train, y_train)

# Save ensemble model
with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print("Ensemble model trained and saved!")

# 3. Individual Models 
print("\n Training Individual Models...")

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)

# SVM
print("Training SVM...")
svm = LinearSVC(random_state=42, max_iter=2000)
svm.fit(X_train, y_train)
with open('models/svm.pkl', 'wb') as f:
    pickle.dump(svm, f)

# KNN
print("Training KNN...")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
with open('models/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Naive Bayes
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb, f)

# MLP Classifier
print("Training MLP Classifier...")
mlp = MLPClassifier(random_state=42, max_iter=500)
mlp.fit(X_train, y_train)
with open('models/mlp.pkl', 'wb') as f:
    pickle.dump(mlp, f)

print("All individual models trained and saved!")

# Quick evaluation on test set 
print("\n" + "="*60)
print("QUICK TRAINING EVALUATION")
print("="*60)

models = {
    'Logistic Regression': lr,
    'SVM': svm,
    'KNN': knn,
    'Naive Bayes': nb,
    'MLP': mlp,
    'Ensemble': ensemble
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': accuracy})
    print(f"{name}: {accuracy:.4f}")

# Neural Network evaluation
y_pred_nn = model.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int).flatten()
nn_accuracy = accuracy_score(y_test, y_pred_nn)
results.append({'Model': 'Neural Network', 'Accuracy': nn_accuracy})
print(f"Neural Network: {nn_accuracy:.4f}")

# Save quick results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
results_df.to_csv('results/training_results.csv', index=False)

print(f"\n Best Model (Training): {results_df.iloc[0]['Model']}")
print(f" Best Accuracy (Training): {results_df.iloc[0]['Accuracy']:.4f}")

print("\n Training complete! All models saved to 'models/' directory")
print("Results saved to 'results/training_results.csv'")