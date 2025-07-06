
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from tensorflow.keras.models import load_model
import pickle
import os

print("Starting model testing...")

# Load test data
print("Loading test data...")
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Create results directory
os.makedirs('results', exist_ok=True)

# Load all models
print("\nLoading trained models...")
models = {}

# Load sklearn models
model_files = ['logistic_regression.pkl', 'svm.pkl', 'knn.pkl', 'naive_bayes.pkl', 'mlp.pkl', 'ensemble_model.pkl']
for file in model_files:
    model_name = file.replace('.pkl', '').replace('_model', '')
    try:
        with open(f'models/{file}', 'rb') as f:
            models[model_name] = pickle.load(f)
        print(f" Loaded {model_name}")
    except Exception as e:
        print(f" Could not load {model_name}: {e}")

# Load neural network
try:
    models['neural_network'] = load_model('models/neural_network_model.h5')
    print(" Loaded neural_network")
except Exception as e:
    print(f" Could not load neural_network: {e}")

# ENHANCED: Better Evaluation Metrics Function
def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive metrics for credit risk evaluation"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add ROC-AUC if probabilities available
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
    
    # Add business-relevant metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    metrics['confusion_matrix'] = cm
    
    return metrics

# ENHANCED: Business Impact Analysis Function
def calculate_business_impact(y_true, y_pred, model_name):
    """Calculate business impact of model predictions"""
    
    # Business costs (adjust based on your business context)
    cost_reject_good_customer = 500    # Lost revenue from rejecting good customer
    cost_accept_bad_customer = 5000    # Loss from bad customer defaulting
    
    # Calculate confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate costs
    cost_false_positives = fp * cost_reject_good_customer
    cost_false_negatives = fn * cost_accept_bad_customer
    total_cost = cost_false_positives + cost_false_negatives
    
    # Calculate baseline cost (if we accepted all customers)
    total_actual_positives = fn + tp
    baseline_cost = total_actual_positives * cost_accept_bad_customer
    
    # Calculate savings
    savings = baseline_cost - total_cost
    savings_percentage = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
    
    return {
        'model': model_name,
        'false_positives': fp,
        'false_negatives': fn,
        'cost_false_positives': cost_false_positives,
        'cost_false_negatives': cost_false_negatives,
        'total_cost': total_cost,
        'baseline_cost': baseline_cost,
        'savings': savings,
        'savings_percentage': savings_percentage
    }

# Test all models with enhanced evaluation
print("\n" + "="*80)
print("COMPREHENSIVE MODEL TESTING")
print("="*80)

all_results = []
all_predictions = {}
business_results = []

for name, model in models.items():
    print(f"\n Testing {name.replace('_', ' ').title()}...")
    
    try:
        # Make predictions
        if name == 'neural_network':
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_pred_proba = y_pred_proba.flatten()
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Display results
        print(f" Metrics:")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        if metrics.get('roc_auc'):
            print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        
        # Business interpretation
        print(f"Business Interpretation:")
        print(f"  → {metrics['precision']:.1%} of predicted high-risk are actually high-risk")
        print(f"  → {metrics['recall']:.1%} of actual high-risk customers are caught")
        print(f"  → {metrics['specificity']:.1%} of low-risk customers are correctly identified")
        
        # Calculate business impact
        business_impact = calculate_business_impact(y_test, y_pred, name)
        business_results.append(business_impact)
        
        print(f"Business Impact:")
        print(f"  False Positives: {business_impact['false_positives']} (good customers rejected)")
        print(f"  False Negatives: {business_impact['false_negatives']} (bad customers accepted)")
        print(f"  Total Cost: ${business_impact['total_cost']:,}")
        print(f"  Savings vs Baseline: ${business_impact['savings']:,} ({business_impact['savings_percentage']:.1f}%)")
        
        # Store results
        result = {
            'Model': name.replace('_', ' ').title(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'Specificity': metrics['specificity'],
            'ROC_AUC': metrics.get('roc_auc', None)
        }
        all_results.append(result)
        all_predictions[name] = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}
        
        print("-" * 60)
        
    except Exception as e:
        print(f"Error testing {name}: {e}")

# Create comprehensive results table
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('F1_Score', ascending=False)

# Display results table
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('results/comprehensive_results.csv', index=False)

# Business impact summary
business_df = pd.DataFrame(business_results)
business_df = business_df.sort_values('savings', ascending=False)

print("\n" + "="*80)
print(" BUSINESS IMPACT RANKING")
print("="*80)

for _, row in business_df.head(3).iterrows():
    print(f" {row['model'].replace('_', ' ').title()}")
    print(f"   Annual Savings: ${row['savings']:,}")
    print(f"   Savings %: {row['savings_percentage']:.1f}%")
    print(f"   Total Cost: ${row['total_cost']:,}")
    print()

# Save business results
business_df.to_csv('results/business_impact_analysis.csv', index=False)

# Find best models
best_f1_model = results_df.iloc[0]
best_business_model = business_df.iloc[0]

print("="*80)
print(" FINAL RECOMMENDATIONS")
print("="*80)
print(f" BEST TECHNICAL MODEL: {best_f1_model['Model']}")
print(f"   F1-Score: {best_f1_model['F1_Score']:.4f}")
print(f"   Accuracy: {best_f1_model['Accuracy']:.4f}")
print(f"   Precision: {best_f1_model['Precision']:.4f}")
print(f"   Recall: {best_f1_model['Recall']:.4f}")

print(f"\n BEST BUSINESS MODEL: {best_business_model['model'].replace('_', ' ').title()}")
print(f"   Annual Savings: ${best_business_model['savings']:,}")
print(f"   Savings Percentage: {best_business_model['savings_percentage']:.1f}%")
print(f"   Risk Reduction: {best_business_model['false_negatives']} fewer bad customers accepted")

# Create visualizations
print("\n Creating visualizations...")

# 1. Model Performance Comparison
plt.figure(figsize=(12, 8))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*1.5, results_df['Model'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('results/model_performance_comparison.png', dpi=300)
plt.show()

# 2. Business Impact Visualization
plt.figure(figsize=(10, 6))
plt.bar(business_df['model'], business_df['savings'], color='green', alpha=0.7)
plt.title('Annual Savings by Model')
plt.xlabel('Model')
plt.ylabel('Annual Savings ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/business_impact_comparison.png', dpi=300)
plt.show()

# 3. ROC Curves for models with probabilities
plt.figure(figsize=(10, 8))
for name, preds in all_predictions.items():
    if preds['y_pred_proba'] is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, preds['y_pred_proba'])
            auc_score = roc_auc_score(y_test, preds['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        except:
            pass

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300)
plt.show()

# 4. Confusion Matrix for Best Model
best_model_name = best_f1_model['Model'].lower().replace(' ', '_')
if best_model_name in all_predictions:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, all_predictions[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    plt.title(f'Confusion Matrix - {best_f1_model["Model"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/best_model_confusion_matrix.png', dpi=300)
    plt.show()

print("\n Testing complete!")
print(" Results saved to 'results/' directory:")

print(f"\n KEY INSIGHTS:")
print(f" Best Technical Performance: {best_f1_model['Model']} (F1: {best_f1_model['F1_Score']:.4f})")
print(f" Best Business Value: {best_business_model['model'].replace('_', ' ').title()} (Saves ${best_business_model['savings']:,}/year)")
