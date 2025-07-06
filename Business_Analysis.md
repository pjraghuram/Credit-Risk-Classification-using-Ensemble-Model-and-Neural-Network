# Business Impact Analysis - Step by Step Explanation

## 💰 Cost Structure

### **Business Costs (Example Values):**
- **False Positive Cost = $500** (Lost revenue from rejecting good customer)
- **False Negative Cost = $5,000** (Loss from bad customer defaulting)

### **Why These Costs?**
1. **Rejecting Good Customer ($500 loss):**
   - Lost interest income over loan term
   - Lost opportunity cost
   - Customer goes to competitor

2. **Accepting Bad Customer ($5,000 loss):**
   - Principal amount lost
   - Collection costs
   - Legal fees
   - Administrative costs

## 📊 Confusion Matrix Breakdown

```
                    ACTUAL
                Low Risk  High Risk
PREDICTED  Low   TN (✅)   FN (❌)
          High   FP (❌)   TP (✅)
```

**Where:**
- **TN (True Negative):** Correctly identified low-risk customer
- **TP (True Positive):** Correctly identified high-risk customer  
- **FP (False Positive):** Wrongly flagged good customer as high-risk
- **FN (False Negative):** Missed a bad customer (flagged as low-risk)

## 🧮 Cost Calculation

### **Model Costs:**
```python
# From confusion matrix
tn, fp, fn, tp = confusion_matrix.ravel()

# Calculate costs
cost_false_positives = fp × $500    # Good customers rejected
cost_false_negatives = fn × $5,000  # Bad customers accepted
total_model_cost = cost_false_positives + cost_false_negatives
```

### **Baseline Costs (No Model):**
```python
# If we accept ALL customers (no screening)
total_actual_bad_customers = fn + tp
baseline_cost = total_actual_bad_customers × $5,000
```

### **Savings Calculation:**
```python
savings = baseline_cost - total_model_cost
savings_percentage = (savings / baseline_cost) × 100
```

## 📈 Example Calculation

### **Example Data:**
- Test set: 1,000 customers
- Actual high-risk: 200 customers
- Actual low-risk: 800 customers

### **Model Results:**
- True Negatives (TN): 750 (correctly identified low-risk)
- False Positives (FP): 50 (wrongly rejected good customers)
- False Negatives (FN): 30 (missed bad customers)
- True Positives (TP): 170 (correctly identified high-risk)

### **Cost Calculation:**
```python
# Model costs
cost_false_positives = 50 × $500 = $25,000
cost_false_negatives = 30 × $5,000 = $150,000
total_model_cost = $25,000 + $150,000 = $175,000

# Baseline cost (accept all customers)
total_bad_customers = 30 + 170 = 200
baseline_cost = 200 × $5,000 = $1,000,000

# Savings
savings = $1,000,000 - $175,000 = $825,000
savings_percentage = ($825,000 / $1,000,000) × 100 = 82.5%
```

## 🎯 Business Interpretation

### **What This Means:**
- **Without Model:** Bank loses $1,000,000 (accepts all bad customers)
- **With Model:** Bank loses only $175,000 (catches most bad customers)
- **Annual Savings:** $825,000 (82.5% cost reduction)

### **Trade-offs:**
- **High Precision Model:** Fewer false positives (don't reject good customers)
- **High Recall Model:** Fewer false negatives (catch more bad customers)





