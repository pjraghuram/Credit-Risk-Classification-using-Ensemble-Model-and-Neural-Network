# Credit Risk Classification - Data Description

## 📊 Dataset Overview

This dataset contains customer information and payment history for credit risk classification. The goal is to predict whether a customer is **high-risk (1)** or **low-risk (0)** based on their demographic and financial information.

## 📁 Files

### customer_data.csv
Contains customer demographic and basic financial information.

**Records:** 1,125 customers  
**Features:** 13 columns

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `id` | Integer | Unique customer identifier | 54982665 - 59004779 |
| `label` | Integer | **Target variable** (0 = Low Risk, 1 = High Risk) | 0, 1 |
| `fea_1` | Integer | Customer feature 1 (likely credit score tier) | 1-7 scale |
| `fea_2` | Float | Customer feature 2 (financial metric) | 1116.5 - 1481 |
| `fea_3` | Integer | Customer feature 3 (account type/category) | 1-3 scale |
| `fea_4` | Float | Customer feature 4 (likely income/credit limit) | 15,000 - 1,200,000 |
| `fea_5` | Integer | Customer feature 5 (binary indicator) | 1-2 scale |
| `fea_6` | Integer | Customer feature 6 (age/tenure related) | 3-16 range |
| `fea_7` | Integer | Customer feature 7 (risk indicator) | -1 to 10 range |
| `fea_8` | Integer | Customer feature 8 (normalized score) | 64-115 range |
| `fea_9` | Integer | Customer feature 9 (rating/tier) | 1-5 scale |
| `fea_10` | Float | Customer feature 10 (financial capacity) | 60,000 - 650,070 |
| `fea_11` | Float | Customer feature 11 (financial ratio) | 1 - 707.11 |

### payment_data.csv
Contains payment history and behavior patterns for each customer.

**Records:** 8,250 payment records  
**Features:** 12 columns

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `id` | Integer | Customer identifier (matches customer_data.csv) | Same as customer_data |
| `OVD_t1` | Integer | Overdue amount in period t1 | 0 - 34 |
| `OVD_t2` | Integer | Overdue amount in period t2 | 0 - 31 |
| `OVD_t3` | Integer | Overdue amount in period t3 | 0 - 23 |
| `OVD_sum` | Integer | **Total overdue amount** | 0 - 31,500 |
| `pay_normal` | Integer | Payment behavior indicator (0=worst, 36=best) | 0-36 scale |
| `prod_code` | Integer | Product code (loan/credit type) | 0-27 range |
| `prod_limit` | Float | Product credit limit | Varies, many nulls |
| `update_date` | String | Record update date | DD/MM/YYYY format |
| `new_balance` | Float | Current account balance | -40,303 to 163,211,958 |
| `highest_balance` | Float | Highest historical balance | Varies, some nulls |
| `report_date` | String | Report generation date | DD/MM/YYYY format |

### Missing Values Analysis
- **customer_data.csv**: 149 missing values in `fea_2` (13.2%)
- **payment_data.csv**: 
  - `prod_limit`: 6,118 missing values (74.1%)
  - `highest_balance`: 409 missing values (5.0%)
  - `report_date`: 1,114 missing values (13.5%)

## 🔧 Feature Engineering Opportunities

### 1. **Payment Behavior Features (High Impact)**
- **Overdue patterns**: `total_overdue`, `max_overdue`, `avg_overdue`
- **Risk indicators**: `has_overdue`, `high_overdue`, `severe_overdue`
- **Payment consistency**: `payment_volatility`, `balance_volatility`

### 2. **Financial Stability Features**
- **Risk ratios**: `overdue_to_income_ratio`, `balance_to_income_ratio`
- **Utilization rates**: `balance_utilization`, `credit_utilization`
- **Stability scores**: `financial_stability_score`, `composite_risk_score`

### 3. **Temporal Features**
- **Trends**: Payment behavior deterioration over time
- **Seasonality**: Monthly payment patterns
- **Recency**: Recent vs historical behavior weighting

## 🎯 Business Context

### Cost Structure
- **False Negative Cost**: $5,000 (accepting bad customer who defaults)
- **False Positive Cost**: $500 (rejecting good customer - lost revenue)
- **Cost Ratio**: Bad customers cost 10x more than rejected good customers

## 💡 Preprocessing Pipeline

### 1. **Data Cleaning**
- Handle missing values (KNN imputation for continuous, mode for categorical)
- Remove outliers using IQR method
- Validate data types and ranges

### 2. **Feature Engineering**
- Create payment behavior aggregations
- Calculate financial ratios and risk indicators
- Generate composite risk scores

### 3. **Preprocessing Steps**
- **Scaling**: StandardScaler for numerical features
- **Dimensionality Reduction**: PCA (99% variance retained)
- **Class Balancing**: SMOTE oversampling for minority class
- **Train-Test Split**: 80:20 stratified split

### Output Files
- **Models**: Saved in `models/` directory
- **Results**: Comprehensive evaluation in `results/`
- **Visualizations**: ROC curves, confusion matrices, business impact charts

