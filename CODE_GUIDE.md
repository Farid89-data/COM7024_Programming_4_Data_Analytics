# ❗👨‍🏫 Code Guide: Manchester Housing Analysis

## Overview

This document explains the full Python data analytics workflow used in the Manchester Housing Analysis project.  
The project is organised into clear, sequential phases, with each step described in terms of **what the code does**, **why it is necessary**, and **how it contributes to the overall analysis**.

The guide is intended to support both technical understanding and academic assessment.

---

## </> Project Workflow

```
Phase 1: Data Import & Exploration
  ↓
Phase 2: Pre-Processing Data Quality Assessment
  ↓
Phase 3: Data Preprocessing & Cleaning
  ↓
Phase 4: Post-Processing Statistical Analysis
  ↓
Phase 5: Waterfront Properties Investigation
  ↓
Phase 6: Floor Space Analysis
  ↓
Phase 7: Build Year vs Price Correlation
  ↓
Phase 8: Data Quality Improvements Visualisation
  ↓
Phase 9: Output & Report Generation
```

Each phase builds on the previous one, ensuring that all statistical conclusions are based on clean, validated data.

---

## 📥 Phase 1: Data Import & Exploration

### Step 1.1: Load the Dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv('Manchester_house_Dataset-3678.csv')

print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

**Purpose**

- Load the dataset into memory  
- Confirm that the file path and format are correct  
- Check dataset size and memory usage  

**Why this matters**

Early verification avoids downstream errors and helps assess whether memory optimisation is required.

---

### Step 1.2: Inspect the Data Structure

```python
print(df.dtypes)
print(df.head())
print(df.info())
```

**Purpose**

- Identify numerical and categorical variables  
- Check for incorrect or unexpected data types  
- Get an initial sense of value ranges and patterns  

This step informs all later preprocessing decisions.

---

## 🔨 Phase 2: Pre‑Processing Data Quality Assessment

### Step 2.1: Missing Value Assessment

```python
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})

print(missing_data)
```

**How to interpret the results**

- **More than 20% missing** → consider removal or special handling  
- **Less than 5% missing** → safe to impute  
- **No missing values** → no action required  

Quantifying missingness ensures that cleaning decisions are justified rather than arbitrary.

---

### Step 2.2: Duplicate Detection

```python
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")

duplicate_rows = df[df.duplicated(keep=False)]
```

**Why duplicates matter**

- Artificially inflate sample size  
- Bias descriptive statistics  
- Distort hypothesis tests  

Duplicates are removed later once their presence is confirmed.

---

### Step 2.3: Baseline Statistical Summary

```python
stats_pre = df.describe().T
stats_pre['Skewness'] = df.skew()
stats_pre['Kurtosis'] = df.kurtosis()

print(stats_pre)
```

**Key metrics**

- **Mean vs median** → sensitivity to outliers  
- **Skewness** → distribution asymmetry  
- **Kurtosis** → presence of extreme values  

These statistics provide a baseline for evaluating preprocessing improvements.

---

## 🧹 Phase 3: Data Preprocessing & Cleaning

### Step 3.1: Missing Value Imputation

```python
df_cleaned = df.copy()
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df_cleaned[col].isnull().any():
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)
        print(f"{col} imputed with median: {median_val:.2f}")
```

**Why median imputation is used**

- Robust to outliers  
- Better suited to housing price data  
- Preserves distribution shape more effectively than the mean  

---

### Step 3.2: Duplicate Removal

```python
df_cleaned = df_cleaned.drop_duplicates()
print(f"Rows after deduplication: {len(df_cleaned)}")
```

This ensures that each row represents a unique property.

---

### Step 3.3: Outlier Treatment (Capping)

```python
for col in numeric_cols:
    lower = df_cleaned[col].quantile(0.05)
    upper = df_cleaned[col].quantile(0.95)
    df_cleaned[col] = df_cleaned[col].clip(lower, upper)
```

**Why capping is preferred over removal**

- Extreme property values are often genuine  
- Capping limits distortion without discarding data  
- Maintains sample size for statistical testing  

---

### Step 3.4: Normalisation (Where Required)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols_to_normalize = ['price', 'sqft', 'livingsqft']

for col in cols_to_normalize:
    df_cleaned[col + '_scaled'] = scaler.fit_transform(df_cleaned[[col]])
```

**When normalisation is useful**

- Distance‑based machine learning models  
- Features measured on very different scales  
- Neural networks  

---

### Step 3.5: Correct Data Types

```python
df_cleaned['waterfront'] = df_cleaned['waterfront'].astype('category')
df_cleaned['renovated'] = df_cleaned['renovated'].astype('category')
```

**Benefits**

- Reduced memory usage  
- Clearer semantic meaning  
- Prevents invalid numerical operations  

---

## 📈 Phase 4: Post‑Processing Statistical Analysis

### Step 4.1: Pre‑ vs Post‑Processing Comparison

```python
stats_post = df_cleaned.describe().T
stats_post['Skewness'] = df_cleaned.skew()
stats_post['Kurtosis'] = df_cleaned.kurtosis()

improvements = pd.DataFrame({
    'Variable': numeric_cols,
    'Skewness_Before': stats_pre.loc[numeric_cols, 'Skewness'],
    'Skewness_After': stats_post.loc[numeric_cols, 'Skewness'],
})

print(improvements)
```

**Indicators of successful preprocessing**

- Skewness values closer to zero  
- Reduced influence of extreme values  
- More stable descriptive statistics  

---

## 🌊 Phase 5: Waterfront Properties Investigation

### Step 5.1: Frequency Analysis

```python
counts = df_cleaned['waterfront'].value_counts()
percentages = df_cleaned['waterfront'].value_counts(normalize=True) * 100

print(counts)
print(percentages.round(1))
```

---

### Step 5.2: Price Comparison

```python
non_waterfront_prices = df_cleaned[df_cleaned['waterfront'] == 0]['price']
waterfront_prices = df_cleaned[df_cleaned['waterfront'] == 1]['price']

premium = ((waterfront_prices.mean() - non_waterfront_prices.mean())
           / non_waterfront_prices.mean() * 100)

print(f"Waterfront price premium: {premium:.2f}%")
```

---

### Step 5.3: Hypothesis Testing

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(waterfront_prices, non_waterfront_prices)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4e}")
```

**Interpretation**

- **p < 0.05** → statistically significant difference  
- **p < 0.001** → highly significant  

---

## 📐 Phase 6: Floor Space Analysis

### Step 6.1: Correlation Testing

```python
from scipy.stats import pearsonr

floor_space_vars = ['sqft', 'livingsqft', 'totalfloors']

for var in floor_space_vars:
    r, p = pearsonr(df_cleaned[var], df_cleaned['price'])
    print(f"{var}: r={r:.3f}, p={p:.3e}")
```

---

### Step 6.2: Quartile Segmentation

```python
df_cleaned['sqft_quartile'] = pd.qcut(
    df_cleaned['sqft'],
    q=4,
    labels=['Q1 (Smallest)', 'Q2', 'Q3', 'Q4 (Largest)']
)

quartile_analysis = df_cleaned.groupby('sqft_quartile')['price'].agg(
    ['count', 'mean', 'median', 'std']
).round(2)

print(quartile_analysis)
```

This highlights how prices change across increasing property sizes.

---

## 🕰 Phase 7: Build Year vs Price Correlation

### Step 7.1: Temporal Summary

```python
print(f"Earliest: {df_cleaned['built'].min()}")
print(f"Latest: {df_cleaned['built'].max()}")
print(f"Mean: {df_cleaned['built'].mean():.1f}")
print(f"Median: {df_cleaned['built'].median():.1f}")
```

---

### Step 7.2: Correlation Analysis

```python
corr_built, p_val_built = pearsonr(df_cleaned['built'], df_cleaned['price'])

print(f"Correlation: {corr_built:.4f}")
print(f"P-value: {p_val_built:.4e}")
```

---

### Step 7.3: Decade‑Level Analysis

```python
df_cleaned['build_decade'] = (df_cleaned['built'] // 10 * 10).astype(int)

decade_analysis = df_cleaned.groupby('build_decade')['price'].agg(
    ['count', 'mean', 'median']
).round(2)

print(decade_analysis)
```

---

## 💾 Phase 8: Saving Outputs

```python
df_cleaned.to_csv('outputs/cleaned_data.csv', index=False)

stats_pre.to_csv('outputs/statistical_summary_pre.csv')
stats_post.to_csv('outputs/statistical_summary_post.csv')
```

---

## 🧠 Core Concepts Demonstrated

- pandas DataFrames and Series  
- Data validation and preprocessing  
- Descriptive and inferential statistics  
- Correlation and hypothesis testing  
- Reproducible analytical workflows  

---

## ✅ Best Practices Followed

- Always work on a copy of the original dataset  
- Validate results after each major transformation  
- Combine statistical tests with visual evidence  
- Document assumptions and decisions clearly  

---

## 🧯 Common Issues and Fixes

| Issue | Solution |
|-----|---------|
| File not found | Check filename and directory |
| NaN values in analysis | Impute missing values first |
| `SettingWithCopyWarning` | Use `.copy()` explicitly |
| High memory usage | Optimise data types |

---
