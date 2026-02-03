# ❗👨‍🏫 CODE GUIDE FOR MANCHESTER HOUSING ANALYSIS

## Overview
This guide breaks down the complete Python data analytics project into logical phases with explanations of each step.

## </> Project Structure

```
Phase 1: Data Import & Exploration
  ↓
Phase 2: Pre-Processing Quality Assessment
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
Phase 8: Data Quality Improvements Visualization
  ↓
Phase 9: Output & Report Generation
```

---

## 📥 PHASE 1: DATA IMPORT & EXPLORATION

### Step 1.1: Load Dataset
```python
import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv('Manchester_house_Dataset-3678.csv')

# Verify dimensions
print(f"Shape: {df.shape}")  # Output: (n_rows, n_columns)
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```
**Purpose:** Load data into memory and verify successful import

**Why this matters:** 
- Confirms file path is correct
- Shows data volume for memory management
- Ensures encoding compatibility

### Step 1.2: Inspect Data Structure
```python
# Check column data types
print(df.dtypes)

# Display first 5 rows
print(df.head())

# Get summary info
print(df.info())
```
**Purpose:** Understand data structure before analysis

**Key Insights:**
- Identifies numeric vs categorical columns
- Reveals current data types (may need conversion)
- Shows immediate data patterns

---

## PHASE 2: PRE-PROCESSING QUALITY ASSESSMENT

### 🧹 Step 2.1: Identify Missing Values
```python
# Count missing values by column
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_data)
```
**Purpose:** Quantify data quality issues

**Interpretation:**
- HIGH missing percentage (>20%): May need removal or special handling
- LOW missing percentage (<5%): Can impute safely
- NO missing values: Column is complete

### 👀 Step 2.2: Check for Duplicates
```python
# Count duplicate rows
duplicates = df.duplicated().sum()
print(f"Total duplicates: {duplicates}")

# Find exact duplicate rows
duplicate_rows = df[df.duplicated(keep=False)]
```
**Purpose:** Detect redundant records

**Why remove duplicates:**
- Artificial inflation of sample size
- Skewed statistical analysis
- Double-counting in aggregations

### 📈 Step 2.3: Generate Baseline Statistics
```python
# Comprehensive statistical summary
stats_pre = df.describe().T  # Transpose for better readability

# Add advanced statistics
stats_pre['Skewness'] = df.skew()  # Measure of asymmetry
stats_pre['Kurtosis'] = df.kurtosis()  # Measure of tail heaviness

print(stats_pre)
```
**Key Metrics Explained:**
- **Mean:** Average value (sensitive to outliers)
- **Median:** Middle value (robust to outliers)
- **Std:** Standard deviation (spread around mean)
- **Skewness:** Asymmetry of distribution (-1 = left-skewed, +1 = right-skewed)
- **Kurtosis:** Tail heaviness (high = more extreme values)

---

## 🧼 PHASE 3: DATA PREPROCESSING & CLEANING

### Step 3.1: Handle Missing Values
```python
# Strategy 1: Median imputation (for numeric data)
df_cleaned = df.copy()
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df_cleaned[col].isnull().any():
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)
        print(f"Imputed {col} with median: {median_val:.2f}")
```
**Why Median over Mean:**
- Median is ROBUST to outliers
- Mean can be skewed by extreme values
- Better preserves data distribution

### Step 3.2: Remove Duplicates
```python
# Remove exact duplicates
df_cleaned = df_cleaned.drop_duplicates()
print(f"Rows after deduplication: {len(df_cleaned)}")
```
**Purpose:** Ensure each observation is unique

### Step 3.3: Treat Outliers
```python
# IQR (Interquartile Range) Method
for col in numeric_cols:
    Q1 = df_cleaned[col].quantile(0.05)  # 5th percentile
    Q3 = df_cleaned[col].quantile(0.95)  # 95th percentile
    
    # Cap values outside range
    df_cleaned[col] = df_cleaned[col].clip(lower=Q1, upper=Q3)
```
**Why Capping vs Removal:**
- **Property data:** Extreme values often legitimate (luxury properties, large estates)
- **Capping:** Preserves information while reducing impact
- **Removal:** Loses valuable data points

### Step 3.4: Normalize Numerical Data
```python
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Select columns to normalize
cols_to_normalize = ['price', 'sqft', 'livingsqft']

# Fit and transform
for col in cols_to_normalize:
    normalized = scaler.fit_transform(df_cleaned[[col]])
    # normalized now ranges from 0 to 1
```
**Purpose:** Scale variables to comparable ranges

**When to normalize:**
- Machine learning models (distance-based algorithms)
- When variables have different units
- Neural networks and deep learning

### Step 3.5: Correct Data Types
```python
# Convert to categorical for binary variables
df_cleaned['waterfront'] = df_cleaned['waterfront'].astype('category')
df_cleaned['renovated'] = df_cleaned['renovated'].astype('category')
```
**Benefits:**
- Reduces memory usage
- Enables proper categorical analysis
- Prevents accidental arithmetic on categories

---

## PHASE 4: POST-PROCESSING STATISTICAL ANALYSIS

### Step 4.1: Compare Pre vs Post Statistics
```python
# Post-processing statistics
stats_post = df_cleaned.describe().T
stats_post['Skewness'] = df_cleaned.skew()
stats_post['Kurtosis'] = df_cleaned.kurtosis()

# Compare improvements
improvements = pd.DataFrame({
    'Variable': numeric_cols,
    'Skewness_Before': stats_pre.loc[numeric_cols, 'Skewness'],
    'Skewness_After': stats_post.loc[numeric_cols, 'Skewness'],
})
print(improvements)
```
**Success Indicators:**
- Skewness closer to 0 (more normally distributed)
- Std Dev reduced (fewer extreme values)
- Statistics more stable

---

## PHASE 5: WATERFRONT PROPERTIES INVESTIGATION

### Step 5.1: Frequency Analysis
```python
# Count properties by waterfront status
waterfront_counts = df_cleaned['waterfront'].value_counts()
waterfront_pct = df_cleaned['waterfront'].value_counts(normalize=True) * 100

print(f"Non-waterfront: {waterfront_counts[0]} ({waterfront_pct[0]:.1f}%)")
print(f"Waterfront: {waterfront_counts[1]} ({waterfront_pct[1]:.1f}%)")
```

### Step 5.2: Price Comparison
```python
# Separate by waterfront status
non_waterfront_prices = df_cleaned[df_cleaned['waterfront'] == 0]['price']
waterfront_prices = df_cleaned[df_cleaned['waterfront'] == 1]['price']

# Compare statistics
print(f"Non-waterfront mean: £{non_waterfront_prices.mean():,.2f}")
print(f"Waterfront mean: £{waterfront_prices.mean():,.2f}")

# Calculate premium
premium = ((waterfront_prices.mean() - non_waterfront_prices.mean()) 
           / non_waterfront_prices.mean() * 100)
print(f"Price premium: {premium:.2f}%")
```

### Step 5.3: Statistical Significance Testing
```python
from scipy import stats

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(waterfront_prices, non_waterfront_prices)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4e}")

# Interpretation
if p_value < 0.05:
    print("✓ SIGNIFICANT DIFFERENCE (reject null hypothesis)")
else:
    print("✗ No significant difference (fail to reject null hypothesis)")
```
**Understanding p-values:**
- p < 0.05: STATISTICALLY SIGNIFICANT (5% chance of occurring by random chance)
- p < 0.001: HIGHLY SIGNIFICANT (rare to occur randomly)
- p > 0.05: Not significant (could easily occur by chance)

### Step 5.4: Create Visualization
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution comparison
axes[0, 0].hist(non_waterfront_prices/1000, bins=50, alpha=0.6, label='Non-Waterfront')
axes[0, 0].hist(waterfront_prices/1000, bins=50, alpha=0.6, label='Waterfront')
axes[0, 0].set_xlabel('Price (£1000s)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Box plot comparison
data_to_plot = [non_waterfront_prices/1000, waterfront_prices/1000]
axes[0, 1].boxplot(data_to_plot, labels=['Non-Waterfront', 'Waterfront'])
axes[0, 1].set_ylabel('Price (£1000s)')

plt.tight_layout()
plt.savefig('waterfront_analysis.png', dpi=300)
```

---

## PHASE 6: FLOOR SPACE ANALYSIS

### Step 6.1: Correlation Analysis
```python
from scipy.stats import pearsonr

floor_space_vars = ['sqft', 'livingsqft', 'totalfloors']

for var in floor_space_vars:
    # Calculate Pearson correlation
    corr, p_val = pearsonr(df_cleaned[var], df_cleaned['price'])
    
    print(f"{var} vs price:")
    print(f"  Correlation: {corr:.4f}")
    print(f"  P-value: {p_val:.4e}")
    
    # Interpretation
    if abs(corr) > 0.7:
        strength = "STRONG"
    elif abs(corr) > 0.4:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    print(f"  Strength: {strength}\n")
```
**Correlation Interpretation:**
- **r = 0.0 to 0.3:** Weak correlation
- **r = 0.3 to 0.7:** Moderate correlation
- **r = 0.7 to 1.0:** Strong correlation
- **Negative values:** Inverse relationships

### Step 6.2: Quartile Segmentation
```python
# Create floor space quartiles
df_cleaned['sqft_quartile'] = pd.qcut(df_cleaned['sqft'], q=4, 
                                      labels=['Q1 (Smallest)', 'Q2', 'Q3', 'Q4 (Largest)'])

# Analyze price by quartile
quartile_analysis = df_cleaned.groupby('sqft_quartile')['price'].agg([
    'count', 'mean', 'median', 'std'
]).round(2)

print(quartile_analysis)
```

### Step 6.3: Visualization
```python
# Scatter plot with trend line
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df_cleaned['sqft'], df_cleaned['price']/1000, alpha=0.5, s=20)

# Add trend line
z = np.polyfit(df_cleaned['sqft'], df_cleaned['price']/1000, 1)
p = np.poly1d(z)
ax.plot(df_cleaned['sqft'].sort_values(), p(df_cleaned['sqft'].sort_values()), 
        "r--", alpha=0.8, linewidth=2)

ax.set_xlabel('Floor Space (sqft)')
ax.set_ylabel('Price (£1000s)')
ax.set_title(f'Floor Space vs Price (r={corr:.3f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('floorspace_analysis.png', dpi=300)
```

---

## PHASE 7: BUILD YEAR VS PRICE CORRELATION

### Step 7.1: Temporal Analysis
```python
# Year range and statistics
print(f"Earliest: {df_cleaned['built'].min()}")
print(f"Latest: {df_cleaned['built'].max()}")
print(f"Mean: {df_cleaned['built'].mean():.1f}")
print(f"Median: {df_cleaned['built'].median():.1f}")
```

### Step 7.2: Correlation Calculation
```python
# Pearson correlation
corr_built, p_val_built = pearsonr(df_cleaned['built'], df_cleaned['price'])

print(f"Correlation: {corr_built:.4f}")
print(f"P-value: {p_val_built:.4e}")

# Interpret direction
if corr_built > 0:
    interpretation = "Positive: Newer buildings tend to be MORE expensive"
else:
    interpretation = "Negative: Newer buildings tend to be LESS expensive"
    
print(interpretation)
```

### Step 7.3: Decade Analysis
```python
# Group by decade
df_cleaned['build_decade'] = (df_cleaned['built'] // 10 * 10).astype(int)

# Analyze by decade
decade_analysis = df_cleaned.groupby('build_decade')['price'].agg([
    'count', 'mean', 'median'
]).round(2)

print(decade_analysis)
```

### Step 7.4: Visualization
```python
# Scatter plot with trend
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter with trend line
axes[0, 0].scatter(df_cleaned['built'], df_cleaned['price']/1000, alpha=0.4, s=20)
z = np.polyfit(df_cleaned['built'], df_cleaned['price']/1000, 1)
p = np.poly1d(z)
axes[0, 0].plot(sorted(df_cleaned['built'].unique()), 
                p(sorted(df_cleaned['built'].unique())), 
                "r--", alpha=0.8, linewidth=2.5)
axes[0, 0].set_xlabel('Year Built')
axes[0, 0].set_ylabel('Price (£1000s)')
axes[0, 0].set_title(f'Build Year vs Price (r={corr_built:.4f})')

# Mean price by decade
decade_means = df_cleaned.groupby('build_decade')['price'].mean() / 1000
axes[0, 1].bar(range(len(decade_means)), decade_means.values)
axes[0, 1].set_xticks(range(len(decade_means)))
axes[0, 1].set_xticklabels(decade_means.index.astype(str), rotation=45)
axes[0, 1].set_ylabel('Mean Price (£1000s)')
axes[0, 1].set_xlabel('Build Decade')
axes[0, 1].set_title('Average Price by Decade')

# Count by decade
decade_counts = df_cleaned.groupby('build_decade').size()
axes[1, 0].bar(range(len(decade_counts)), decade_counts.values)
axes[1, 0].set_xticks(range(len(decade_counts)))
axes[1, 0].set_xticklabels(decade_counts.index.astype(str), rotation=45)
axes[1, 0].set_ylabel('Number of Properties')
axes[1, 0].set_xlabel('Build Decade')

plt.tight_layout()
plt.savefig('builtyear_analysis.png', dpi=300)
```

---

## PHASE 8: SAVE OUTPUTS

### Step 8.1: Save Cleaned Data
```python
# Save processed dataset
df_cleaned.to_csv('outputs/cleaned_data.csv', index=False)
print(f"✓ Saved {len(df_cleaned)} rows to cleaned_data.csv")
```

### Step 8.2: Save Statistical Summaries
```python
# Pre and post statistics
stats_pre.to_csv('outputs/statistical_summary_pre.csv')
stats_post.to_csv('outputs/statistical_summary_post.csv')
```

### Step 8.3: Generate Report
```python
# Create text summary
report = f"""
MANCHESTER HOUSING ANALYSIS REPORT
{'='*60}
Dataset: {len(df_cleaned)} properties
Waterfront Premium: {premium:.1f}%
Floor Space Correlation: {corr:.4f}
Build Year Correlation: {corr_built:.4f}
"""

with open('outputs/analysis_report.txt', 'w') as f:
    f.write(report)
```

---

## KEY PYTHON CONCEPTS USED

### 1. Data Structures
- **DataFrame:** 2D table with labeled columns and rows
- **Series:** 1D labeled array (column or row)
- **NumPy arrays:** Numerical computing foundation

### 2. Data Manipulation
- **`.loc[]`:** Label-based indexing
- **`.iloc[]`:** Position-based indexing
- **`.fillna()`:** Handle missing values
- **`.groupby()`:** Group and aggregate data
- **`.describe()`:** Comprehensive statistics

### 3. Statistical Methods
- **`.mean()`:** Average
- **`.median()`:** Middle value
- **`.std()`:** Standard deviation
- **`pearsonr()`:** Correlation coefficient
- **`ttest_ind()`:** Statistical hypothesis testing

### 4. Visualization
- **matplotlib:** Low-level plotting library
- **seaborn:** Statistical visualization
- **plt.scatter():** Scatter plots
- **plt.histogram():** Distribution plots
- **plt.boxplot():** Box and whisker plots

### 5. File Operations
- **`.read_csv()`:** Load CSV files
- **`.to_csv()`:** Save to CSV
- **`open()`:** Open files for writing

---

## BEST PRACTICES

### Code Organization
✓ Comment code extensively
✓ Use descriptive variable names
✓ Group related code into sections
✓ Create reusable functions

### Data Handling
✓ Always work with a copy when modifying original
✓ Validate data after each transformation
✓ Document preprocessing decisions
✓ Preserve original data for reference

### Statistical Analysis
✓ State hypotheses before testing
✓ Check assumptions before using statistical tests
✓ Report both p-values and effect sizes
✓ Use visualizations to support findings

### Visualization
✓ Use clear titles and axis labels
✓ Choose appropriate chart types
✓ Include legends for multiple series
✓ Ensure readability (font size, colors)

---

## TROUBLESHOOTING COMMON ERRORS

### Error: "FileNotFoundError"
**Solution:** Ensure CSV file is in same directory as script

### Error: "SettingWithCopyWarning"
**Solution:** Use `.copy()` when creating subsets

### Error: "ValueError: 'x' contains NaN"
**Solution:** Handle missing values before statistical operations

### Error: "Memory usage too high"
**Solution:** Use appropriate data types, filter unnecessary columns

---
