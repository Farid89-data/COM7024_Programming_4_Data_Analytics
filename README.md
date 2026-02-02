# Manchester Housing Data Analytics Project

## Executive Summary

This project demonstrates a comprehensive data analytics workflow on the Manchester housing dataset using Python. The analysis focuses on exploratory data analysis (EDA), data preprocessing, and statistical investigation of key housing market variables, particularly waterfront properties, floor space, and the relationship between build year and price.

## Project Objectives

As outlined in the COM7024 Programming for Data Analytics module (Arden University), this project aims to:

1. **Import and validate** the Manchester housing dataset with correct data type management
2. **Perform baseline statistical analysis** on all variables before preprocessing
3. **Execute data preprocessing** using appropriate cleaning, transformation, and quality improvement techniques
4. **Demonstrate improvements** through post-preprocessing statistical analysis
5. **Conduct focused investigations** on:
   - Waterfront property characteristics
   - Floor space distribution and impact
   - Correlation between build year (built) and price
6. **Provide evidence-based insights** with graphical representations suitable for executive reporting

## Dataset Overview

**File:** `Manchester_house_Dataset-3678.csv`

**Size:** 5,000+ housing records with 13 variables

**Key Variables:**
- **price** (target variable): Sale price in £
- **waterfront**: Binary indicator (0 = non-waterfront, 1 = waterfront property)
- **sqft, livingsqft, totalfloors**: Space measurements
- **built**: Year of construction
- **bedrooms, bathrooms, condition, grade, view, renovated**: Property characteristics

## Project Structure

```
├── data_analytics_project.py      # Main analysis script
├── Manchester_house_Dataset-3678.csv  # Input data
├── outputs/                       # Generated outputs directory
│   ├── cleaned_data.csv          # Processed dataset
│   ├── statistical_summary.csv    # Pre/post preprocessing stats
│   └── visualizations/           # EDA plots
└── README.md                      # This file
```

## Methodology

### Phase 1: Data Import & Initial Analysis
- Load CSV using pandas with proper encoding
- Verify data types and dimensionality
- Generate descriptive statistics (mean, median, std, quartiles, skewness, kurtosis)
- Identify missing values, outliers, and data quality issues

### Phase 2: Data Preprocessing
**Techniques applied based on data characteristics:**

1. **Missing Values**
   - Strategy: Domain-appropriate imputation (median for numerical, mode for categorical)
   - Rationale: Preserves distribution while handling sparse data

2. **Outliers & Anomalies**
   - Detection: IQR method and z-score analysis
   - Treatment: Capping at 95th/5th percentile (preserves data while reducing extreme values)
   - Justification: Property data often has legitimate extreme values; removal would lose information

3. **Data Normalization**
   - Technique: Min-Max scaling for price-related variables
   - Purpose: Enables fair comparison across variables with different ranges
   - Applied to: Continuous numeric variables

4. **Duplicate Removal**
   - Identification: Exact row duplicates
   - Action: Removal of identified duplicates with documentation

5. **Data Type Corrections**
   - Ensure binary variables are categorical (waterfront, renovated)
   - Convert temporal variables appropriately

### Phase 3: Exploratory Data Analysis
**Focused investigations with visualizations:**

1. **Waterfront Property Analysis**
   - Comparison of price distributions (waterfront vs. non-waterfront)
   - Count and percentage of waterfront properties
   - Statistical significance testing (t-test)
   - Visualization: Box plots, distribution plots

2. **Floor Space Analysis**
   - Relationship between floor space variables (sqft, livingsqft, totalfloors)
   - Impact on price through correlation analysis
   - Segmentation by floor space quartiles
   - Visualization: Scatter plots, regression plots

3. **Build Year vs. Price Correlation**
   - Pearson correlation coefficient calculation
   - Temporal trend analysis (price trends by decade)
   - Visualization: Scatter plot with trend line, heatmap

### Phase 4: Statistical Validation
- Pre-processing vs. post-processing comparison
- Impact metrics on data quality improvements
- Evidence of effectiveness for each preprocessing technique

## Technical Stack

- **Python 3.8+**
- **Libraries:** pandas, numpy, matplotlib, seaborn, scipy.stats, scikit-learn
- **Coding Standards:** PEP 8 compliant, reproducible, commented

## Key Findings Structure

The analysis will produce:

1. **Quantitative Evidence**
   - Descriptive statistics tables
   - Correlation matrices
   - Regression coefficients

2. **Visual Evidence**
   - Distribution plots (histograms, KDEs)
   - Relationship plots (scatter, regression)
   - Comparative plots (box plots, violin plots)
   - Correlation heatmaps

3. **Statistical Tests**
   - Normality tests (Shapiro-Wilk)
   - Relationship significance (Pearson correlation p-values)
   - Group differences (independent t-tests)

## Academic Standards Compliance

This project adheres to:
- **Arden University academic writing standards**: Formal tone, evidence-based claims, proper referencing
- **PEP 8 Python coding conventions**: Consistent style, meaningful variable names, comprehensive documentation
- **Reproducibility**: All analysis is script-based with seed control for statistical operations
- **Integrity**: No fabricated data; all visualizations and statistics derive from actual dataset values

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Execute analysis
python data_analytics_project.py

# Output files will be created in ./outputs/ directory
```

## Output Files Generated

1. **cleaned_data.csv** - Final preprocessed dataset ready for predictive modeling
2. **statistical_summary.csv** - Comparison of pre/post preprocessing statistics
3. **waterfront_analysis.png** - Waterfront property comparative analysis
4. **floorspace_analysis.png** - Floor space impact visualizations
5. **builtyear_price_correlation.png** - Build year and price relationship
6. **data_quality_improvements.png** - Pre/post preprocessing improvements

## References

The preprocessing methodologies and EDA techniques referenced in this project follow established data science literature including:
- Pyle, D. (1999). Data Preparation for Data Mining. Morgan Kaufmann.
- Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
- Hair, J. F., et al. (2019). Multivariate Data Analysis (8th ed.).

## Author Notes

This project demonstrates practical application of core data analytics concepts from the COM7024 module, including:
- **LO1**: Programming concepts (Python structures, control flow, data handling)
- **LO2**: Data manipulation and analysis techniques (pandas, statistical methods)
- **LO3**: Critical evaluation of results and methodological choices
- **LO4**: Appropriate tool selection and digital capability application

Each analytical step includes justification for chosen methods and interpretation of results suitable for non-technical stakeholders (estate managers).

---
*Manchester Housing Data Analytics Project | COM7024 Programming for Data Analytics | Arden University*
