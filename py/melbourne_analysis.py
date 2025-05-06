import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv('melb_data.csv')

# 1. Analyze features and data types
print("=== Feature Analysis ===")
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())

# 2. Identify outliers using different methods
def identify_outliers(df, column):
    # Boxplot method
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.savefig(f'boxplot_{column}.png')
    plt.close()
    
    # Z-score method
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers_z = df[column][z_scores > 3]
    
    # IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = df[column][(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    
    return {
        'z_score_outliers': len(outliers_z),
        'iqr_outliers': len(outliers_iqr)
    }

# Analyze numeric columns for outliers
numeric_columns = df.select_dtypes(include=[np.number]).columns
outlier_analysis = {}

for column in numeric_columns:
    outlier_analysis[column] = identify_outliers(df, column)

# 3. Identify features without outliers
features_without_outliers = []
for column, analysis in outlier_analysis.items():
    if analysis['z_score_outliers'] == 0 and analysis['iqr_outliers'] == 0:
        features_without_outliers.append(column)

# 4. Handle outliers in BuildingArea
def handle_building_area_outliers(df):
    # Calculate statistics
    min_val = df['BuildingArea'].min()
    max_val = df['BuildingArea'].max()
    mean_val = df['BuildingArea'].mean()
    median_val = df['BuildingArea'].median()
    
    # Create copies for different methods
    df_min = df.copy()
    df_max = df.copy()
    df_mean = df.copy()
    df_median = df.copy()
    
    # Identify outliers using IQR method
    Q1 = df['BuildingArea'].quantile(0.25)
    Q3 = df['BuildingArea'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df['BuildingArea'] < (Q1 - 1.5 * IQR)) | (df['BuildingArea'] > (Q3 + 1.5 * IQR))
    
    # Fill outliers with different methods
    df_min.loc[outlier_mask, 'BuildingArea'] = min_val
    df_max.loc[outlier_mask, 'BuildingArea'] = max_val
    df_mean.loc[outlier_mask, 'BuildingArea'] = mean_val
    df_median.loc[outlier_mask, 'BuildingArea'] = median_val
    
    return df_min, df_max, df_mean, df_median

# 5. Handle outliers in Landsize and Price
def handle_landsize_price_outliers(df):
    # Create copies for different methods
    df_prev = df.copy()
    df_next = df.copy()
    
    # Identify outliers using IQR method for both columns
    for column in ['Landsize', 'Price']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        
        # Fill with previous value
        df_prev.loc[outlier_mask, column] = df_prev[column].shift(1)
        
        # Fill with next value
        df_next.loc[outlier_mask, column] = df_next[column].shift(-1)
    
    return df_prev, df_next

# Generate report
with open('analysis_report.txt', 'w') as f:
    f.write("=== Melbourne Real Estate Dataset Analysis ===\n\n")
    
    f.write("1. Feature Analysis:\n")
    f.write("-------------------\n")
    f.write(str(df.dtypes))
    f.write("\n\n")
    
    f.write("2. Outlier Analysis:\n")
    f.write("-------------------\n")
    for column, analysis in outlier_analysis.items():
        f.write(f"\n{column}:\n")
        f.write(f"Z-score outliers: {analysis['z_score_outliers']}\n")
        f.write(f"IQR outliers: {analysis['iqr_outliers']}\n")
    
    f.write("\n3. Features Without Outliers:\n")
    f.write("---------------------------\n")
    f.write(", ".join(features_without_outliers))
    
    f.write("\n\n4. BuildingArea Outlier Handling:\n")
    f.write("-------------------------------\n")
    f.write("Outliers in BuildingArea have been handled using:\n")
    f.write("- Minimum value\n")
    f.write("- Maximum value\n")
    f.write("- Mean value\n")
    f.write("- Median value\n")
    
    f.write("\n5. Landsize and Price Outlier Handling:\n")
    f.write("------------------------------------\n")
    f.write("Outliers in Landsize and Price have been handled using:\n")
    f.write("- Previous value\n")
    f.write("- Next value\n")

# Execute the outlier handling functions
building_area_dfs = handle_building_area_outliers(df)
landsize_price_dfs = handle_landsize_price_outliers(df)

print("Analysis complete. Check analysis_report.txt for detailed results.") 