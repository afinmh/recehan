import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('../data/data3_asuransi.csv')

# Set style
plt.style.use('seaborn-v0_8')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analisis Data Asuransi', fontsize=16)

# Histograms
sns.histplot(data=data, x='age', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram Usia')
axes[0, 0].set_xlabel('Usia (Tahun)')
axes[0, 0].set_ylabel('Frekuensi')

sns.histplot(data=data, x='bmi', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Histogram BMI')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('Frekuensi')

# Boxplots
sns.boxplot(data=data, y='age', ax=axes[1, 0])
axes[1, 0].set_title('Boxplot Usia')
axes[1, 0].set_ylabel('Usia (Tahun)')

sns.boxplot(data=data, y='bmi', ax=axes[1, 1])
axes[1, 1].set_title('Boxplot BMI')
axes[1, 1].set_ylabel('BMI')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig('asuransi_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 