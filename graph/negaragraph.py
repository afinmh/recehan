import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('../data/data4_negara.csv')

# Set style
plt.style.use('seaborn-v0_8')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analisis Data Negara', fontsize=16)

# Histograms
sns.histplot(data=data, x='child_mort', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram child_mord')
axes[0, 0].set_xlabel('Child Mortality')
axes[0, 0].set_ylabel('Frekuensi')

sns.histplot(data=data, x='gdpp', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Histogram GDP per Kapita')
axes[0, 1].set_xlabel('GDP per Kapita')
axes[0, 1].set_ylabel('Frekuensi')

# Boxplots
sns.boxplot(data=data, y='child_mort', ax=axes[1, 0])
axes[1, 0].set_title('Boxplot child_mort')
axes[1, 0].set_ylabel('Child Mortality')

sns.boxplot(data=data, y='gdpp', ax=axes[1, 1])
axes[1, 1].set_title('Boxplot GDP per Kapita')
axes[1, 1].set_ylabel('GDP per Kapita')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig('negara_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 