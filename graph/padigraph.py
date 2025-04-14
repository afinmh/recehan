import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('../data/data1_padi.csv')

# Set style
plt.style.use('seaborn-v0_8')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analisis Data Padi', fontsize=16)

# Histograms
sns.histplot(data=data, x='Produksi', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram Produksi')
axes[0, 0].set_xlabel('Produksi (Ton)')
axes[0, 0].set_ylabel('Frekuensi')

sns.histplot(data=data, x='Luas Panen', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Histogram Luas Panen')
axes[0, 1].set_xlabel('Luas Panen (Hektar)')
axes[0, 1].set_ylabel('Frekuensi')

# Boxplots
sns.boxplot(data=data, y='Produksi', ax=axes[1, 0])
axes[1, 0].set_title('Boxplot Produksi')
axes[1, 0].set_ylabel('Produksi (Ton)')

sns.boxplot(data=data, y='Luas Panen', ax=axes[1, 1])
axes[1, 1].set_title('Boxplot Luas Panen')
axes[1, 1].set_ylabel('Luas Panen (Hektar)')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig('padi_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 