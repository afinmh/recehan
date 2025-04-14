import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('../data/data5_toko.csv')

# Set style
plt.style.use('seaborn-v0_8')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analisis Data Toko', fontsize=16)

# Histograms
sns.histplot(data=data, x='itemDescription', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram itemDescription')
axes[0, 0].set_xlabel('itemDescription')
axes[0, 0].set_ylabel('Frekuensi')


sns.boxplot(data=data, y='itemDescription', ax=axes[1, 1])
axes[1, 1].set_title('Boxplot itemDescription')
axes[1, 1].set_ylabel('itemDescription')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig('toko_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 