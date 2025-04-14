import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('../data/data2_iris.csv')

# Set style
plt.style.use('seaborn-v0_8')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analisis Data Iris', fontsize=16)

# Histograms
sns.histplot(data=data, x='SepalLengthCm', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram Sepal Length')
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Frekuensi')

sns.histplot(data=data, x='SepalWidthCm', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Histogram Sepal Width')
axes[0, 1].set_xlabel('Sepal Width (cm)')
axes[0, 1].set_ylabel('Frekuensi')

# Boxplots
sns.boxplot(data=data, y='SepalLengthCm', ax=axes[1, 0])
axes[1, 0].set_title('Boxplot Sepal Length')
axes[1, 0].set_ylabel('Sepal Length (cm)')

sns.boxplot(data=data, y='SepalWidthCm', ax=axes[1, 1])
axes[1, 1].set_title('Boxplot Sepal Width')
axes[1, 1].set_ylabel('Sepal Width (cm)')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 