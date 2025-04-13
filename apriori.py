# Install mlxtend jika belum terinstal
# pip install mlxtend

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

dataset = [
    ['Susu', 'Roti'],
    ['Susu', 'Popok', 'Bir', 'Roti'],
    ['Susu', 'Popok', 'Bir', 'Cola'],
    ['Popok', 'Bir'],
    ['Susu', 'Popok', 'Bir', 'Roti']
]

# 2. One-hot encoding
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 3. Cari frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# 4. Generate aturan asosiasi
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Tampilkan aturan asosiasi
print("=== Aturan Asosiasi ===")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 5. Visualisasi grafik asosiasi
G = nx.DiGraph()

for _, row in rules.iterrows():
    antecedent = ', '.join(list(row['antecedents']))
    consequent = ', '.join(list(row['consequents']))
    G.add_edge(antecedent, consequent, weight=row['lift'])

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, k=1, seed=42)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=10, font_weight='bold', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_color='red')
plt.title("Grafik Aturan Asosiasi (Lift)")
plt.show()
