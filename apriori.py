from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Data transaksi
transactions = [
    ['Sarung Wadimor', 'Sarung Gajah Duduk'],
    ['Sarung Gajah Duduk', 'Sarung Atlas'],
    ['Sarung Wadimor', 'Sarung Gajah Duduk', 'Sarung Atlas'],
    ['Sarung Atlas', 'Sarung Mangga'],
    ['Sarung Wadimor', 'Sarung Gajah Duduk', 'Sarung Mangga']
]

# Encode data transaksi
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori dan aturan asosiasi
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Tampilkan aturan
print("Aturan Asosiasi:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Membuat graph dari aturan asosiasi
G = nx.DiGraph()

for idx, row in rules.iterrows():
    for ant in row['antecedents']:
        for cons in row['consequents']:
            G.add_edge(ant, cons, weight=row['confidence'], lift=row['lift'])

# Posisi node
pos = nx.spring_layout(G, k=1)

# Gambar node dan edge
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, font_size=10, font_weight='bold', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
edge_labels = {k: f"conf: {v:.2f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)

plt.title("Association Rules - Network Graph")
plt.axis('off')
plt.show()
