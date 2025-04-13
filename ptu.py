import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

vectors = np.array([
    [1.0, -0.5, 0.8, -1.2, 0.4, 0.6],   #V1
    [-1.1, 1.3, -0.9, 0.6, -0.7, 1.0],  #V2
    [0.4, 1.0, -0.3, -0.6, 1.1, -0.8],  #V3
    [-0.9, -1.0, 1.2, 1.1, -1.3, 0.7],  #V4
    [1.1, -0.6, 0.9, -1.0, 0.5, 0.5],   #V5
    [-1.2, 1.1, -1.1, 0.5, -0.8, 1.2],  #V6
    [0.3, 1.2, -0.5, -0.7, 1.0, -0.9],  #V7
    [-0.8, -1.1, 1.3, 1.0, -1.2, 0.8],  #V8
    [1.2, -0.4, 0.7, -1.3, 0.3, 0.4],   #V9
    [-1.0, 1.0, -1.2, 0.7, -0.6, 1.1],  #V10
    [0.5, 1.1, -0.4, -0.5, 0.9, -1.0],  #V11
    [-0.7, -1.2, 1.1, 1.2, -1.1, 0.6],  #V12
    [1.3, -0.3, 1.0, -1.1, 0.6, 0.3],   #V13
    [-1.3, 1.2, -1.0, 0.4, -0.9, 1.3],  #V14
    [0.6, 1.3, -0.6, -0.8, 0.8, -1.1],  #V15
    [-0.6, -1.3, 1.4, 0.9, -1.4, 0.9],  #V16
    [1.4, -0.2, 0.6, -1.4, 0.2, 0.2],   #V17
    [-1.4, 1.4, -1.3, 0.3, -1.0, 1.4],  #V18
    [0.2, 1.4, -0.7, -0.9, 0.7, -1.2],  #V19
    [-0.5, -1.4, 1.5, 1.3, -1.5, 1.0]   #V20
])

# Pake KMeans
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
kmeans.fit(vectors)

codebook = kmeans.cluster_centers_
labels = kmeans.predict(vectors)
reconstructed = codebook[labels]

mse = mean_squared_error(vectors, reconstructed)
print("Codebook (k = 4)")
for i, centroid in enumerate(codebook):
    print(f"Codeword {i + 1}: {centroid}")

print(f"\nMean Squared Error (MSE): {mse:.6f}")

print("\nAssignment of Vectors to Codebook")
for i, (vec, label) in enumerate(zip(vectors, labels), start=1):
    print(f"V{i:>2} → Codeword {label + 1} → {codebook[label]}")
