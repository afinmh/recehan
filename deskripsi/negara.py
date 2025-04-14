import pandas as pd
import numpy as np

print("\n=== ANALISIS DATASET NEGARA ===")
df = pd.read_csv("../data/data4_negara.csv")

print("\n1. Lima Data Pertama:")
print(df.head())

print("\n2. Informasi Dataset:")
print(df.info())

print("\n3. Jumlah Data Null:")
print(df.isnull().sum())

print("\n4. Statistik Deskriptif (Numerik):")
print(df.describe())

print("\n5. Statistik Deskriptif (Kategorikal):")
print(df.describe(exclude=np.number)) 