import pandas as pd
import numpy as np

print("\n=== ANALISIS DATASET TOKO ===")
df = pd.read_csv("../data/data5_toko.csv")

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

print("\n6. Jumlah Unik untuk Setiap Kolom:")
print("\nMember_number unik:", df['Member_number'].nunique())
print("Tanggal unik:", df['Date'].nunique())
print("Produk unik:", df['itemDescription'].nunique())

print("\n7. Top 5 Produk Terlaris:")
print(df['itemDescription'].value_counts().head()) 