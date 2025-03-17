import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('titanic.csv')

def describe_data(df):
    print("Deskripsi Data:")
    print(df.info())
    print("\nJumlah elemen yang tidak kosong per atribut:")
    print(df.count())
    print("\nJumlah elemen yang kosong (NaN) per atribut:")
    print(df.isnull().sum())

def quartiles(df):
    print("\nQuartiles untuk Age dan Fare:")
    print(df[['Age', 'Fare']].quantile([0.15, 0.50, 0.80]))

def age_gender_stats(df):
    print("\nUmur termuda dan tertua berdasarkan jenis kelamin:")
    print(df.groupby('Sex')['Age'].agg(['min', 'max']))
    print("\nRata-rata umur berdasarkan jenis kelamin:")
    print(df.groupby('Sex')['Age'].mean())

def correlations(df):
    print("\nKorelasi antara Age dan Fare:")
    print(df[['Age', 'Fare']].corr())

    print("\nKorelasi Age dengan atribut lainnya:")
    numeric_df = df.select_dtypes(include=['number'])  # Pilih hanya kolom numerik
    print(numeric_df.corr()['Age'])


def std_var(df):
    print("\nStandar deviasi dan Ragam Age:")
    print(f"Std: {df['Age'].std()}, Var: {df['Age'].var()}")
    print("\nStandar deviasi dan Ragam Fare:")
    print(f"Std: {df['Fare'].std()}, Var: {df['Fare'].var()}")

def process_and_analyze(df, method, fill_value=None):
    if method == 'zero':
        df_filled = df.fillna(0)
    elif method == 'pad':
        df_filled = df.fillna(method='pad')
    elif method == 'bfill':
        df_filled = df.fillna(method='bfill')
    elif method == 'drop':
        df_filled = df.dropna()
    elif method == 'mean_age':
        df_filled = df.copy()
        df_filled['Age'].fillna(df['Age'].mean(), inplace=True)
    else:
        raise ValueError("Method not recognized.")
    
    print(f"\n=== ANALISIS DATA ({method.upper()}) ===")
    describe_data(df_filled)
    quartiles(df_filled)
    age_gender_stats(df_filled)
    correlations(df_filled)
    std_var(df_filled)
    return df_filled

# Analisis dengan berbagai metode
methods = ['zero', 'pad', 'bfill', 'drop', 'mean_age']
for method in methods:
    process_and_analyze(df, method)

# Simpan hasil perbaikan dengan rata-rata Age ke dalam Excel
df_final = process_and_analyze(df, 'mean_age')
df_final.to_csv('titanic_rev.csv', index=False)
