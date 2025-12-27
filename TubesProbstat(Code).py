import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

file_path = "TubesProbstat(Responses).xlsx"
df = pd.read_excel(file_path)

print(df.info())
print(df.head())

plt.figure(figsize=(10, 6))

sns.histplot(
    df['Kos'],
    bins=10,
    kde=True,
    color='steelblue',
    alpha=0.6,
    label='Kos'
)

sns.histplot(
    df['Orang Tua'],
    bins=10,
    kde=True,
    color='seagreen',
    alpha=0.6,
    label='Orang Tua'
)

plt.title('Histogram Perbandingan Skor Stres: Kos vs Orang Tua')
plt.xlabel('Total Skor Stres (Variabel X)')
plt.ylabel('Frekuensi Jumlah Mahasiswa (Variabel Y)')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
