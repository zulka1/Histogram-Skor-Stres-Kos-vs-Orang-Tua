import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"Your File Directory\TubesProbstat(Responses).xlsx"
df = pd.read_excel(file_path)

kolom_stres = [col for col in df.columns if 'stres' in col.lower()]

print("Kolom stres yang dipakai:")
for k in kolom_stres:
    print("-", k)

df[kolom_stres] = df[kolom_stres].apply(pd.to_numeric, errors='coerce')

df['Total_Skor_Stres'] = df[kolom_stres].sum(axis=1)

kos = df[df['Tempat Tinggal (Saat Ini)'].str.contains('kos', case=False, na=False)]['Total_Skor_Stres']
ortu = df[df['Tempat Tinggal (Saat Ini)'].str.contains('orang', case=False, na=False)]['Total_Skor_Stres']

plt.figure(figsize=(10, 6))

sns.histplot(kos, bins=10, kde=True, stat="density",
             color='purple', alpha=0.5, label='Kos')

sns.histplot(ortu, bins=10, kde=True, stat="density",
             color='orange', alpha=0.5, label='Orang Tua')

batas_likert = [5, 10, 15, 20]

for batas in batas_likert:
    plt.axvline(batas, color='red', linestyle='--', alpha=0.7)

plt.title('Distribusi Skor Stres dengan Batas Kategori Likert')
plt.xlabel('Total Skor Stres')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
