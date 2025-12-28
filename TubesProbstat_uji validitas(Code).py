import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

path_file = r'/Users/michaelalex/Michael Alex/File Perkuliahan/VS Code/Probstat/TubesProbstat(Responses) - Form Responses 1.csv'

if not os.path.exists(path_file):
    path_file = 'TubesProbstat(Responses) - Form Responses 1.csv'

try:
    df = pd.read_csv(path_file)
except Exception as e:
    print(f"Error: {e}")
    exit()

df.rename(columns={df.columns[6]: 'Grup_Asli'}, inplace=True)
df['Total_Stress'] = df.iloc[:, 7:12].sum(axis=1)
df['Grup'] = df['Grup_Asli'].apply(lambda x: 'Mahasiswa Indekos' if 'Kos' in x else 'Mahasiswa Tinggal dengan Orang Tua')

g_kos = df[df['Grup'] == 'Mahasiswa Indekos']['Total_Stress']
g_rumah = df[df['Grup'] == 'Mahasiswa Tinggal dengan Orang Tua']['Total_Stress']

stat_k, p_k = stats.shapiro(g_kos)
stat_r, p_r = stats.shapiro(g_rumah)

stat_lev, p_lev = stats.levene(g_kos, g_rumah)

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

stats.probplot(g_kos, dist="norm", plot=axes[0, 0])
axes[0, 0].set_title(f"Q-Q Plot: Indekos (p={p_k:.4f})")

stats.probplot(g_rumah, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title(f"Q-Q Plot: Rumah (p={p_r:.4f})")

sns.boxplot(data=df, x='Grup', y='Total_Stress', palette="Set2", ax=axes[1, 0])
axes[1, 0].set_title(f"Uji Homogenitas (Levene p={p_lev:.4f})")

sns.histplot(data=df, x='Total_Stress', hue='Grup', kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Distribusi & Kurva KDE")

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("           HASIL VALIDASI ASUMSI STATISTIK")
print("="*60)

status_k = "NORMAL" if p_k > 0.05 else "TIDAK NORMAL"
status_r = "NORMAL" if p_r > 0.05 else "TIDAK NORMAL"
status_h = "HOMOGEN" if p_lev > 0.05 else "HETEROGEN"

print(f"1. Uji Normalitas (Shapiro-Wilk):")
print(f"   - Grup Indekos   : p = {p_k:.4f} -> {status_k}")
print(f"   - Grup Rumah     : p = {p_r:.4f} -> {status_r}")

print(f"\n2. Uji Homogenitas (Levene's Test):")
print(f"   - P-Value Levene : p = {p_lev:.4f} -> {status_h}")

print("-" * 60)
if p_k > 0.05 and p_r > 0.05 and p_lev > 0.05:
    print("KESIMPULAN: Data Normal & Homogen. Siap untuk Student's T-Test.")
else:
    print("KESIMPULAN: Ada asumsi yang tidak terpenuhi, cek kembali data.")
print("="*60)
