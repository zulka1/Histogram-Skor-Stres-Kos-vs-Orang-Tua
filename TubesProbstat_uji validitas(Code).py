import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

path_file = 'Your File Directory/Formulir Tugas Besar Probstat  (Responses Finished) - Form Responses 1.csv'

if not os.path.exists(path_file):
    path_file = 'TubesProbstat(Responses) - Form Responses 1.csv'

try:
    df = pd.read_csv(path_file)
except Exception as e:
    print(f"Error: {e}")
    exit()

df.rename(columns={df.columns[4]: 'Grup_Asli'}, inplace=True)

cols_to_sum = df.columns[5:10]
for col in cols_to_sum:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Total_Stress'] = df[cols_to_sum].sum(axis=1)

df['Grup'] = df['Grup_Asli'].apply(lambda x: 'Mahasiswa Indekos' if 'Kos' in str(x) else 'Mahasiswa Tinggal dengan Orang Tua')

g_kos = df[df['Grup'] == 'Mahasiswa Indekos']['Total_Stress']
g_rumah = df[df['Grup'] == 'Mahasiswa Tinggal dengan Orang Tua']['Total_Stress']

stat_k, p_k = stats.shapiro(g_kos)
stat_r, p_r = stats.shapiro(g_rumah)

stat_lev, p_lev = stats.levene(g_kos, g_rumah)

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

stats.probplot(g_kos, dist="norm", plot=axes[0, 0])
axes[0, 0].set_title(f"Q-Q Plot: Indekos (p-Shapiro={p_k:.4f})", fontweight='bold')

stats.probplot(g_rumah, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title(f"Q-Q Plot: Rumah (p-Shapiro={p_r:.4f})", fontweight='bold')

sns.boxplot(data=df, x='Grup', y='Total_Stress', palette="Set2", width=0.5, ax=axes[1, 0])
axes[1, 0].set_title(f"Uji Homogenitas (Levene p={p_lev:.4f})", fontweight='bold')

sns.histplot(data=df, x='Total_Stress', hue='Grup', kde=True, palette="husl", element="step", ax=axes[1, 1])
axes[1, 1].set_title("Distribusi Data & Kurva KDE", fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(f"{'HASIL VALIDASI ASUMSI STATISTIK':^60}")
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
    print("KESIMPULAN: Asumsi Normalitas/Homogenitas tidak terpenuhi sepenuhnya.")
    print("Saran: Cek pencilan (outliers) atau gunakan Central Limit Theorem (N > 30).")
print("="*60)
