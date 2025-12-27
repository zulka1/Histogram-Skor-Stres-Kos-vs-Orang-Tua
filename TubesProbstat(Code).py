import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

path_file = r"d:/kuliah/3/probstat/New folder/Histogram-Skor-Stres-Kos-vs-Orang-Tua-main/TubesProbstat(Responses).csv"

if not os.path.exists(path_file):
    path_file = 'TubesProbstat(Responses) - Form Responses 1.csv'

try:
    df = pd.read_csv(path_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

df.rename(columns={
    df.columns[6]: 'Tempat_Tinggal',
    df.columns[7]: 'Stres_Kebersihan',
    df.columns[8]: 'Stres_Fisik',
    df.columns[9]: 'Stres_Privasi',
    df.columns[10]: 'Stres_Suara',
    df.columns[11]: 'Stres_Interaksi'
}, inplace=True)

df['Total_Stress'] = df.iloc[:, 7:12].sum(axis=1)

df['Grup'] = df['Tempat_Tinggal'].apply(
    lambda x: 'Mahasiswa Indekos' if 'Kos' in x else 'Mahasiswa Tinggal dengan Orang Tua'
)

grup_kos = df[df['Grup'] == 'Mahasiswa Indekos']['Total_Stress']
grup_rumah = df[df['Grup'] == 'Mahasiswa Tinggal dengan Orang Tua']['Total_Stress']

t_stat, p_val = stats.ttest_ind(grup_kos, grup_rumah, equal_var=False)
v1, v2 = grup_kos.var()/len(grup_kos), grup_rumah.var()/len(grup_rumah)
df_welch = (v1 + v2)**2 / ( (v1**2 / (len(grup_kos)-1)) + (v2**2 / (len(grup_rumah)-1)) )
t_kritis = stats.t.ppf(1 - 0.05/2, df_welch)

sns.set(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

bins_list = [5, 14, 23, 32, 41, 50]
colors = {"Mahasiswa Indekos": "#1E90FF", "Mahasiswa Tinggal dengan Orang Tua": "#FF0000"}

sns.histplot(data=df, x='Total_Stress', hue='Grup', bins=bins_list, 
             palette=colors, multiple="dodge", shrink=0.8, alpha=0.3, edgecolor="black", ax=ax1)

def plot_kde_and_stats(data, color_line, label_tag, ax):
    xs = np.linspace(5, 50, 500)
    kde = stats.gaussian_kde(data)
    ys = kde(xs)
    ys_scaled = ys * len(data) * 9 
    ax.plot(xs, ys_scaled, color=color_line, linewidth=2, zorder=10) 

    mean_v = data.mean()
    med_v = data.median()
    mode_v = data.mode()[0]
    
    ax.axvline(mean_v, color=color_line, linestyle='-', linewidth=2, label=f'{label_tag} Mean: {mean_v:.4f}')
    ax.axvline(med_v, color=color_line, linestyle='--', linewidth=1.5, label=f'{label_tag} Median: {med_v:.4f}')
    ax.axvline(mode_v, color=color_line, linestyle=':', linewidth=1.5, label=f'{label_tag} Mode: {mode_v:.4f}')

plot_kde_and_stats(grup_kos, "#0000CD", "Indekos", ax1)
plot_kde_and_stats(grup_rumah, "#8B0000", "Orang Tua", ax1)

ax1.set_title("Deskripsi Statistik Data: Distribusi Skor Stres", fontweight='bold')
ax1.set_xticks(bins_list)
ax1.legend(fontsize='x-small', loc='upper right', frameon=True)

sns.boxplot(data=df, x='Grup', y='Total_Stress', palette=colors, width=0.4, ax=ax2)
ax2.set_title("Visualisasi Variansi: Boxplot Skor Stres", fontweight='bold')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
x_t = np.linspace(-4, 4, 1000)
y_t = stats.t.pdf(x_t, df_welch)

plt.plot(x_t, y_t, color='black', linewidth=2)

plt.fill_between(x_t, y_t, where=(x_t >= t_kritis), color='red', alpha=0.3, label='Area Penolakan H0')
plt.fill_between(x_t, y_t, where=(x_t <= -t_kritis), color='red', alpha=0.3)
plt.fill_between(x_t, y_t, where=((x_t > -t_kritis) & (x_t < t_kritis)), color='green', alpha=0.1, label='Area Terima H0')

plt.axvline(t_stat, color='#1E90FF', linestyle='-', linewidth=4, label=f't-hitung: {t_stat:.4f}')
plt.axvline(t_kritis, color='red', linestyle='--', label=f'Batas Kritis: ±{t_kritis:.4f}')

plt.title("Statistik Inferensi: Kurva Distribusi T (Welch's T-Test)", fontsize=14, fontweight='bold')
plt.xlim(-4, 4)
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print(f"{'STATISTIK AKURAT (4 DESIMAL)':^50}")
print("="*50)
for label, data in [("INDEKOS", grup_kos), ("ORANG TUA", grup_rumah)]:
    print(f"[{label}] -> Mean: {data.mean():.4f} | Median: {data.median():.4f} | Mode: {data.mode()[0]:.4f} | SD: {data.std():.4f} | Var: {data.var():.4f}")

print("-" * 50)
print(f"Hasil Uji T -> t-stat: {t_stat:.4f} | p-val: {p_val:.4f} | t-kritis: {t_kritis:.4f}")
print("="*50)
