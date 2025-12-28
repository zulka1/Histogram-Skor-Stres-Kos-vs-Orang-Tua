import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

path_file = 'Your File Directory/Formulir Tugas Besar Probstat  (Responses Finished) - Form Responses 1.csv'

if not os.path.exists(path_file):
    path_file = 'TubesProbstat(Responses) - Form Responses 1.csv'

try:
    df = pd.read_csv(path_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

df.drop(df.columns[1], axis=1, inplace=True)

df.rename(columns={
    df.columns[3]: 'Tempat_Tinggal',
    df.columns[4]: 'Stres_Kebersihan',
    df.columns[5]: 'Stres_Fisik',
    df.columns[6]: 'Stres_Privasi',
    df.columns[7]: 'Stres_Suara',
    df.columns[8]: 'Stres_Interaksi'
}, inplace=True)

kolom_stres = ['Stres_Kebersihan', 'Stres_Fisik', 'Stres_Privasi', 'Stres_Suara', 'Stres_Interaksi']

for col in kolom_stres:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Total_Stress'] = df[kolom_stres].sum(axis=1)

df['Grup'] = df['Tempat_Tinggal'].apply(
    lambda x: 'Mahasiswa Indekos' if 'Kos' in str(x) else 'Mahasiswa Tinggal dengan Orang Tua'
)

grup_kos = df[df['Grup'] == 'Mahasiswa Indekos']['Total_Stress']
grup_rumah = df[df['Grup'] == 'Mahasiswa Tinggal dengan Orang Tua']['Total_Stress']

t_stat, p_val = stats.ttest_ind(grup_kos, grup_rumah, equal_var=True)
df_student = len(grup_kos) + len(grup_rumah) - 2
t_kritis = stats.t.ppf(1 - 0.05/2, df_student)

sns.set(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

bins_list = [5, 14, 23, 32, 41, 50]
colors = {"Mahasiswa Indekos": "#1E90FF", "Mahasiswa Tinggal dengan Orang Tua": "#FF0000"}

sns.histplot(data=df, x='Total_Stress', hue='Grup', bins=bins_list, 
             palette=colors, multiple="dodge", shrink=0.8, alpha=0.3, edgecolor="black", ax=ax1)

def plot_kde_and_stats(data, color_line, label_tag, ax):
    if len(data) > 1:
        xs = np.linspace(5, 50, 500)
        kde = stats.gaussian_kde(data)
        ys = kde(xs)
        ys_scaled = ys * len(data) * 9 
        ax.plot(xs, ys_scaled, color=color_line, linewidth=2, zorder=10)

        mean_v = data.mean()
        med_v = data.median()
        mode_res = data.mode()
        mode_v = mode_res[0] if not mode_res.empty else mean_v
        
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
y_t = stats.t.pdf(x_t, df_student)

plt.plot(x_t, y_t, color='black', linewidth=2)
plt.fill_between(x_t, y_t, where=(x_t >= t_kritis), color='red', alpha=0.3, label='Area Penolakan H0')
plt.fill_between(x_t, y_t, where=(x_t <= -t_kritis), color='red', alpha=0.3)
plt.fill_between(x_t, y_t, where=((x_t > -t_kritis) & (x_t < t_kritis)), color='green', alpha=0.1, label='Area Terima H0')

plt.axvline(t_stat, color='#1E90FF', linestyle='-', linewidth=4, label=f't-hitung: {t_stat:.4f}')
plt.axvline(t_kritis, color='red', linestyle='--', label=f'Batas Kritis: ±{t_kritis:.4f}')

plt.title("Statistik Inferensi: Kurva Distribusi T (Student's T-Test)", fontsize=14, fontweight='bold')
plt.xlim(-4, 4)
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print(f"{'STATISTIK AKURAT (STUDENT T-TEST)':^50}")
print("="*50)
print(f"Total Responden: {len(df)}")
for label, data in [("INDEKOS", grup_kos), ("ORANG TUA", grup_rumah)]:
    print(f"[{label}] -> n: {len(data)} | Mean: {data.mean():.4f} | SD: {data.std():.4f} | Var: {data.var():.4f}")

print("-" * 50)
print(f"Hasil Uji T -> t-stat: {t_stat:.4f} | p-val: {p_val:.4f} | df: {df_student}")
print(f"Batas Kritis -> ±{t_kritis:.4f}")
print("="*50)
