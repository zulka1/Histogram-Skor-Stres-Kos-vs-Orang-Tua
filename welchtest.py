import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

path_file = 'Your File Directory/Formulir Tugas Besar Probstat  (Responses Finished) - Form Responses 1.csv'

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

df['Grup'] = df['Grup_Asli'].apply(
    lambda x: 'Mahasiswa Indekos' if 'Kos' in str(x) else 'Mahasiswa Rumah'
)

g_kos = df[df['Grup'] == 'Mahasiswa Indekos']['Total_Stress']
g_rumah = df[df['Grup'] == 'Mahasiswa Rumah']['Total_Stress']

t_stat, p_val = stats.ttest_ind(g_kos, g_rumah, equal_var=False)

v1, v2 = g_kos.var(), g_rumah.var()
n1, n2 = len(g_kos), len(g_rumah)
df_welch = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

t_kritis = stats.t.ppf(1 - 0.05/2, df_welch)

sns.set(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

colors = {"Mahasiswa Indekos": "#1E90FF", "Mahasiswa Rumah": "#FF0000"}
sns.boxplot(data=df, x='Grup', y='Total_Stress', palette=colors, width=0.4, ax=ax1)
sns.stripplot(data=df, x='Grup', y='Total_Stress', color='black', alpha=0.3, ax=ax1)
ax1.set_title("Perbandingan Sebaran Skor Stres\n(Dasar Analisis Welch's T-Test)", fontweight='bold', fontsize=13)
ax1.set_ylabel("Total Skor Stres")

x_t = np.linspace(-4, 4, 1000)
y_t = stats.t.pdf(x_t, df_welch)

ax2.plot(x_t, y_t, color='black', linewidth=2, label=f'Distribusi t (df={df_welch:.2f})')
ax2.fill_between(x_t, y_t, where=(x_t >= t_kritis), color='red', alpha=0.3, label='Area Penolakan H0')
ax2.fill_between(x_t, y_t, where=(x_t <= -t_kritis), color='red', alpha=0.3)
ax2.fill_between(x_t, y_t, where=((x_t > -t_kritis) & (x_t < t_kritis)), color='green', alpha=0.1, label='Area Terima H0')

ax2.axvline(t_stat, color='#1E90FF', linestyle='-', linewidth=4, label=f't-hitung: {t_stat:.4f}')
ax2.axvline(t_kritis, color='red', linestyle='--', label=f'Batas Kritis: ±{t_kritis:.4f}')

ax2.set_title(f"Statistik Inferensi: Distribusi t (Welch's T-Test)", fontweight='bold', fontsize=13)
ax2.set_xlabel("Nilai t")
ax2.set_xlim(-4, 4)
ax2.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(f"{'RINGKASAN STATISTIK INFERENSI (WELCH T-TEST)':^60}")
print("="*60)
print(f"Metode Uji       : Welch's T-Test (Unequal Variances)")
print(f"Derajat Kebebasan: {df_welch:.4f} (Satterthwaite Approx.)")
print(f"t-Statistic      : {t_stat:.4f}")
print(f"t-Tabel (Kritis) : ±{t_kritis:.4f}")
print(f"P-Value          : {p_val:.4f}")
print("-" * 60)

if p_val > 0.05:
    print("KESIMPULAN: Gagal Tolak H0 (Terima H0)")
    print("Tidak terdapat perbedaan rata-rata tingkat stres yang signifikan")
    print("antara mahasiswa indekos dan mahasiswa yang tinggal di rumah.")
else:
    print("KESIMPULAN: Tolak H0")
    print("Terdapat perbedaan rata-rata tingkat stres yang signifikan")
    print("antara mahasiswa indekos dan mahasiswa yang tinggal di rumah.")
print("="*60)
