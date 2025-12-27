import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load data
# =========================
file_path = r"Your File Directory\TubesProbstat(Responses).xlsx"
df = pd.read_excel(file_path)

# =========================
# Kolom skor stres
# =========================
skor_columns = [
    "Seberapa besar stres yang Anda rasakan terkait urusan kebersihan dan kerapian tempat tinggal? (Misal: harus nyapu/cuci sendiri vs. disuruh-suruh orang tua)",
    "Seberapa stres Anda dengan kondisi fisik tempat tinggal Anda saat ini? (Misal: ukuran kamar sempit, panas, fasilitas rusak, air mati, atau kamar tidak estetik)",
    "Seberapa tertekan Anda dengan kurangnya privasi atau adanya aturan di tempat tinggal? (Misal: jam malam, diawasi orang tua, atau teman kos yang suka masuk kamar sembarangan)",
    "Seberapa besar tingkat stres Anda akibat gangguan suara/keributan di tempat tinggal? (Misal: tetangga kos berisik vs. keributan adik/kakak di rumah)",
    "Seberapa sering interaksi dengan orang-orang di tempat tinggal membuat Anda stres? (Misal: konflik dengan ibu bapak kos/teman kos vs. berantem dengan orang tua)"
]

# Paksa jadi numerik
for col in skor_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Total skor stres
df["Total Skor Stres"] = df[skor_columns].sum(axis=1)

# =========================
# Kelompok tempat tinggal
# =========================
kos = df[df["Tempat Tinggal (Saat Ini)"].str.contains("kos", case=False, na=False)]
ortu = df[df["Tempat Tinggal (Saat Ini)"].str.contains("orang", case=False, na=False)]

# =========================
# Interval stres (BARU)
# =========================
stress_bins = [5, 13, 22, 31, 40, 50]
stress_labels = [
    "Sangat Rendah",
    "Rendah",
    "Sedang",
    "Tinggi",
    "Sangat Tinggi"
]

# =========================
# Plot histogram + KDE
# =========================
plt.figure(figsize=(12, 6))

sns.histplot(
    kos["Total Skor Stres"],
    bins=stress_bins,
    kde=True,
    stat="count",
    label="Kos",
    alpha=0.5
)

sns.histplot(
    ortu["Total Skor Stres"],
    bins=stress_bins,
    kde=True,
    stat="count",
    label="Orang Tua",
    alpha=0.5,
    element="step"
)

# Garis batas interval stres
for batas in stress_bins:
    plt.axvline(batas, linestyle="--", color="gray", alpha=0.6)

plt.xlabel("Total Skor Stres")
plt.ylabel("Frekuensi Mahasiswa")
plt.title("Histogram & Density Tingkat Stres Mahasiswa\nBerdasarkan Tempat Tinggal")
plt.legend()
plt.tight_layout()
plt.show()
