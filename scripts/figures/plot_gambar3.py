"""Gambar 3 naskah rev13: kurva deletion tingkat kohort pada 450 instance uji.
Masukan : results/stage_d/d3_c4_m2_cohort.csv
Keluaran: docs/figures/gambar3_deletion_kohort.png
Pemakaian: python scripts/figures/plot_gambar3.py [jalur_csv] [jalur_png]
"""
import sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/stage_d/d3_c4_m2_cohort.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/gambar3_deletion_kohort.png"
K = [0, 1, 3, 5, 10, 15]
rows = {r["metode"]: [float(r[f"f1_k{k}"]) for k in K] for r in csv.DictReader(open(src))}
acak = [k for k in rows if k.startswith("acak")]
rows["acak_rerata"] = [sum(rows[a][i] for a in acak) / len(acak) for i in range(len(K))]

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(3.3, 2.2), dpi=300)
style = [("deepshap", "DeepSHAP", "-", "o", "black"),
         ("lime_flat", "LIME flat", "--", "s", "#555555"),
         ("kustom_flat", "Kustom-flat", (0, (3, 1, 1, 1)), "^", "#333333"),
         ("per_fitur", "Per fitur", ":", "D", "#777777"),
         ("acak_rerata", f"Acak (rerata {len(acak)})", "-", "x", "#aaaaaa")]
for key, lab, ls, mk, col in style:
    ax.plot(K, rows[key], ls=ls, marker=mk, color=col, lw=1.1, ms=4, label=lab)
ax.set_xticks(K); ax.set_ylim(0, 1.0)
ax.set_xlabel("Jumlah fitur teratas yang diganti (k)")
ax.set_ylabel("F1-macro kohort")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(out)
print("tersimpan:", out)
