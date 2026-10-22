# Evaluasi Kualitas Penjelasan SHAP dan LIME pada Sistem Deteksi Intrusi Berbasis LSTM untuk Industrial IoT

**English summary.** This repository contains the code, result tables, and small
artifacts behind the manuscript *"Assessing SHAP and LIME Explanation Quality in
LSTM-Based Intrusion Detection for Industrial IoT"* (rev13). A two-layer LSTM is
trained on session-based, leakage-controlled windows (w = 10) built from the
TON_IoT network dataset, and DeepSHAP, LIME (flat), and a custom LIME are
compared with four explanation metrics (agreement, faithfulness via deletion,
stability, and runtime) on 450 test instances. The TON_IoT data are **not**
redistributed here; download them from UNSW Canberra and place them as
described below. Every file in `artifacts/` and `results/`, including those
excluded from git, is listed with its SHA-256 hash in `MANIFEST.sha256`.

---

- **Penulis:** Benediktus Aditya Pratama, Hendi Hermawan (Universitas Pembangunan Jaya)
- **Naskah:** [menunggu terbit]
- **Peta tabel/gambar naskah → berkas hasil → skrip:** [`docs/peta_hasil.md`](docs/peta_hasil.md)

## 1. Dataset

Penelitian memakai **TON_IoT** (UNSW Canberra, Moustafa et al.), subset
*Processed Network dataset*: 23 berkas `Network_dataset_1.csv` sampai
`Network_dataset_23.csv`. Sumber resmi: <https://research.unsw.edu.au/projects/toniot-datasets>.

Data **tidak didistribusikan ulang** di repositori ini, baik mentah maupun
turunan yang memuat baris data (`X_*.npy`, `y_*.npy`, `rows_*`, `block_rows.csv.gz`,
`rows_meta.csv`, dan sejenisnya). Letakkan ke-23 berkas CSV di:

```
Dataset/Processed_datasets/Processed_Network_dataset/Network_dataset_{1..23}.csv
```

Jalur ini ditetapkan di `scripts/pipeline_common.py` (`DATA_DIR`). Semua skrip
dijalankan dari akar repositori.

Catatan tentang artefak yang ikut: kosakata kolom `dns_query` di
`artifacts/stage_a/categorical_codes.json`, `artifacts/stage_a/encoder.pkl`, dan
`artifacts/stage_b*/features.json` diturunkan dari dataset publik TON_IoT. Alamat
IP yang muncul di dalamnya (misalnya nama reverse-DNS `*.in-addr.arpa`) berasal
dari testbed TON_IoT, bukan dari jaringan nyata.

## 2. Environment

Semua eksperimen dijalankan di CPU Apple M1 (RAM 8 GB), macOS, Python 3.11,
conda env `skripsi-shap`. Ekspor lengkap ada di `environment/`:

| Berkas | Isi |
|---|---|
| `environment/environment.yml` | `conda env export --no-builds -n skripsi-shap` |
| `environment/requirements-freeze.txt` | `pip freeze` dari env yang sama |
| `environment/versions.txt` | versi Python, TensorFlow, tf-keras, shap, lime, scikit-learn, xgboost, numpy, chip |

Versi kunci: TensorFlow 2.16.2, tf-keras 2.16.0, shap 0.44.1, lime 0.2.0.1,
scikit-learn 1.9.0, xgboost 3.2.0, numpy 1.26.4. Kombinasi ini rapuh: shap
versi lebih baru dan Keras 3 gagal menjalankan DeepSHAP pada LSTM.

```bash
conda env create -f environment/environment.yml
conda activate skripsi-shap
```

### Urutan impor wajib

Skrip yang memuat model Keras atau menjalankan XAI dibuka dengan urutan ini
(lihat misalnya `scripts/stage_c/c4_xai.py`, `scripts/stage_d/d_predict.py`):

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"      # SEBELUM import tensorflow (Keras 2)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # XAI di CPU
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()           # mode graph, wajib untuk DeepSHAP
import shap

tf.keras.utils.set_random_seed(42)           # SEBELUM memuat model
model = tf.keras.models.load_model("....h5", compile=False)
```

- DeepSHAP dan LIME **harus di CPU**; di GPU DeepSHAP gagal (`shap_CudnnRNNV3`).
- Model `.h5` selalu dimuat dengan `compile=False`.
- Pelatihan (C2, D1) memakai mode eager dengan seed dikunci tepat sebelum model
  dibangun; detailnya di docstring `scripts/stage_c/c2_train_one.py`.

## 3. Urutan pipeline

Setiap tahap membaca keluaran tahap sebelumnya dari `artifacts/` dan menulis
tabel ke `results/<tahap>/`. Driver Tahap C dan D melewati langkah yang
keluarannya sudah ada, sehingga bisa dilanjutkan. Eksperimen asli dijalankan satu
proses sekaligus, dengan `caffeinate -i` agar mesin tidak tidur.

| # | Tahap | Perintah | Keluaran utama |
|---|---|---|---|
| 1 | **A**: sesionisasi tanpa label (kunci `src_ip, dst_ip, dst_port, proto`, jeda 15 detik), kuota, pembagian per kelas 70/15/15 kronologis, purging; pemilihan 450 instance utama dan 40 pilot | `python scripts/stage_a/stage_a.py` | `artifacts/stage_a/`, `results/stage_a/` (Tabel 2–5) |
| 2 | **B0/B1**: tensor window w=10 (prapemrosesan lama), latih model utama seed 42 | `python scripts/stage_b/b0_tensors.py`<br>`python scripts/stage_b/b1_train.py` | `artifacts/stage_b/` |
| 3 | **Pilot B**: pilot XAI 40 instance | `python scripts/stage_b/b2_pilot_xai.py` | `results/stage_b/` |
| 4 | **diag_val**: diagnostik model B1 pada validasi | `python scripts/diag_val/diag_val.py`<br>`python scripts/diag_val/diag_val2.py` | `results/diag_val/`, `results/diag_val2/` |
| 5 | **diag_log**: uji satu perubahan prapemrosesan (log1p sebelum Min-Max) | `python scripts/diag_log/dl1_prep_train.py`<br>`python scripts/diag_log/dl2_eval.py` | `artifacts/diag_log/`, `results/diag_log/` |
| 6 | **Pilot B2**: pilot ulang dengan prapemrosesan final (log1p) | `python scripts/stage_b2/b2_0_tensors_log.py`<br>`python scripts/stage_b2/b2_pilot_xai_log.py` | `artifacts/stage_b2/`, `results/stage_b2/` |
| 7 | **C**: C0 persiapan + pembekuan konfigurasi, C1 uji validasi, C2 pelatihan & pencarian hyperparameter, C3 evaluasi uji, C4 XAI 450 instance | `python scripts/stage_c/run_stage_c.py all`<br>(atau per langkah: `c0`, `c1`, `c2`, `c3`, `c4`) | `artifacts/stage_c/`, `results/stage_c/` |
| 8 | **D**: D0 persiapan, D1 model lama multi-seed, prediksi, D1/D2 laporan, D3 evaluasi akhir (450 instance v2, sesi pilot dikeluarkan), D4 kepekaan seed | `python scripts/stage_d/run_stage_d.py all`<br>(atau per langkah: `d0`, `d1`, `pred`, `d12`, `d3`, `d4`) | `artifacts/stage_d/`, `results/stage_d/` |

Urutan di atas mengikuti dependensi berkas dan cap waktu keluaran asli (B1 harus
ada sebelum diag_val, dan diag_log harus ada sebelum B2). Log eksekusi asli ada
di `results/stage_a/run_log.txt`, `results/stage_c/run_log.txt`,
`results/stage_d/run_log.txt`, serta berkas `*_stdout.txt`.

Gambar 3 naskah dibuat ulang dari `results/stage_d/d3_c4_m2_cohort.csv` dengan
`python scripts/figures/plot_gambar3.py` (keluaran
`docs/figures/gambar3_deletion_kohort.png`). Gambar 1 dan 2 adalah diagram statis.

Langkah-langkah individual C dan D (misalnya `c2_train_one.py '<json spec>'`,
`c4_xai.py <bagian>`, `d4_seed.py <seed>`) dipanggil oleh driver; argumennya
tercatat di docstring tiap skrip dan di `run_log.txt`.

## 4. Aturan data uji

- Prediksi dan metrik pada **data uji** untuk naskah hanya dihitung di **C3**
  (performa), **C4** (XAI), dan **D3** (evaluasi akhir). Semua pemilihan
  model, epoch, dan prapemrosesan memakai data **validasi**.
- **Sesi asal 40 instance pilot** (`artifacts/stage_a/pilot_40_index.csv`)
  dikeluarkan dari evaluasi akhir D3 (`artifacts/stage_d/test_final_mask.npz`,
  `results/stage_d/d3_pilot_exclusion.csv`). 450 instance evaluasi akhir dipilih
  ulang dari sesi uji yang tersisa (`artifacts/stage_d/main_450_index_v2.csv`).
- Partisi uji juga dibaca pada langkah berikut, tetapi tanpa memilih apa pun
  berdasarkan hasilnya:
  - B0/B2-0 dan C0/D0 membentuk atau memverifikasi tensor uji.
  - Pilot B dan B2 menyegel F1 uji pilot tanpa mencetaknya
    (`_sealed_system_check.npz`, `m2_kohort_f1_TIDAK_DICETAK.csv`).
  - C1 (T4) hanya meng-hash window uji.
  - `d_predict.py` mengulang prediksi uji model C3 dan memeriksa bahwa hasilnya
    identik (`d_predict_check.csv`).
  - D2 membandingkan window lintas partisi (Tabel 8).

## 5. Memverifikasi hash

`MANIFEST.sha256` berisi hash SHA-256 untuk **semua** berkas di `artifacts/`
dan `results/` (kecuali `.DS_Store`), termasuk yang tidak masuk git. Untuk berkas
yang ada di repositori:

```bash
shasum -a 256 -c MANIFEST.sha256 --ignore-missing
```

Setelah data dan artefak besar dibuat ulang atau diunduh, jalankan tanpa
`--ignore-missing` untuk memeriksa semuanya. Hash model dan artefak yang dicatat
saat eksperimen (`results/stage_c/config_frozen.json`,
`results/stage_c/freeze_after_c2.json`, `results/stage_d/config_d.json`,
`results/stage_d/d1_models_hash.json`) cocok dengan manifest (93 dari 93 entri).

### Artefak yang tidak ada di repositori

| Berkas | Alasan | Cara memperoleh |
|---|---|---|
| `artifacts/stage_c/models/rf_s4{2,3,4}.joblib` (53 MB/berkas) | di atas batas 25 MB | akan diunggah lewat GitHub Release atau Zenodo; hash ada di manifest |
| `X_*.npy`, `y_*.npy`, `rows_*.npy`, `rows_meta.csv`, `block_rows.csv.gz`, `lime_perturbation_cat_codes.npz` | memuat baris, label, atau sampel data TON_IoT (dan IP pada `rows_meta.csv`) | jalankan ulang tahap terkait dari dataset asli |
| `results/stage_c_hasil.zip`, `results/stage_d.zip` | duplikat isi `results/stage_c/` dan `results/stage_d/` | tidak diperlukan |

## 6. Struktur folder

```
.
├── README.md, LICENSE, CITATION.cff, MANIFEST.sha256, .gitignore
├── environment/            ekspor conda + pip, versi paket, chip
├── docs/
│   ├── peta_hasil.md       tabel/gambar naskah -> berkas hasil -> skrip
│   └── figures/            Gambar 1-3 naskah rev13
├── scripts/
│   ├── pipeline_common.py  pembacaan 23 berkas CSV (dipakai A, B0, B2-0, C0, D0, diag_log)
│   ├── figures/            plot_gambar3.py (Gambar 3 dari d3_c4_m2_cohort.csv)
│   ├── stage_a/            Tahap A
│   ├── stage_b/            B0, B1, pilot B
│   ├── stage_b2/           B2-0, pilot B2
│   ├── diag_val/           diagnostik validasi
│   ├── diag_log/           uji log1p
│   ├── stage_c/            Tahap C (driver run_stage_c.py)
│   └── stage_d/            Tahap D (driver run_stage_d.py)
├── artifacts/              encoder, scaler, indeks sesi/partisi/instance, model .h5 & XGBoost .ubj,
│                           atribusi .npz, prediksi uji (preds/)
└── results/                semua tabel hasil (CSV/JSON/log) per tahap
```

## 7. Arsip yang tidak dipakai naskah rev13

- `scripts/pipeline_common.py` berasal dari pipeline v1. Skrip Tahap A sampai D
  dan diag_log hanya mengimpor `CHUNKSIZE` dan `sorted_network_files` darinya.
  Konstanta lain di berkas itu (misalnya `GROUP_COLS` 3 atribut, `BUFFER_ROWS`)
  milik v1 dan **tidak dipakai** naskah rev13.
- `artifacts/encoders.pkl` dan `artifacts/scaler.pkl` (tingkat atas) adalah
  encoder/scaler pipeline v1.

## 8. Lisensi

- **Kode** (`scripts/`): MIT License, lihat [`LICENSE`](LICENSE).
- **Hasil, artefak, dan dokumentasi** (`results/`, `artifacts/`, `docs/`,
  README): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Dataset TON_IoT** tunduk pada ketentuan pemiliknya (UNSW Canberra) dan tidak
  termasuk dalam lisensi di atas.

## 9. Sitasi

Lihat [`CITATION.cff`](CITATION.cff). Sampai naskah terbit:

```
Pratama, B. A., & Hermawan, H. (2026). Evaluasi Kualitas Penjelasan SHAP dan LIME
pada Sistem Deteksi Intrusi Berbasis LSTM untuk Industrial IoT [Kode dan data
hasil]. Universitas Pembangunan Jaya. https://github.com/aditbap/xai-ids-iiot-lstm
```

Sitasi dataset:

N. Moustafa, “A new distributed architecture for evaluating AI-based security
systems at the edge: Network TON_IoT datasets,” *Sustainable Cities and Society*,
vol. 72, Art. no. 102994, 2021, doi: [10.1016/j.scs.2021.102994](https://doi.org/10.1016/j.scs.2021.102994).
