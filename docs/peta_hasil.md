# Peta hasil: naskah rev14 → berkas hasil → skrip

Setiap tabel dan gambar di naskah rev14 serta Suplemen S2 dan S3 dipetakan ke
berkas hasil di `results/` dan skrip yang menulisnya. Jalur relatif terhadap
akar repositori. Perintah menjalankan tiap tahap ada di `README.md`.

Kolom "rev12" mencatat nomor yang sama di naskah rev12, karena penomoran
bergeser antara rev12 dan rev14.

## Naskah utama: tabel

| Naskah rev14 | Isi | Berkas hasil | Skrip pembuat | rev12 |
|---|---|---|---|---|
| Tabel 1 | Perbandingan penelitian terdahulu | tidak ada di repositori; bersumber dari Suplemen S1 (matriks literatur) | — | Tabel 1 |
| Tabel 2 | Hasil uji validasi pipeline (T1–T6) | `results/stage_c/c1_validation_tests.csv` (T1–T5; baris T6 ditambahkan oleh `c2_summary.py`), `results/stage_c/c2_t6.csv` (T6 urut vs acak, 5 seed) | `scripts/stage_c/c1_tests.py`, `scripts/stage_c/c2_train_one.py` (model `t6_*`), `scripts/stage_c/c2_summary.py` | Tabel 9 |
| Tabel 3 | F1 per kelas pada partisi uji final, rerata 3 seed | `results/stage_d/d3_c3_test_performance.csv` (per seed), `results/stage_d/d3_c3_test_performance_ringkasan.csv` (rerata ± sd) | `scripts/stage_d/d3_c3.py` (prediksi dari `scripts/stage_d/d_predict.py`) | Tabel 10 |
| Tabel 4 | Window yang cocok eksak lintas partisi | `results/stage_d/d2_cross_split.csv` | `scripts/stage_d/d12_report.py` | Tabel 11 |
| Tabel 5 | Ringkasan metrik penjelasan (M1–M4) pada 450 instance uji | `results/stage_d/d3_c4_m1.csv`, `d3_c4_m2_local.csv`, `d3_c4_m2_system.csv`, `d3_c4_temporal.csv`, `d3_c4_m3.csv`, `d3_c4_m4.csv` (semua di `results/stage_d/`) | `scripts/stage_d/d3_c4_xai.py` (atribusi, M4), `scripts/stage_d/d3_c4_report.py` (tabel) | Tabel 12 |
| Tabel 6 | Perbandingan primer dengan bootstrap tingkat sesi | `results/stage_d/d3_c4_bootstrap_primary.csv` | `scripts/stage_d/d3_c4_report.py` | Tabel 13 |

Pendukung Tabel 3 sampai 6 (tidak dicetak di naskah utama):
`d3_c3_confusion_seed42.csv`, `d3_c3_confusion_mean.csv`, `d3_c3_session_level.csv`,
`d3_c3_test_ambiguity.csv`, `d3_c3_ablation_test.csv`, `d3_c3_control_test.csv`,
`d3_c4_additivity.csv`, `d3_c4_m2_local_per_instance.csv`, `d3_c4_m3_per_instance.csv`,
`d3_c4_m4_mentah.csv`, `d3_c4_per_class.csv`, `d3_c4_shortcut_control.csv`
(`scripts/stage_d/d3_c4_shortcut.py`), `d3_c4_bootstrap_exploratory.csv`,
`d3_pilot_exclusion.csv` dan `d3_selection_v2.csv` (`scripts/stage_d/d3_select.py`).

## Naskah utama: gambar

| Naskah rev14 | Isi | Berkas | Sumber |
|---|---|---|---|
| Gambar 1 | Alur penelitian | `docs/figures/gambar1_alur_penelitian.png` | diagram statis, tanpa data |
| Gambar 2 | Jalur DeepSHAP, LIME flat, dan LIME kustom pada masukan LSTM 3D | `docs/figures/gambar2_jalur_xai.png` | diagram statis, tanpa data |
| Gambar 3 | Kurva deletion tingkat kohort, 450 instance uji (acak = rerata 5 ulangan) | `docs/figures/gambar3_deletion_kohort.png` | data: `results/stage_d/d3_c4_m2_cohort.csv` (`scripts/stage_d/d3_c4_report.py`); plot: `scripts/figures/plot_gambar3.py` |

## Suplemen S3: analisis kepekaan, pencarian hyperparameter, dan tabel pendukung

| Bagian | Isi | Berkas hasil | Skrip pembuat | rev12 |
|---|---|---|---|---|
| Tabel S3.1 | Cakupan data pada tiga skema kunci sesi | `results/stage_a/table3_key_schemes.csv` | `scripts/stage_a/stage_a.py` | Tabel 3 |
| Tabel S3.2 | Kepekaan ambang jeda sesionisasi | `results/stage_a/table4_gap.csv` | `scripts/stage_a/stage_a.py` | Tabel 4 |
| Tabel S3.3 | Ruang pencarian hyperparameter pembanding | `results/stage_c/c2_search_{lstm1,mlp,rf,xgb}_configs.json` (20 konfigurasi per model) | `scripts/stage_c/run_stage_c.py` (`search_configs`, `default_rng(42)`) | Tabel 7 |
| C2 | Hasil pencarian hyperparameter | `results/stage_c/c2_search_{lstm1,mlp,rf,xgb}.csv`, `results/stage_c/c2_runs/search_*.json` | `scripts/stage_c/c2_train_one.py` / `c2_tree_one.py`, `scripts/stage_c/c2_summary.py` | — |
| D1 | Prapemrosesan log1p vs lama, multi-seed (validasi) | `results/stage_d/d1_log1p_multiseed.csv`, `results/stage_d/d1_runs/old_s4{3,4}.json`, `results/stage_d/d1_models_hash.json` | `scripts/stage_d/d1_train_old.py`, `scripts/stage_d/d12_report.py`, `scripts/stage_d/run_stage_d.py` (hash) | — |
| D2 | Window novel vs berpadanan (validasi) | `results/stage_d/d2_novel_val.csv` (juga `d2_novel_input.csv`, `d2_within_split.csv`) | `scripts/stage_d/d12_report.py` | — |
| D3 | Kepekaan XAI: background, perturbasi, dsb. | `results/stage_d/d3_c4_sensitivity.csv` | `scripts/stage_d/d3_c4_xai.py` (bagian `sens_bg`), `scripts/stage_d/d3_c4_report.py` | — |
| D4 | Kepekaan seed model untuk XAI (90 instance) | `results/stage_d/d4_seed_sensitivity.csv` | `scripts/stage_d/d4_seed.py` | — |
| Tabel S3.12 | Jumlah baris dan sesi per kelas sebelum/sesudah sesionisasi | `results/stage_a/table2_sessions.csv` | `scripts/stage_a/stage_a.py` | Tabel 2 |
| Tabel S3.13 | Jumlah sesi per partisi sebelum purging | `results/stage_a/table5_partitions.csv` (juga `purging_log.csv`) | `scripts/stage_a/stage_a.py` | Tabel 5 |
| Tabel S3.14 | Rancangan uji validasi pipeline | tabel rancangan, tanpa berkas hasil | — | Tabel 6 |
| Tabel S3.15 | Matriks metrik, level analisis, dan status perbandingan | tabel rancangan, tanpa berkas hasil | — | Tabel 8 |

## Suplemen S2: pilot XAI (40 instance)

| Bagian | Berkas hasil | Skrip pembuat |
|---|---|---|
| Pilot B (prapemrosesan lama) | `results/stage_b/*`: `pilot_checks.csv`, `m1_agreement.csv`, `m2_lokal_auc.csv`, `m2_empiris_auc.csv`, `temporal_deletion_auc.csv`, `m3_stability.csv`, `runtime.csv`, `bootstrap_contoh.csv`, `deletion_baseline.csv`, `pilot_log.json`, `galat_diperbaiki.json`, `train_log.json`, `_sealed_system_check.npz` | `scripts/stage_b/b0_tensors.py`, `scripts/stage_b/b1_train.py`, `scripts/stage_b/b2_pilot_xai.py` |
| Pilot B2 (log1p, final) | `results/stage_b2/*`: berkas yang sama dengan pilot B, ditambah `runtime_mentah.csv` | `scripts/stage_b2/b2_0_tensors_log.py`, `scripts/stage_b2/b2_pilot_xai_log.py` |
| Contoh perturbasi LIME | `artifacts/stage_b/lime_perturbation_examples.npz`, `artifacts/stage_b2/lime_perturbation_examples.npz` | `b2_pilot_xai.py` / `b2_pilot_xai_log.py` |

`m2_kohort_f1_TIDAK_DICETAK.csv` dan `_sealed_system_check.npz` adalah
keluaran tersegel (performa uji pilot); keduanya sengaja tidak dicetak di naskah.

## Diagnostik validasi (dirujuk dari Bagian Metode)

| Berkas hasil | Skrip pembuat |
|---|---|
| `results/diag_val/*` | `scripts/diag_val/diag_val.py` |
| `results/diag_val2/*` | `scripts/diag_val/diag_val2.py` |
| `results/diag_log/*` | `scripts/diag_log/dl1_prep_train.py`, `scripts/diag_log/dl2_eval.py` |
