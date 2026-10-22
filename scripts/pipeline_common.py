"""
Modul bersama dipakai oleh build_dataset.py dan leakage_tests.py. Berisi
konstanta pipeline dan fungsi-fungsi yang dipakai di lebih dari satu skrip,
supaya logikanya tidak digandakan (terutama logika baca-filter 23 file CSV,
yang mahal kalau diulang).

TIDAK ada model skripsi asli dibangun di sini -- hanya utilitas data.
"""

import glob
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

DATA_DIR = "Dataset/Processed_datasets/Processed_Network_dataset"
SEED = 42
CHUNKSIZE = 500_000

GROUP_COLS = ["src_ip", "dst_ip", "dst_port"]
MIN_GROUP_SIZE = 10
DASH_THRESHOLD = 0.95
TARGET_ROWS = 1_000_000
WINDOW = 10
STRIDE = 1

# Kolom yang TIDAK BOLEH dibuang oleh aturan ">=95% placeholder '-'" walaupun
# secara kebetulan memenuhi ambang itu -- karena dibutuhkan untuk grouping,
# windowing, atau jadi target label.
PROTECTED_COLS = set(GROUP_COLS) | {"ts", "label", "type"}

TS_YEAR_MIN = 2018
TS_YEAR_MAX = 2021

# --- Konstanta sesionisasi (putaran build ke-2, unit atomik = session_id) ---
SESSION_GAP_SECONDS = 15     # ambang jeda waktu utk mulai sesi baru, sudah diputuskan pengguna
SESSION_MIN_ROWS = 10        # sesi <10 baris dibuang
LARGE_SESSION_THRESHOLD = 100  # sesi >=100 baris displit internal (temporal), <100 displit acak utuh
BUFFER_ROWS = 9              # baris penyangga dibuang di ujung segmen train & val (anti-bocor window w=10)
TRAIN_CUT = 0.70
VAL_CUT = 0.85


def session_index_array(ts_sorted, threshold=SESSION_GAP_SECONDS):
    """Dari array ts yang SUDAH TERURUT, hasilkan array indeks sesi (0,1,2,...)
    sepanjang array yang sama: nilai indeks bertambah 1 setiap kali selisih ts
    dgn baris sebelumnya > threshold. Dipakai identik di PASS A (hitung
    statistik agregat per sesi) dan PASS B (assign baris asli ke sesi) --
    karena murni fungsi deterministik dari ts yang sama, hasilnya PASTI
    konsisten di kedua pass tanpa perlu menyimpan assignment ke disk."""
    if len(ts_sorted) == 0:
        return np.array([], dtype=np.int64)
    diffs = np.diff(ts_sorted)
    new_session = diffs > threshold
    return np.concatenate(([0], np.cumsum(new_session))).astype(np.int64)


def sorted_network_files(data_dir=DATA_DIR):
    """Ambil semua file Network_dataset_N.csv, urutkan berdasarkan angka N
    (bukan abjad string), supaya file ke-2 tidak nyasar setelah file ke-19."""
    files = glob.glob(os.path.join(data_dir, "Network_dataset_*.csv"))

    def file_number(path):
        m = re.search(r"Network_dataset_(\d+)\.csv$", path)
        return int(m.group(1)) if m else float("inf")

    return sorted(files, key=file_number)


def read_filtered_chunks(files, chunksize=CHUNKSIZE):
    """Generator yang membaca semua file secara chunked dan menerapkan
    LANGKAH 1-3 dari spesifikasi pipeline pada setiap chunk sebelum
    di-yield: (1) buang kolom 'uid' kalau ada [supaya semua file punya skema
    kolom sama walau Network_dataset_6.csv aslinya punya kolom ekstra itu],
    (2) buang baris rusak (src_ip=='0' atau src_bytes=='0.0.0.0'), (3) buang
    baris berkelas 'mitm'. Dipakai bersama oleh PASS A dan PASS B supaya
    logika filter baris selalu konsisten di kedua pass."""
    dtype_overrides = {"src_bytes": "string"}
    for path in files:
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype=dtype_overrides, low_memory=False):
            if "uid" in chunk.columns:
                chunk = chunk.drop(columns=["uid"])

            bad_mask = (chunk["src_ip"] == "0") | (chunk["src_bytes"] == "0.0.0.0")
            chunk = chunk.loc[~bad_mask]
            chunk = chunk.loc[chunk["type"] != "mitm"]
            if chunk.empty:
                continue

            chunk["src_bytes"] = chunk["src_bytes"].astype("int64")
            yield os.path.basename(path), chunk


def is_categorical(series):
    """Helper kecil: cek apakah suatu kolom pandas bertipe string/kategorikal
    (bukan numerik). Dipakai untuk memutuskan cara mengganti placeholder '-'
    (kategorikal -> 'none', numerik -> 0) dan kolom mana yang perlu di-encode."""
    return pd.api.types.is_string_dtype(series)


def safe_stratified_split(items, classes, test_size, seed):
    """Bungkus sklearn train_test_split(..., stratify=classes) supaya tidak
    error kalau ada kelas dengan cuma 1 anggota (stratifikasi sklearn butuh
    minimal 2 anggota per kelas). Kelas dengan <2 anggota dipaksa masuk ke
    bagian 'kiri' (train/val) apa adanya, dengan peringatan dicetak -- sisanya
    displit stratified seperti biasa. Ini penyesuaian teknis supaya split
    yang DIMINTA (stratified per grup) bisa benar-benar dijalankan, bukan
    perubahan keputusan pipeline."""
    from sklearn.model_selection import train_test_split

    counts = Counter(classes)
    forced_left, forced_left_classes = [], []
    pool_items, pool_classes = [], []
    for it, c in zip(items, classes):
        if counts[c] < 2:
            forced_left.append(it)
            forced_left_classes.append(c)
            print(f"  PERINGATAN: kelas '{c}' cuma py 1 grup -- dipaksa masuk sisi kiri split (tidak bisa distratifikasi).")
        else:
            pool_items.append(it)
            pool_classes.append(c)

    if len(pool_items) == 0:
        return forced_left, [], forced_left_classes, []

    left_i, right_i, left_c, right_c = train_test_split(
        pool_items, pool_classes, test_size=test_size, random_state=seed, stratify=pool_classes
    )
    left_i = forced_left + left_i
    left_c = forced_left_classes + left_c
    return left_i, right_i, left_c, right_c


def build_windows_for_group(group_df, feature_cols, window=WINDOW, stride=STRIDE, shuffle_seed=None):
    """Membentuk window w=10 stride=1 DI DALAM satu grup. Kalau shuffle_seed
    None: baris di-sort berdasarkan ts dulu (urutan waktu asli -- dipakai di
    pipeline utama). Kalau shuffle_seed diisi: urutan baris DIACAK (bukan
    di-sort by ts) sebelum windowing -- dipakai khusus UJI 3 (uji shuffle)
    untuk sengaja merusak struktur temporal dan lihat apakah LSTM masih bisa
    'belajar' (kalau bisa, berarti window tidak benar-benar bermakna).
    Label window = nilai kolom 'type' pada baris TERAKHIR window."""
    if shuffle_seed is None:
        g = group_df.sort_values("ts")
    else:
        g = group_df.sample(frac=1, random_state=shuffle_seed)

    feat_values = g[feature_cols].to_numpy()
    type_values = g["type"].to_numpy()
    n = len(g)
    windows = []
    labels = []
    for start in range(0, n - window + 1, stride):
        windows.append(feat_values[start:start + window])
        labels.append(type_values[start + window - 1])
    return windows, labels
