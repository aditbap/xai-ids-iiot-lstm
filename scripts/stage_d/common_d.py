"""
Modul bersama Tahap D (tiga gerbang bimbingan 5). TIDAK mengimpor TensorFlow.
Memakai ulang common_c (baris, window, hash) apa adanya; keluaran Tahap D ke
results/stage_d/ dan artifacts/stage_d/. Tidak ada berkas Tahap A-C yang ditimpa.

Baris tambahan (dibuat d0_prep.py, urutan identik dengan rows_meta.csv):
  rows_raw.npy  float64 (n_baris, 22)  NILAI ASLI 22 fitur model utama:
                kode kategorikal bilangan bulat (encoder Tahap A) dan nilai
                numerik asli sebelum log1p/Min-Max
  rows_old.npy  float32 (n_baris, 22)  prapemrosesan LAMA (kode -> Min-Max
                scaler Tahap A, tanpa log1p); window = artifacts/stage_b/X_*
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_c"))
from common_c import (A, B2, CLASSES, N_CLS, ROWS, load_rows, save_json, sha256,  # noqa: E402,F401
                      windows)
from common_c import build_inputs as build_inputs_c  # noqa: E402,F401

SB = "artifacts/stage_b"
C_ART_C, C_RES_C = "artifacts/stage_c", "results/stage_c"
D_ART, D_RES = "artifacts/stage_d", "results/stage_d"
D_MODELS = f"{D_ART}/models"
D_RUNS = f"{D_RES}/d1_runs"
for d in (D_ART, D_RES, D_MODELS, D_RUNS):
    os.makedirs(d, exist_ok=True)


def run_log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(f"{D_RES}/run_log.txt", "a") as f:
        f.write(line + "\n")


def load_meta():
    return pd.read_csv(f"{ROWS}/rows_meta.csv")


def window_ends(meta, p, w=10):
    pos, part = meta.pos.to_numpy(), meta.partisi.to_numpy()
    return np.flatnonzero((pos >= w - 1) & (part == p))


def pilot_sessions():
    return set(pd.read_csv(f"{A}/pilot_40_index.csv").session_id)
