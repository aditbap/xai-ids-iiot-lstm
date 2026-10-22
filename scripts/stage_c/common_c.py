"""
Modul bersama Tahap C. TIDAK mengimpor TensorFlow (setiap skrip pemanggil
mengatur urutan impor TF sendiri).

Baris (unit sebelum window) disimpan oleh c0_prep.py di artifacts/stage_c/rows/:
  rows_main.npy     float32 (n_baris, 22)  ruang masukan model utama
                    (kode kategorikal -> log1p 10 fitur -> Min-Max scaler_log.pkl)
  rows_srcport.npy  float32 (n_baris,)     src_port Min-Max (tanpa log, fit latih)
  rows_meta.csv     session_id, pos, partisi, label, kunci sesi
Baris terurut (session_id, pos); setiap sesi kontinu 0..L-1.
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = "."
A, B2, DL = "artifacts/stage_a", "artifacts/stage_b2", "artifacts/diag_log"
SMOKE = os.environ.get("STAGE_C_SMOKE")          # uji asap: direktori keluaran terpisah (bukan run resmi)
C_ART, C_RES = ("artifacts/stage_c", "results/stage_c") if not SMOKE else (f"{SMOKE}/art", f"{SMOKE}/res")
ROWS = "artifacts/stage_c/rows"
MODELS = f"{C_ART}/models"
RUNS = f"{C_RES}/c2_runs"
for d in (C_ART, C_RES, ROWS, MODELS, RUNS):
    os.makedirs(d, exist_ok=True)

CLASSES = ["backdoor", "ddos", "dos", "injection", "normal", "password", "ransomware", "scanning", "xss"]
N_CLS = 9
EPOCHS, BATCH, LR_MAIN = 12, 64, 0.001
LOG_FEATS = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes",
             "dst_pkts", "dst_ip_bytes", "http_request_body_len", "http_response_body_len"]


def run_log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(f"{C_RES}/run_log.txt", "a") as f:
        f.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows():
    Xm = np.load(f"{ROWS}/rows_main.npy")
    sp = np.load(f"{ROWS}/rows_srcport.npy")
    meta = pd.read_csv(f"{ROWS}/rows_meta.csv")
    return Xm, sp, meta


def shuffle_within_sessions(X, sid, seed):
    """Acak urutan baris di dalam setiap sesi (seed tetap). Posisi baru = indeks baru."""
    rng = np.random.default_rng(seed)
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    lens = np.diff(np.r_[starts, len(sid)])
    perm = np.concatenate([s + rng.permutation(L) for s, L in zip(starts, lens)])
    return X[perm], perm


def windows(X, lab, sid, pos, part, p, w):
    """Window stride 1 di dalam sesi, panjang w, untuk partisi p. Label = time-step terakhir."""
    end = np.flatnonzero((pos >= w - 1) & (part == p))
    assert (sid[end] == sid[end - (w - 1)]).all()
    return X[end[:, None] + np.arange(-(w - 1), 1)], lab[end], end


def build_inputs(kind, w, p, seed_shuffle=None, rows=None):
    """kind: 'main' (22 fitur) | 'control' (23 fitur, src_port di indeks 0).
    seed_shuffle: acak baris di dalam sesi (hanya untuk latih/validasi T6)."""
    Xm, sp, meta = rows if rows is not None else load_rows()
    X = np.concatenate([sp[:, None], Xm], axis=1) if kind == "control" else Xm
    lab = meta.label.to_numpy()
    sid, pos, part = meta.session_id.to_numpy(), meta.pos.to_numpy(), meta.partisi.to_numpy()
    if seed_shuffle is not None:
        assert p in ("train", "val")
        X, perm = shuffle_within_sessions(X, sid, seed_shuffle)
        lab = lab[perm]
    Xw, yw, end = windows(X, lab, sid, pos, part, p, w)
    return Xw.astype(np.float32), yw.astype(int), end


def jacc_topk(a, b, k):
    A_ = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    B_ = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(A_ & B_) / len(A_ | B_)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
