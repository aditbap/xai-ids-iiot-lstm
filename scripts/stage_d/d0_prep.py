"""
D0: tulis results/stage_d/config_d.json (sekali; hash semua model & scaler yang
dipakai) dan siapkan baris nilai asli + baris prapemrosesan lama.

Pembacaan CSV identik dengan scripts/stage_c/c0_prep.py (block_rows Tahap A,
encoder Tahap A). Verifikasi wajib (berhenti bila gagal):
  - log1p + scaler_log pada rows_raw  == artifacts/stage_c/rows/rows_main.npy (persis)
  - scaler lama Tahap A pada rows_raw (+src_port), lalu buang src_port
    -> window w=10 == artifacts/stage_b/X_{train,val,test}.npy (persis)
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from pipeline_common import CHUNKSIZE, sorted_network_files  # noqa: E402
from common_d import (A, B2, D_ART, D_RES, ROWS, SB, C_RES_C, run_log, save_json, sha256,  # noqa: E402
                      windows)

DL = "artifacts/diag_log"
CONFIG = f"{D_RES}/config_d.json"


def write_config():
    if os.path.exists(CONFIG):
        return run_log("config_d.json sudah ada, tidak ditimpa")
    freeze = json.load(open(f"{C_RES_C}/freeze_after_c2.json"))["hash_sha256"]
    now = {p: sha256(p) for p in freeze}
    bad = [p for p in freeze if now[p] != freeze[p]]
    assert not bad, f"hash model Tahap C berubah: {bad}"
    extra = [f"{SB}/lstm_main_seed42.h5", f"{A}/scaler.pkl", f"{DL}/scaler_log.pkl", "artifacts/stage_c/scaler_srcport.pkl",
             f"{A}/encoder.pkl", f"{A}/main_450_index.csv", f"{A}/pilot_40_index.csv", f"{A}/block_rows.csv.gz",
             f"{A}/session_index.pkl", f"{ROWS}/rows_main.npy", f"{ROWS}/rows_srcport.npy", f"{ROWS}/rows_meta.csv",
             f"{B2}/X_train.npy", f"{B2}/deletion_baseline.csv", "artifacts/stage_c/c4_parts/bg_primary_idx.npy",
             f"{C_RES_C}/config_frozen.json", f"{C_RES_C}/freeze_after_c2.json"]
    cfg = dict(
        dibuat_pada=pd.Timestamp.now().isoformat(),
        tujuan="Tahap D: tiga gerbang bimbingan 5 atas eksperimen Tahap C (konfigurasi Tahap C tetap beku)",
        d1=dict(konfigurasi_lama="kode bilangan bulat -> Min-Max scaler Tahap A (artifacts/stage_a/scaler.pkl), tanpa log1p",
                seed_lama_42=f"{SB}/lstm_main_seed42.h5 (tidak dilatih ulang)", seed_lama_dilatih=[43, 44],
                protokol="identik scripts/stage_c/c2_train_one.py (mode eager, 12 epoch, batch 64, Adam 0.001, "
                         "class_weight balanced, epoch terbaik F1-macro val)",
                log1p="model utama Tahap C main_s42/43/44"),
        d2=dict(hash="blake2b 8 byte atas NILAI ASLI 22 fitur (kode kategorikal + numerik asli sebelum log1p/Min-Max), "
                     "tanpa pembulatan; -0.0 dinormalkan ke 0.0",
                tingkat=["window penuh 10x22", "time-step terakhir 22"]),
        d3=dict(eksklusi="seluruh sesi asal 40 instance pilot (artifacts/stage_a/pilot_40_index.csv) dikeluarkan dari uji",
                seleksi_450="kode round_robin Tahap A (scripts/stage_a/stage_a.py) dengan rng default_rng(42) dan urutan "
                            "langkah yang sama (termasuk pengambilan pilot yang mengonsumsi rng; hasil pilot dibuang)",
                c4="prosedur & konfigurasi C4 beku; background primer = artifacts/stage_c/c4_parts/bg_primary_idx.npy; "
                   "M2 sistem pada partisi uji final; M4 = 45 instance (5 pertama per kelas) seperti C4"),
        d4=dict(instance="90 = 10 pertama per kelas dari indeks 450 v2", model="main_s43, main_s44",
                metode=["deepshap", "lime_flat", "kustom_flat"]),
        hash_sha256_model_tahap_c=freeze,
        hash_sha256_lain={p: sha256(p) for p in extra},
        catatan="hash model D1 (lama seed 43/44) dicatat setelah pelatihan di results/stage_d/d1_models_hash.json",
    )
    save_json(cfg, CONFIG)
    run_log("config_d.json ditulis")


def prep_rows():
    if os.path.exists(f"{D_ART}/rows_old.npy"):
        return run_log("rows_raw/rows_old sudah ada, dilewati")
    FJ = json.load(open(f"{B2}/features.json"))
    feats, cat_cols = FJ["features"], FJ["categorical_columns"]
    codes = json.load(open(f"{A}/categorical_codes.json"))
    feats23 = codes["feature_order"]
    log_feats = json.load(open(f"{DL}/features_log.json"))["log_features"]
    log_idx = [feats.index(f) for f in log_feats]
    encoders = pickle.load(open(f"{A}/encoder.pkl", "rb"))
    sc_old = pickle.load(open(f"{A}/scaler.pkl", "rb"))
    sc_log = pickle.load(open(f"{DL}/scaler_log.pkl", "rb"))

    br = pd.read_csv(f"{A}/block_rows.csv.gz")
    want = {fi: set(g.raw_row) for fi, g in br.groupby("file_idx")}
    parts = []
    for fi, path in enumerate(sorted_network_files()):
        if fi not in want:
            continue
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype={"src_bytes": "string"}, low_memory=False):
            m = chunk.index.isin(want[fi])
            if m.any():
                sub = chunk.loc[m, feats23].copy()
                sub["file_idx"], sub["raw_row"] = fi, sub.index.to_numpy()
                parts.append(sub)
        run_log(f"D0 baca file {fi} selesai")
    D = pd.concat(parts, ignore_index=True)
    assert len(D) == len(br)
    D = br.merge(D, on=["file_idx", "raw_row"], how="left", validate="one_to_one")
    D = D.sort_values(["session_id", "pos"]).reset_index(drop=True)
    meta = pd.read_csv(f"{ROWS}/rows_meta.csv", usecols=["session_id", "pos", "partisi", "label"])
    assert (D.session_id.to_numpy() == meta.session_id.to_numpy()).all() and (D.pos.to_numpy() == meta.pos.to_numpy()).all()
    for c in cat_cols:
        D[c] = encoders[c].transform(D[[c]].astype(str).replace("-", "none")).ravel()
    X23 = D[feats23].astype("float64").to_numpy() + 0.0
    assert np.isfinite(X23).all()
    keep = [i for i, f in enumerate(feats23) if f != "src_port"]
    raw = np.ascontiguousarray(X23[:, keep])

    # verifikasi 1: prapemrosesan final
    Xl = raw.copy()
    Xl[:, log_idx] = np.log1p(Xl[:, log_idx])
    ok_main = bool(np.array_equal(sc_log.transform(Xl).astype(np.float32), np.load(f"{ROWS}/rows_main.npy")))
    # verifikasi 2: prapemrosesan lama
    old = sc_old.transform(X23)[:, keep].astype(np.float32)
    lab, sid = meta.label.to_numpy(), meta.session_id.to_numpy()
    pos, part = meta.pos.to_numpy(), meta.partisi.to_numpy()
    ok_old = {}
    for p in ("train", "val", "test"):
        Xw, yw, _ = windows(old, lab, sid, pos, part, p, 10)
        ok_old[p] = bool(np.array_equal(Xw, np.load(f"{SB}/X_{p}.npy")) and np.array_equal(yw, np.load(f"{SB}/y_{p}.npy")))
        del Xw
    assert ok_main and all(ok_old.values()), (ok_main, ok_old)
    np.save(f"{D_ART}/rows_raw.npy", raw)
    np.save(f"{D_ART}/rows_old.npy", old)
    save_json(dict(fitur=feats, identik_rows_main=ok_main, identik_stage_b=ok_old, n_baris=len(raw),
                   n_kode_minus1_per_fitur={c: int((raw[:, feats.index(c)] == -1).sum()) for c in cat_cols}),
              f"{D_ART}/rows_info_d.json")
    run_log(f"D0 rows_raw/rows_old disimpan; identik rows_main={ok_main}, identik stage_b={ok_old}")


if __name__ == "__main__":
    write_config()
    prep_rows()
