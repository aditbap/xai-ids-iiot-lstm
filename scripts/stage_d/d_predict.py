"""
D: prediksi (argmax) semua model yang dibutuhkan D1-D3, mode graph (urutan
impor wajib), cara memuat identik dengan scripts/stage_c/c3_eval.py
(clear_session, set_random_seed(42), load_model compile=False, batch 2048).
Hash model Tahap C diverifikasi terhadap freeze_after_c2.json; model D1
terhadap results/stage_d/d1_models_hash.json.

Keluaran per model: artifacts/stage_d/preds/{partisi}_{nama}.npz
  pred (int8), end (indeks baris akhir window pada rows_meta), y (int8)
Validasi: main_s42/43/44 + old_s42/43/44. Uji: seluruh model C3 (utama,
lstm1, mlp, rf, xgb, kontrol, ablasi w=1/5/20; 3 seed).
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import json  # noqa: E402
import sys  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common_d import (C_RES_C, D_ART, D_RES, SB, build_inputs_c, load_rows, run_log, sha256,  # noqa: E402
                      windows)

PRED = f"{D_ART}/preds"
os.makedirs(PRED, exist_ok=True)


def model_path(name):
    if name == "old_s42":
        p = f"{SB}/lstm_main_seed42.h5"
        return p, json.load(open(f"{D_RES}/config_d.json"))["hash_sha256_lain"][p]
    if name.startswith("old_"):
        p = f"{D_ART}/models/{name}.h5"
        return p, json.load(open(f"{D_RES}/d1_models_hash.json"))[p]
    freeze = json.load(open(f"{C_RES_C}/freeze_after_c2.json"))["hash_sha256"]
    p = [f for f in freeze if os.path.basename(f).startswith(name + ".")][0]
    return p, freeze[p]


def predictor(name):
    path, h = model_path(name)
    if h is not None:
        assert sha256(path) == h, f"hash berubah: {path}"
    if path.endswith(".h5"):
        tf.compat.v1.keras.backend.clear_session()
        tf.compat.v1.set_random_seed(42)
        m = tf.keras.models.load_model(path, compile=False)
        return lambda X: m.predict(X, batch_size=2048, verbose=0).argmax(1)
    if path.endswith(".joblib"):
        m = joblib.load(path)
        return lambda X: m.predict(X)
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(path)
    return lambda X: m.predict(X)


def main():
    rows = load_rows()
    meta = rows[2]
    lab, sid = meta.label.to_numpy(), meta.session_id.to_numpy()
    pos, part = meta.pos.to_numpy(), meta.partisi.to_numpy()
    old_rows = np.load(f"{D_ART}/rows_old.npy")
    jobs = [("val", f"{k}_s{s}", k, 10, False) for k in ("main", "old") for s in (42, 43, 44)]
    for fam, flat in (("main", False), ("lstm1", False), ("mlp", True), ("rf", True), ("xgb", True)):
        jobs += [("test", f"{fam}_s{s}", "main", 10, flat) for s in (42, 43, 44)]
    jobs += [("test", f"control_s{s}", "control", 10, False) for s in (42, 43, 44)]
    jobs += [("test", f"abl_w{w}_s{s}", "main", w, False) for w in (1, 5, 20) for s in (42, 43, 44)]
    cache = {}
    for p, name, kind, w, flat in jobs:
        out = f"{PRED}/{p}_{name}.npz"
        if os.path.exists(out):
            continue
        key = (p, kind, w)
        if key not in cache:
            cache.clear()
            if kind == "old":
                X, y, end = windows(old_rows, lab, sid, pos, part, p, w)
                cache[key] = (X.astype(np.float32), y.astype(int), end)
            else:
                cache[key] = build_inputs_c(kind, w, p, None, rows)
        X, y, end = cache[key]
        Xin = X.reshape(len(X), -1) if flat else X
        pred = predictor(name)(Xin)
        np.savez_compressed(out, pred=pred.astype(np.int8), end=end, y=y.astype(np.int8))
        run_log(f"D prediksi {p} {name}: n={len(y)}")
    run_log("D prediksi selesai")


if __name__ == "__main__":
    main()
