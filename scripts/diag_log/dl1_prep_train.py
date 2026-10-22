"""
Uji satu perubahan prapemrosesan: log1p sebelum Min-Max. HANYA latih + validasi.

Baris, sesi, posisi, partisi: dari artifacts/stage_a/block_rows.csv.gz (tidak
dibentuk ulang). Encoder kategorikal: artifacts/stage_a/encoder.pkl. log1p
pada 10 fitur byte/paket/durasi, lalu MinMaxScaler baru di-fit pada baris
latih (sama seperti Tahap A, fit pada baris latih). src_port tidak dipakai.

Pelatihan identik dengan scripts/stage_b/b1_train.py (arsitektur, Adam 0,001,
categorical CE, batch 64, 12 epoch, class_weight balanced, seed 42 dikunci
tepat sebelum model dibangun, TF_DETERMINISTIC_OPS, thread 4/4, pemilihan
epoch = F1-macro validasi terbaik).

Aturan keputusan (ditetapkan sebelum run): log1p dipakai untuk eksperimen
penuh bila F1-macro validasi baru > 0,6326; bila tidak, prapemrosesan lama
dipertahankan. Tidak ada percobaan ketiga.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

import json
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline_common import CHUNKSIZE, sorted_network_files  # noqa: E402

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.experimental.enable_op_determinism()

A, SB, ART, RES = "artifacts/stage_a", "artifacts/stage_b", "artifacts/diag_log", "results/diag_log"
os.makedirs(ART, exist_ok=True)
os.makedirs(RES, exist_ok=True)
LOG_FEATS = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes",
             "dst_pkts", "dst_ip_bytes", "http_request_body_len", "http_response_body_len"]
DECISION = ("log1p dipakai untuk eksperimen penuh bila F1-macro validasi baru > 0,6326; "
            "bila tidak, prapemrosesan lama dipertahankan. Tidak ada percobaan ketiga.")
SEED, EPOCHS, BATCH, N_CLS, W = 42, 12, 64, 9, 10
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def build_tensors():
    FJ = json.load(open(f"{SB}/features.json"))
    classes, feats22 = FJ["classes"], FJ["features"]
    cat_cols = FJ["categorical_columns"]
    encoders = pickle.load(open(f"{A}/encoder.pkl", "rb"))
    br = pd.read_csv(f"{A}/block_rows.csv.gz")
    br = br[br.partisi.isin(["train", "val"])]           # data uji tidak dimuat
    want = {fi: set(g.raw_row) for fi, g in br.groupby("file_idx")}
    parts = []
    for fi, path in enumerate(sorted_network_files()):
        if fi not in want:
            continue
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype={"src_bytes": "string"}, low_memory=False):
            m = chunk.index.isin(want[fi])
            if m.any():
                sub = chunk.loc[m, feats22 + ["type"]].copy()
                sub["file_idx"], sub["raw_row"] = fi, sub.index.to_numpy()
                parts.append(sub)
    D = pd.concat(parts, ignore_index=True)
    assert len(D) == len(br)
    D = br.merge(D, on=["file_idx", "raw_row"], how="left", validate="one_to_one")
    D = D.sort_values(["session_id", "pos"]).reset_index(drop=True)
    for c in cat_cols:
        D[c] = encoders[c].transform(D[[c]].astype(str).replace("-", "none")).ravel()
    Xr = D[feats22].astype("float64").to_numpy()
    li = [feats22.index(f) for f in LOG_FEATS]
    mins = Xr[:, li].min(0)
    assert (mins >= 0).all(), dict(zip(LOG_FEATS, mins))   # log1p butuh nilai tidak negatif
    Xr[:, li] = np.log1p(Xr[:, li])
    tr_rows = (D.partisi == "train").to_numpy()
    scaler = MinMaxScaler().fit(Xr[tr_rows])
    pickle.dump(scaler, open(f"{ART}/scaler_log.pkl", "wb"))
    Xs = scaler.transform(Xr).astype(np.float32)
    lab = D["type"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int8)
    pos, sid = D.pos.to_numpy(), D.session_id.to_numpy()
    end = np.flatnonzero(pos >= W - 1)
    assert (sid[end] == sid[end - (W - 1)]).all()
    part = D.partisi.to_numpy()[end]
    out = {}
    for p in ["train", "val"]:
        e = end[part == p]
        out[p] = (Xs[e[:, None] + np.arange(-(W - 1), 1)], lab[e])
        # urutan window harus sama dengan tensor Tahap B (label & meta identik)
        assert (lab[e] == np.load(f"{SB}/y_{p}.npy")).all()
        meta_b = pd.read_csv(f"{SB}/meta_{p}.csv")
        assert (meta_b.session_id.to_numpy() == sid[e]).all() and (meta_b.posisi_akhir_window.to_numpy() == pos[e]).all()
        np.save(f"{ART}/X_{p}_log.npy", out[p][0])
        log(f"{p}: {out[p][0].shape}")
    json.dump(dict(log_features=LOG_FEATS, features=feats22, scale_=scaler.scale_.tolist(),
                   min_=scaler.min_.tolist(), data_min_=scaler.data_min_.tolist(),
                   data_max_=scaler.data_max_.tolist(), min_asli_fitur_log=dict(zip(LOG_FEATS, mins.tolist()))),
              open(f"{ART}/features_log.json", "w"), indent=1)
    return out


class BestValF1(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.best_f1, self.best_epoch, self.best_w = -1.0, -1, None
        self.curve, self._t = [], None

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, batch_size=2048, verbose=0).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro", labels=np.arange(N_CLS), zero_division=0)
        self.curve.append(dict(epoch=epoch + 1, train_loss=float(logs["loss"]), val_f1_macro=float(f1),
                               detik=round(time.time() - self._t, 1)))
        if f1 > self.best_f1:
            self.best_f1, self.best_epoch = f1, epoch + 1
            self.best_w = [w.copy() for w in self.model.get_weights()]
        print(f"epoch {epoch + 1}/{EPOCHS} loss={logs['loss']:.4f} val_f1_macro={f1:.4f}", flush=True)


def main():
    data = build_tensors()
    (X_tr, y_tr), (X_va, y_va) = data["train"], data["val"]
    y_tr, y_va = y_tr.astype(int), y_va.astype(int)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLS), y=y_tr)
    cw = {i: float(w) for i, w in enumerate(cw)}

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 22)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(N_CLS, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy")
    cb = BestValF1(X_va, y_va)
    t0 = time.time()
    model.fit(X_tr, tf.keras.utils.to_categorical(y_tr, N_CLS), epochs=EPOCHS, batch_size=BATCH,
              class_weight=cw, callbacks=[cb], verbose=0, shuffle=True)
    t_train = time.time() - t0
    model.set_weights(cb.best_w)
    model.save(f"{ART}/lstm_main_seed42_log.h5")
    pd.DataFrame(cb.curve).to_csv(f"{RES}/train_curve.csv", index=False)
    json.dump(dict(aturan_keputusan=DECISION, seed=SEED, class_weight=cw, curve=cb.curve,
                   epoch_terpilih=cb.best_epoch, val_f1_macro_terpilih=float(cb.best_f1),
                   waktu_latih_detik=round(t_train, 1)), open(f"{RES}/train_log.json", "w"), indent=1)
    print(f"epoch terpilih {cb.best_epoch}, val F1 {cb.best_f1:.4f}, waktu {t_train:.0f}s")


if __name__ == "__main__":
    main()
