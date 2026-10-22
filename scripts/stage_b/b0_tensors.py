"""
Tahap B0: bentuk tensor window (n, 10, 22) dari indeks Tahap A.
TIDAK membentuk ulang sesi/partisi: baris, sesi, posisi, dan partisi diambil
apa adanya dari artifacts/stage_a/block_rows.csv.gz; encoder dan scaler dari
artifacts/stage_a/. src_port dibuang SETELAH scaling (MinMax per kolom, jadi
membuang satu kolom tidak mengubah kolom lain).

Keluaran (artifacts/stage_b/):
  X_{train,val,test}.npy  float32 (n, 10, 22), ruang Min-Max
  y_{train,val,test}.npy  int8, indeks kelas alfabetis
  meta_{train,val,test}.csv  session_id, posisi_akhir_window, kelas_window
  features.json  urutan 22 fitur, indeks kategorikal, parameter scaler
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline_common import CHUNKSIZE, sorted_network_files  # noqa: E402

A = "artifacts/stage_a"
B = "artifacts/stage_b"
os.makedirs(B, exist_ok=True)
W = 10
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    summ = json.load(open("results/stage_a/summary.json"))
    classes = summ["classes"]
    codes = json.load(open(f"{A}/categorical_codes.json"))
    feats23, cat_cols = codes["feature_order"], codes["categorical_columns"]
    encoders = pickle.load(open(f"{A}/encoder.pkl", "rb"))
    scaler = pickle.load(open(f"{A}/scaler.pkl", "rb"))

    br = pd.read_csv(f"{A}/block_rows.csv.gz")
    log(f"block_rows: {len(br):,} baris")
    want = {fi: set(g.raw_row) for fi, g in br.groupby("file_idx")}

    parts = []
    for fi, path in enumerate(sorted_network_files()):
        if fi not in want:
            continue
        rows = want[fi]
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype={"src_bytes": "string"}, low_memory=False):
            m = chunk.index.isin(rows)
            if m.any():
                sub = chunk.loc[m, feats23 + ["type"]].copy()
                sub["file_idx"] = fi
                sub["raw_row"] = sub.index.to_numpy()
                parts.append(sub)
    D = pd.concat(parts, ignore_index=True)
    assert len(D) == len(br), (len(D), len(br))
    D = br.merge(D, on=["file_idx", "raw_row"], how="left", validate="one_to_one")
    D = D.sort_values(["session_id", "pos"]).reset_index(drop=True)
    log(f"baris terambil: {len(D):,}")

    for c in cat_cols:
        D[c] = encoders[c].transform(D[[c]].astype(str).replace("-", "none")).ravel()
    X23 = scaler.transform(D[feats23].astype("float64").to_numpy())
    keep = [i for i, f in enumerate(feats23) if f != "src_port"]
    feats22 = [feats23[i] for i in keep]
    Xs = X23[:, keep].astype(np.float32)
    lab = D["type"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int8)

    pos = D.pos.to_numpy()
    sid = D.session_id.to_numpy()
    # posisi harus kontinu 0..L-1 di tiap sesi
    first = np.r_[True, sid[1:] != sid[:-1]]
    assert (pos[first] == 0).all() and (np.diff(pos)[~first[1:]] == 1).all()

    end = np.flatnonzero(pos >= W - 1)
    assert (sid[end] == sid[end - (W - 1)]).all()
    part = D.partisi.to_numpy()[end]
    for p in ["train", "val", "test"]:
        e = end[part == p]
        X = Xs[e[:, None] + np.arange(-(W - 1), 1)]
        np.save(f"{B}/X_{p}.npy", X)
        np.save(f"{B}/y_{p}.npy", lab[e])
        pd.DataFrame(dict(session_id=sid[e], posisi_akhir_window=pos[e],
                          kelas_window=[classes[i] for i in lab[e]])).to_csv(f"{B}/meta_{p}.csv", index=False)
        log(f"{p}: X={X.shape}")

    cat_idx22 = [feats22.index(c) for c in cat_cols]
    json.dump(dict(classes=classes, features=feats22, categorical_columns=cat_cols,
                   categorical_index=cat_idx22,
                   scale_=scaler.scale_[keep].tolist(), min_=scaler.min_[keep].tolist(),
                   codes={c: codes["codes"][c] for c in cat_cols}),
              open(f"{B}/features.json", "w"), indent=1)
    log("selesai")


if __name__ == "__main__":
    main()
