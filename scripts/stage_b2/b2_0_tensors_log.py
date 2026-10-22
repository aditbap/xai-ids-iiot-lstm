"""
Tahap B2-0: tensor window (n, 10, 22) dengan prapemrosesan final:
kode bilangan bulat (encoder Tahap A) -> log1p pada 10 fitur -> Min-Max
(artifacts/diag_log/scaler_log.pkl, TIDAK di-fit ulang). Baris/sesi/partisi
dari Tahap A. Tensor latih & validasi diverifikasi identik dengan
artifacts/diag_log/X_{train,val}_log.npy.

Juga menghitung nilai pengganti M2: median (numerik) / modus (kategorikal,
biner = numerik dengan <= 2 nilai unik) dari data latih dalam SATUAN ASLI,
dihitung atas seluruh window latih x time-step (definisi sama dengan Tahap
B), lalu diubah ke ruang masukan model dengan prapemrosesan yang sama.
"""

import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline_common import CHUNKSIZE, sorted_network_files  # noqa: E402

A, SB, DL, B2 = "artifacts/stage_a", "artifacts/stage_b", "artifacts/diag_log", "artifacts/stage_b2"
os.makedirs(B2, exist_ok=True)
W = 10
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    FJ = json.load(open(f"{SB}/features.json"))
    FL = json.load(open(f"{DL}/features_log.json"))
    classes, feats = FJ["classes"], FJ["features"]
    assert FL["features"] == feats
    cat_cols, cat_idx = FJ["categorical_columns"], FJ["categorical_index"]
    log_idx = [feats.index(f) for f in FL["log_features"]]
    encoders = pickle.load(open(f"{A}/encoder.pkl", "rb"))
    scaler = pickle.load(open(f"{DL}/scaler_log.pkl", "rb"))

    br = pd.read_csv(f"{A}/block_rows.csv.gz")
    br = br[br.partisi.isin(["train", "val", "test"])]
    want = {fi: set(g.raw_row) for fi, g in br.groupby("file_idx")}
    parts = []
    for fi, path in enumerate(sorted_network_files()):
        if fi not in want:
            continue
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype={"src_bytes": "string"}, low_memory=False):
            m = chunk.index.isin(want[fi])
            if m.any():
                sub = chunk.loc[m, feats + ["type"]].copy()
                sub["file_idx"], sub["raw_row"] = fi, sub.index.to_numpy()
                parts.append(sub)
    D = pd.concat(parts, ignore_index=True)
    assert len(D) == len(br)
    D = br.merge(D, on=["file_idx", "raw_row"], how="left", validate="one_to_one")
    D = D.sort_values(["session_id", "pos"]).reset_index(drop=True)
    for c in cat_cols:
        D[c] = encoders[c].transform(D[[c]].astype(str).replace("-", "none")).ravel()
    Xraw = D[feats].astype("float64").to_numpy()            # satuan asli (kategorikal = kode)

    def prep(M):
        M = np.array(M, dtype=np.float64)
        M[..., log_idx] = np.log1p(M[..., log_idx])
        return scaler.transform(M.reshape(-1, len(feats))).reshape(M.shape)

    Xs = prep(Xraw).astype(np.float32)
    lab = D["type"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int8)
    pos, sid = D.pos.to_numpy(), D.session_id.to_numpy()
    end = np.flatnonzero(pos >= W - 1)
    assert (sid[end] == sid[end - (W - 1)]).all()
    part = D.partisi.to_numpy()[end]
    check = {}
    for p in ["train", "val", "test"]:
        e = end[part == p]
        win = e[:, None] + np.arange(-(W - 1), 1)
        X = Xs[win]
        np.save(f"{B2}/X_{p}.npy", X)
        np.save(f"{B2}/y_{p}.npy", lab[e])
        assert (lab[e] == np.load(f"{SB}/y_{p}.npy")).all()
        shutil.copy(f"{SB}/meta_{p}.csv", f"{B2}/meta_{p}.csv")
        meta = pd.read_csv(f"{SB}/meta_{p}.csv")
        assert (meta.session_id.to_numpy() == sid[e]).all() and (meta.posisi_akhir_window.to_numpy() == pos[e]).all()
        if p in ("train", "val"):
            check[f"identik_dgn_diag_log_{p}"] = bool(np.array_equal(X, np.load(f"{DL}/X_{p}_log.npy")))
        if p == "train":
            flat = Xraw[win].reshape(-1, len(feats))         # semua window latih x time-step, satuan asli
            base_raw, kind = np.zeros(len(feats)), []
            for f in range(len(feats)):
                u, cnt = np.unique(flat[:, f], return_counts=True)
                if f in cat_idx or len(u) <= 2:
                    base_raw[f] = u[np.argmax(cnt)]
                    kind.append("kategorikal_modus" if f in cat_idx else "biner_modus")
                else:
                    base_raw[f] = np.median(flat[:, f])
                    kind.append("numerik_median")
            del flat
            base_model = prep(base_raw[None])[0]
            pd.DataFrame(dict(fitur=feats, jenis=kind, nilai_asli=base_raw, nilai_ruang_model=base_model)).to_csv(
                f"{B2}/deletion_baseline.csv", index=False)
        log(f"{p}: {X.shape}")
    print(check)
    assert all(check.values()), check

    json.dump(dict(FJ, scale_=scaler.scale_.tolist(), min_=scaler.min_.tolist(), log_features=FL["log_features"],
                   log_index=log_idx, scaler="artifacts/diag_log/scaler_log.pkl",
                   verifikasi_tensor=check), open(f"{B2}/features.json", "w"), indent=1)
    log("selesai")


if __name__ == "__main__":
    main()
