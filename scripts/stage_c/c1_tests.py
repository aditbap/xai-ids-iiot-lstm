"""
C1: uji validasi pipeline. Tidak ada prediksi pada data uji (T4 hanya hash).

T1 session_id lintas partisi; T2 kunci sesi (src_ip, dst_ip, dst_port, proto)
lintas partisi per kelas (label sesi), dihitung atas sesi final (sesudah
purging) semua kelas; T3 disalin dari Tahap A; T4 window identik (hash ruang
masukan model utama, 6 desimal) per kelas window partisi yang lebih akhir;
T5 regresi logistik pada window diratakan (F1-macro validasi).
T6 dikerjakan di C2 (c2_t6.csv) dan ditambahkan ke tabel ini oleh c2_summary.
"""

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from common_c import B2, C_RES, CLASSES, N_CLS, load_rows, run_log, save_json  # noqa: E402

OUT = f"{C_RES}/c1_validation_tests.csv"


def row_hash(M):
    M = np.ascontiguousarray(np.round(np.asarray(M, dtype=np.float64).reshape(len(M), -1), 6) + 0.0)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in M],
                    dtype=np.uint64)


def main():
    if os.path.exists(OUT):
        run_log("C1 sudah ada, dilewati")
        return
    t0 = time.time()
    rows = []
    _, _, meta = load_rows()
    meta = meta[meta.partisi.isin(["train", "val", "test"])]

    # T1
    sp = meta.groupby("session_id").partisi.nunique()
    rows.append(dict(uji="T1", kelas="SEMUA", metrik="n_session_id_di_>1_partisi", nilai=int((sp > 1).sum()),
                     status="Utama", lolos=bool((sp > 1).sum() == 0)))

    # T2
    s = meta.groupby("session_id").agg(partisi=("partisi", "first"), label=("label", "first"),
                                       src_ip=("src_ip", "first"), dst_ip=("dst_ip", "first"),
                                       dst_port=("dst_port", "first"), proto=("proto", "first"))
    si = pd.read_pickle("artifacts/stage_a/session_index.pkl").set_index("session_id")
    s["kelas_sesi"] = si.loc[s.index, "kelas_sesi"].values
    s["kunci"] = s.src_ip.astype(str) + "|" + s.dst_ip.astype(str) + "|" + s.dst_port.astype(str) + "|" + s.proto.astype(str)
    kp = s.groupby("kunci").partisi.agg(lambda x: "+".join(sorted(set(x))))
    s["partisi_kunci"] = s.kunci.map(kp)
    s["multi"] = s.partisi_kunci.str.contains(r"\+")
    for c in CLASSES + ["SEMUA"]:
        sub = s if c == "SEMUA" else s[s.kelas_sesi == c]
        kk = sub.drop_duplicates("kunci")
        src = sub[sub.multi].partisi_kunci.value_counts().to_dict()
        rows.append(dict(uji="T2", kelas=c, metrik="persen_kunci_di_>1_partisi", nilai=100 * kk.multi.mean(),
                         status="Dilaporkan", n=len(kk),
                         sumber="; ".join(f"{k}={v}" for k, v in src.items()) or "-"))
        rows.append(dict(uji="T2", kelas=c, metrik="persen_sesi_dgn_kunci_di_>1_partisi", nilai=100 * sub.multi.mean(),
                         status="Dilaporkan", n=len(sub)))

    # T3 (Tahap A)
    t3 = pd.read_csv("results/stage_a/t3_audit.csv")
    for tk, g in t3.groupby("tingkat"):
        rows.append(dict(uji="T3", kelas="SEMUA", metrik=f"{tk}_semua_batas_lolos", nilai=int(g.lolos.sum()),
                         n=len(g), status="Utama" if tk == "label_sesi" else "Dilaporkan", lolos=bool(g.lolos.all()),
                         sumber=f"results/stage_a/t3_audit.csv; window melanggar={int(g.n_window_melanggar.sum())}"))

    # T4 (hash saja)
    H, Y = {}, {}
    for p in ["train", "val", "test"]:
        H[p] = row_hash(np.load(f"{B2}/X_{p}.npy", mmap_mode="r"))
        Y[p] = np.load(f"{B2}/y_{p}.npy").astype(int)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        lab_a = {}
        for h, y in zip(H[a], Y[a]):
            lab_a.setdefault(h, set()).add(y)
        same = np.array([h in lab_a for h in H[b]])
        other = np.array([h in lab_a and any(l != y for l in lab_a[h]) for h, y in zip(H[b], Y[b])])
        for ci, c in enumerate(CLASSES + ["SEMUA"]):
            m = np.ones(len(Y[b]), bool) if c == "SEMUA" else Y[b] == ci
            rows.append(dict(uji="T4", kelas=c, metrik=f"persen_window_{b}_identik_dgn_{a}", nilai=100 * same[m].mean(),
                             n=int(m.sum()), status="Dilaporkan",
                             sumber=f"identik dgn label berbeda: {100 * other[m].mean():.4f}%"))

    # T5
    Xtr = np.load(f"{B2}/X_train.npy").reshape(-1, 220)
    Xva = np.load(f"{B2}/X_val.npy").reshape(-1, 220)
    t = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(Xtr, Y["train"])
    f1 = f1_score(Y["val"], lr.predict(Xva), average="macro", labels=np.arange(N_CLS), zero_division=0)
    rows.append(dict(uji="T5", kelas="SEMUA", metrik="f1_macro_validasi_logreg", nilai=f1, status="Sanity check",
                     sumber=f"multinomial lbfgs, max_iter 1000, class_weight balanced, seed 42; n_iter={int(lr.n_iter_[0])}; "
                            f"{time.time() - t:.0f}s"))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    run_log(f"C1 selesai ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
