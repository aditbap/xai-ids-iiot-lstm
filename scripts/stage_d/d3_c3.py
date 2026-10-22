"""
D3.2: ulang seluruh C3 pada partisi uji FINAL (sesi pilot dikeluarkan), tanpa
pelatihan ulang. Prediksi diambil dari artifacts/stage_d/preds/ (dihasilkan
d_predict.py dengan cara memuat model identik C3 dan diverifikasi sama dengan
C3 pada uji penuh, lihat d_predict_check.csv). Fungsi metrik dan hash
ambiguitas DIIMPOR dari scripts/stage_c/c3_eval.py (metrics, row_hash).
Format keluaran sama dengan C3, berawalan d3_c3_.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_c"))
from c3_eval import metrics, row_hash  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common_d import A, B2, CLASSES, D_ART, D_RES, load_meta, pilot_sessions, run_log  # noqa: E402

PRED = f"{D_ART}/preds"
OUT_DONE = f"{D_RES}/d3_c3_test_ambiguity.csv"


def main():
    if os.path.exists(OUT_DONE):
        return run_log("D3 C3 sudah ada, dilewati")
    meta = load_meta()
    sid_all = meta.session_id.to_numpy()
    P = pilot_sessions()
    si = pd.read_pickle(f"{A}/session_index.pkl").set_index("session_id")
    sess_true = {s: CLASSES.index(c) for s, c in si.kelas_sesi.items()}

    def ev(name):
        z = np.load(f"{PRED}/test_{name}.npz")
        pred, end, y = z["pred"].astype(int), z["end"], z["y"].astype(int)
        sess = sid_all[end]
        keep = ~np.isin(sess, list(P))
        return metrics(y[keep], pred[keep], sess[keep], sess_true)

    def with_mean(df, keys):
        num = [c for c in df.columns if c not in keys + ["seed"]]
        agg = df.groupby(keys)[num].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        return agg.reset_index()

    perf, cms = [], {}
    for fam in ("main", "lstm1", "mlp", "rf", "xgb"):
        for s in (42, 43, 44):
            d, cm = ev(f"{fam}_s{s}")
            perf.append(dict(model=fam, seed=s, **d))
            cms[(fam, s)] = cm
    perf = pd.DataFrame(perf)
    perf.to_csv(f"{D_RES}/d3_c3_test_performance.csv", index=False)
    with_mean(perf, ["model"]).to_csv(f"{D_RES}/d3_c3_test_performance_ringkasan.csv", index=False)
    perf[["model", "seed", "f1_macro_sesi"] + [f"f1_sesi_{c}" for c in CLASSES] + ["n_sesi"]].to_csv(
        f"{D_RES}/d3_c3_session_level.csv", index=False)

    def cm_df(cm):
        a = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        b = (100 * a.div(a.sum(axis=1).replace(0, np.nan), axis=0)).round(2)
        a.insert(0, "satuan", "jumlah")
        b.insert(0, "satuan", "persen_baris")
        return pd.concat([a, b]).rename_axis("label_sebenarnya")

    cm_df(cms[("main", 42)]).to_csv(f"{D_RES}/d3_c3_confusion_seed42.csv")
    cm_df(np.mean([cms[("main", s)] for s in (42, 43, 44)], axis=0).round(2)).to_csv(f"{D_RES}/d3_c3_confusion_mean.csv")

    ab = []
    for w in (1, 5, 10, 20):
        for s in (42, 43, 44):
            d, _ = ev(f"main_s{s}" if w == 10 else f"abl_w{w}_s{s}")
            ab.append(dict(w=w, seed=s, f1_macro=d["f1_macro"], f1_weighted=d["f1_weighted"],
                           f1_macro_sesi=d["f1_macro_sesi"], n_window=d["n_window"]))
    ab = pd.DataFrame(ab)
    pd.concat([ab, with_mean(ab, ["w"]).assign(seed="rerata_sd")], ignore_index=True).to_csv(
        f"{D_RES}/d3_c3_ablation_test.csv", index=False)

    ct = []
    for s in (42, 43, 44):
        d, _ = ev(f"control_s{s}")
        ct.append(dict(model="control", seed=s, **d))
    ct = pd.DataFrame(ct)
    pd.concat([ct, with_mean(ct, ["model"]).assign(seed="rerata_sd")], ignore_index=True).to_csv(
        f"{D_RES}/d3_c3_control_test.csv", index=False)

    # ambiguitas uji (definisi C3: hash ruang model dibulatkan 6 desimal vs latih)
    h_tr = row_hash(np.load(f"{B2}/X_train.npy", mmap_mode="r"))
    y_tr = np.load(f"{B2}/y_train.npy").astype(int)
    lab_tr = {}
    for h, yy in zip(h_tr, y_tr):
        lab_tr.setdefault(h, set()).add(yy)
    X = np.load(f"{B2}/X_test.npy", mmap_mode="r")
    y = np.load(f"{B2}/y_test.npy").astype(int)
    z = np.load(f"{PRED}/test_main_s42.npz")
    keep = ~np.isin(sid_all[z["end"]], list(P))
    assert np.array_equal(z["y"].astype(int), y)
    h_te = row_hash(np.asarray(X)[keep])
    y = y[keep]
    same = np.array([h in lab_tr for h in h_te])
    other = np.array([h in lab_tr and any(l != yy for l in lab_tr[h]) for h, yy in zip(h_te, y)])
    amb = []
    for i, c in enumerate(CLASSES + ["SEMUA"]):
        m = np.ones(len(y), bool) if c == "SEMUA" else y == i
        amb.append(dict(kelas=c, n_window_uji=int(m.sum()), persen_identik_dgn_latih=100 * same[m].mean(),
                        persen_identik_dgn_latih_label_lain=100 * other[m].mean()))
    pd.DataFrame(amb).to_csv(OUT_DONE, index=False)
    run_log("D3 C3 selesai")


if __name__ == "__main__":
    main()
