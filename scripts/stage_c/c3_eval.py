"""
C3: evaluasi akhir pada data UJI, sekali jalan, setelah freeze_after_c2.json.
Hash setiap model diverifikasi terhadap freeze sebelum dipakai.
Label sesi sebenarnya = label sesi Tahap A; label sesi prediksi = suara
mayoritas prediksi window (seri -> indeks kelas terkecil).
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common_c import B2, C_RES, CLASSES, MODELS, N_CLS, build_inputs, load_rows, run_log, sha256  # noqa: E402

LAB = np.arange(N_CLS)
OUT_DONE = f"{C_RES}/c3_test_ambiguity.csv"


def metrics(y, pred, sess, sess_true):
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    d = dict(f1_macro=f1_score(y, pred, labels=LAB, average="macro", zero_division=0),
             f1_weighted=f1_score(y, pred, labels=LAB, average="weighted", zero_division=0))
    for i, c in enumerate(CLASSES):
        d.update({f"precision_{c}": pr[i], f"recall_{c}": rc[i], f"f1_{c}": f1[i]})
    df = pd.DataFrame(dict(s=sess, p=pred))
    vote = df.groupby("s").p.apply(lambda x: int(np.argmax(np.bincount(x, minlength=N_CLS))))
    st = np.array([sess_true[s] for s in vote.index])
    sf = f1_score(st, vote.values, labels=LAB, average=None, zero_division=0)
    d["f1_macro_sesi"] = f1_score(st, vote.values, labels=LAB, average="macro", zero_division=0)
    for i, c in enumerate(CLASSES):
        d[f"f1_sesi_{c}"] = sf[i]
    d["n_window"], d["n_sesi"] = len(y), len(vote)
    return d, confusion_matrix(y, pred, labels=LAB)


def row_hash(M):
    M = np.ascontiguousarray(np.round(np.asarray(M, dtype=np.float64).reshape(len(M), -1), 6) + 0.0)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in M],
                    dtype=np.uint64)


def main():
    if os.path.exists(OUT_DONE):
        run_log("C3 sudah ada, dilewati")
        return
    freeze = json.load(open(f"{C_RES}/freeze_after_c2.json"))["hash_sha256"]
    rows = load_rows()
    meta = rows[2]
    si = pd.read_pickle("artifacts/stage_a/session_index.pkl").set_index("session_id")
    sess_true = {s: CLASSES.index(c) for s, c in si.kelas_sesi.items()}
    sid_all = meta.session_id.to_numpy()

    cache = {}

    def data(kind, w, flat=False):
        key = (kind, w)
        if key not in cache:
            X, y, end = build_inputs(kind, w, "test", None, rows)
            cache[key] = (X, y, sid_all[end])
        X, y, s = cache[key]
        return (X.reshape(len(X), -1) if flat else X), y, s

    def predict(name):
        path = [f for f in freeze if os.path.basename(f).startswith(name + ".")][0]
        assert sha256(path) == freeze[path], f"hash berubah: {path}"
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

    perf, sess_rows, cms = [], [], {}
    groups = [("main", "main", 10, False), ("lstm1", "main", 10, False), ("mlp", "main", 10, True),
              ("rf", "main", 10, True), ("xgb", "main", 10, True)]
    for fam, kind, w, flat in groups:
        for s in (42, 43, 44):
            X, y, sess = data(kind, w, flat)
            d, cm = metrics(y, predict(f"{fam}_s{s}")(X), sess, sess_true)
            perf.append(dict(model=fam, seed=s, **d))
            cms[(fam, s)] = cm
    perf = pd.DataFrame(perf)

    def with_mean(df, keys):
        num = [c for c in df.columns if c not in keys + ["seed"]]
        agg = df.groupby(keys)[num].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        return agg.reset_index()

    perf.to_csv(f"{C_RES}/c3_test_performance.csv", index=False)
    with_mean(perf, ["model"]).to_csv(f"{C_RES}/c3_test_performance_ringkasan.csv", index=False)
    perf[["model", "seed", "f1_macro_sesi"] + [f"f1_sesi_{c}" for c in CLASSES] + ["n_sesi"]].to_csv(
        f"{C_RES}/c3_session_level.csv", index=False)

    def cm_df(cm):
        a = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        b = (100 * a.div(a.sum(axis=1).replace(0, np.nan), axis=0)).round(2)
        a.insert(0, "satuan", "jumlah")
        b.insert(0, "satuan", "persen_baris")
        return pd.concat([a, b]).rename_axis("label_sebenarnya")

    cm_df(cms[("main", 42)]).to_csv(f"{C_RES}/c3_confusion_seed42.csv")
    cm_df(np.mean([cms[("main", s)] for s in (42, 43, 44)], axis=0).round(2)).to_csv(f"{C_RES}/c3_confusion_mean.csv")

    # ablasi
    ab = []
    for w in (1, 5, 10, 20):
        for s in (42, 43, 44):
            X, y, sess = data("main", w)
            name = f"main_s{s}" if w == 10 else f"abl_w{w}_s{s}"
            d, _ = metrics(y, predict(name)(X), sess, sess_true)
            ab.append(dict(w=w, seed=s, f1_macro=d["f1_macro"], f1_weighted=d["f1_weighted"],
                           f1_macro_sesi=d["f1_macro_sesi"], n_window=d["n_window"]))
    ab = pd.DataFrame(ab)
    pd.concat([ab, with_mean(ab, ["w"]).assign(seed="rerata_sd")], ignore_index=True).to_csv(
        f"{C_RES}/c3_ablation_test.csv", index=False)

    # kontrol
    ct = []
    for s in (42, 43, 44):
        X, y, sess = data("control", 10)
        d, _ = metrics(y, predict(f"control_s{s}")(X), sess, sess_true)
        ct.append(dict(model="control", seed=s, **d))
    ct = pd.DataFrame(ct)
    pd.concat([ct, with_mean(ct, ["model"]).assign(seed="rerata_sd")], ignore_index=True).to_csv(
        f"{C_RES}/c3_control_test.csv", index=False)

    # ambiguitas uji
    h_tr = row_hash(np.load(f"{B2}/X_train.npy", mmap_mode="r"))
    y_tr = np.load(f"{B2}/y_train.npy").astype(int)
    lab_tr = {}
    for h, yy in zip(h_tr, y_tr):
        lab_tr.setdefault(h, set()).add(yy)
    X, y, _ = data("main", 10)
    h_te = row_hash(X)
    same = np.array([h in lab_tr for h in h_te])
    other = np.array([h in lab_tr and any(l != yy for l in lab_tr[h]) for h, yy in zip(h_te, y)])
    amb = []
    for i, c in enumerate(CLASSES + ["SEMUA"]):
        m = np.ones(len(y), bool) if c == "SEMUA" else y == i
        amb.append(dict(kelas=c, n_window_uji=int(m.sum()), persen_identik_dgn_latih=100 * same[m].mean(),
                        persen_identik_dgn_latih_label_lain=100 * other[m].mean()))
    pd.DataFrame(amb).to_csv(OUT_DONE, index=False)
    run_log("C3 selesai")


if __name__ == "__main__":
    main()
