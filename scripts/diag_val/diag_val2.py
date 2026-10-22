"""
Diagnostik validasi lanjutan: ambiguitas masukan. Hanya data LATIH dan
VALIDASI; data uji tidak dimuat. Tidak ada pelatihan ulang.

Pencocokan window: hash blake2b (64-bit) dari array masukan (10, 22) atau
(22,) di ruang Min-Max tanpa src_port, dibulatkan ke 6 desimal (float64,
-0.0 dinormalkan ke 0.0). Label mayoritas seri -> indeks kelas terkecil.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from itertools import combinations  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402

B, OUT = "artifacts/stage_b", "results/diag_val2"
os.makedirs(OUT, exist_ok=True)
FJ = json.load(open(f"{B}/features.json"))
CLASSES, FEATS = FJ["classes"], FJ["features"]
N = len(CLASSES)
LAB = np.arange(N)


def row_hash(M):
    M = np.round(np.asarray(M, dtype=np.float64).reshape(len(M), -1), 6) + 0.0
    M = np.ascontiguousarray(M)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little")
                     for r in M], dtype=np.uint64)


def label_sets(h, y):
    """hash -> Counter label."""
    d = {}
    for hh, yy in zip(h, y):
        d.setdefault(hh, Counter())[int(yy)] += 1
    return d


def ambiguity_val(h_tr, y_tr, h_va, y_va):
    tr = label_sets(h_tr, y_tr)
    va = label_sets(h_va, y_va)
    in_tr = np.array([h in tr for h in h_va])
    diff_tr = np.array([h in tr and any(l != y for l in tr[h]) for h, y in zip(h_va, y_va)])
    diff_va = np.array([any(l != y for l in va[h]) for h, y in zip(h_va, y_va)])
    rows = []
    for c in range(N):
        m = y_va == c
        rows.append(dict(kelas=CLASSES[c], n_window_val=int(m.sum()),
                         persen_identik_dgn_latih=100 * in_tr[m].mean(),
                         persen_identik_dgn_latih_label_beda=100 * diff_tr[m].mean(),
                         persen_identik_dgn_val_lain_label_beda=100 * diff_va[m].mean()))
    rows.append(dict(kelas="SEMUA", n_window_val=len(y_va), persen_identik_dgn_latih=100 * in_tr.mean(),
                     persen_identik_dgn_latih_label_beda=100 * diff_tr.mean(),
                     persen_identik_dgn_val_lain_label_beda=100 * diff_va.mean()))
    return pd.DataFrame(rows), tr


def ambiguity_train(h_tr, y_tr):
    tr = label_sets(h_tr, y_tr)
    multi = {h: c for h, c in tr.items() if len(c) > 1}
    n_win_multi = sum(sum(c.values()) for c in multi.values())
    summ = dict(n_window_latih=len(h_tr), n_hash_unik=len(tr), n_hash_multi_label=len(multi),
                n_window_di_hash_multi_label=n_win_multi, persen_window_di_hash_multi_label=100 * n_win_multi / len(h_tr))
    pair_h, pair_w = Counter(), Counter()
    for c in multi.values():
        for a, b in combinations(sorted(c), 2):
            pair_h[(a, b)] += 1
            pair_w[(a, b)] += c[a] + c[b]
    pairs = pd.DataFrame([dict(label_a=CLASSES[a], label_b=CLASSES[b], n_hash=pair_h[(a, b)],
                               n_window_kedua_label=pair_w[(a, b)]) for (a, b) in pair_h])
    if len(pairs):
        pairs = pairs.sort_values("n_window_kedua_label", ascending=False).head(10)
    return summ, pairs, tr


def write_train(summ, pairs, path):
    s = pd.DataFrame([dict(bagian="ringkasan", **summ)])
    p = pairs.assign(bagian="pasangan_label_top10")
    pd.concat([s, p], ignore_index=True).to_csv(path, index=False)


def to_original(v):
    """vektor 22 fitur Min-Max -> satuan asli; kategorikal sebagai nama."""
    sc, mn = np.array(FJ["scale_"]), np.array(FJ["min_"])
    raw = (np.asarray(v, dtype=np.float64) - mn) / sc
    out = {}
    for f, name in enumerate(FEATS):
        if f in FJ["categorical_index"]:
            inv = {code: k for k, code in FJ["codes"][name].items()}
            out[name] = inv.get(int(np.rint(raw[f])), f"<kode {int(np.rint(raw[f]))}>")
        else:
            out[name] = round(float(raw[f]), 4)
    return out


def main():
    X_tr = np.load(f"{B}/X_train.npy")
    y_tr = np.load(f"{B}/y_train.npy").astype(int)
    X_va = np.load(f"{B}/X_val.npy")
    y_va = np.load(f"{B}/y_val.npy").astype(int)

    h_tr, h_va = row_hash(X_tr), row_hash(X_va)
    l_tr, l_va = row_hash(X_tr[:, -1, :]), row_hash(X_va[:, -1, :])

    # 1. keluaran identik
    tf.compat.v1.set_random_seed(42)
    model = tf.keras.models.load_model(f"{B}/lstm_main_seed42.h5", compile=False)
    P = model.predict(X_va, batch_size=2048, verbose=0)
    hp = row_hash(P)
    df = pd.DataFrame(dict(hp=hp, y=y_va, pred=P.argmax(1), hin=h_va, pmax=P.max(1)))
    g = df.groupby("hp")
    top = g.size().sort_values(ascending=False).head(15)
    rows = []
    for k, (hk, n) in enumerate(top.items(), 1):
        s = df[df.hp == hk]
        dist = s.y.value_counts()
        rows.append(dict(peringkat=k, ukuran_kelompok=int(n), pmax=round(float(s.pmax.iloc[0]), 6),
                         kelas_prediksi=CLASSES[int(s.pred.iloc[0])],
                         distribusi_label_sebenarnya="; ".join(f"{CLASSES[i]}={c}" for i, c in dist.items()),
                         n_label_berbeda=int(len(dist)), n_window_unik_hash_masukan=int(s.hin.nunique())))
    pd.DataFrame(rows).to_csv(f"{OUT}/identical_outputs_val.csv", index=False)

    # 2-3. ambiguitas window penuh
    wa, tr_sets = ambiguity_val(h_tr, y_tr, h_va, y_va)
    wa.to_csv(f"{OUT}/window_ambiguity.csv", index=False)
    summ, pairs, _ = ambiguity_train(h_tr, y_tr)
    write_train(summ, pairs, f"{OUT}/train_ambiguity.csv")

    # 4. batas atas lookup
    maj = {h: max(sorted(c), key=lambda l: c[l]) for h, c in tr_sets.items()}
    m = np.array([h in maj for h in h_va])
    yp = np.array([maj[h] for h in h_va[m]])
    yt = y_va[m]
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=LAB, zero_division=0)
    lub = pd.DataFrame(dict(kelas=CLASSES, n_window_subset=sup, n_window_val_kelas=np.bincount(y_va, minlength=N),
                            precision=pr, recall=rc, F1=f1))
    lub.loc[len(lub)] = ["F1-macro", int(m.sum()), len(y_va), np.nan, np.nan,
                         f1_score(yt, yp, labels=LAB, average="macro", zero_division=0)]
    lub.to_csv(f"{OUT}/lookup_upper_bound.csv", index=False)

    # 5. time-step terakhir
    wa_l, _ = ambiguity_val(l_tr, y_tr, l_va, y_va)
    summ_l, pairs_l, _ = ambiguity_train(l_tr, y_tr)
    wa_l.insert(0, "bagian", "val_vs_latih")
    s = pd.DataFrame([dict(bagian="latih_ringkasan", **summ_l)])
    p = pairs_l.assign(bagian="latih_pasangan_label_top10")
    pd.concat([wa_l, s, p], ignore_index=True).to_csv(f"{OUT}/last_step_ambiguity.csv", index=False)

    # 6. contoh ambigu (window penuh, latih + validasi)
    allsets = label_sets(np.r_[h_tr, h_va], np.r_[y_tr, y_va])
    multi = sorted(((sum(c.values()), h) for h, c in allsets.items() if len(c) > 1), reverse=True)[:5]
    first = {}
    for arr_h, X in ((h_tr, X_tr), (h_va, X_va)):
        for i, hh in enumerate(arr_h):
            if hh in {h for _, h in multi} and hh not in first:
                first[hh] = X[i, -1, :]
    ex = []
    trs, vas = label_sets(h_tr, y_tr), label_sets(h_va, y_va)
    for k, (tot, hh) in enumerate(multi, 1):
        row = dict(peringkat=k, total_window=tot,
                   window_per_label_latih="; ".join(f"{CLASSES[l]}={c}" for l, c in sorted(trs.get(hh, {}).items())),
                   window_per_label_val="; ".join(f"{CLASSES[l]}={c}" for l, c in sorted(vas.get(hh, {}).items())))
        row.update(to_original(first[hh]))
        ex.append(row)
    pd.DataFrame(ex).to_csv(f"{OUT}/ambiguous_examples.csv", index=False)
    print("selesai")


if __name__ == "__main__":
    main()
