"""
Diagnostik model artifacts/stage_b/lstm_main_seed42.h5 pada data VALIDASI
(partisi Tahap A sesudah purging). Hanya pengukuran: tidak melatih ulang,
tidak memilih ulang epoch. Data uji tidak dimuat.

Definisi:
- Label sesi sebenarnya = label sesi Tahap A (session_index); label sesi
  prediksi = suara mayoritas prediksi window di sesi itu, seri -> indeks kelas
  terkecil (urutan alfabetis).
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

B, OUT = "artifacts/stage_b", "results/diag_val"
os.makedirs(OUT, exist_ok=True)

FJ = json.load(open(f"{B}/features.json"))
CLASSES = FJ["classes"]
N = len(CLASSES)
LAB = np.arange(N)


def main():
    X = np.load(f"{B}/X_val.npy")
    y = np.load(f"{B}/y_val.npy").astype(int)
    meta = pd.read_csv(f"{B}/meta_val.csv")
    assert len(X) == len(y) == len(meta)

    tf.compat.v1.set_random_seed(42)
    model = tf.keras.models.load_model(f"{B}/lstm_main_seed42.h5", compile=False)
    P = model.predict(X, batch_size=2048, verbose=0)
    pred = P.argmax(1)

    # 1. per kelas
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    n_sess = meta.groupby("kelas_window").session_id.nunique().reindex(CLASSES, fill_value=0).values
    pc = pd.DataFrame(dict(kelas=CLASSES, n_window_validasi=sup, n_sesi_validasi=n_sess,
                           precision=pr, recall=rc, F1=f1, support=sup))
    for avg in ("macro", "weighted"):
        pc.loc[len(pc)] = [f"F1-{avg}", len(y), int(n_sess.sum()), np.nan, np.nan,
                           f1_score(y, pred, labels=LAB, average=avg, zero_division=0), len(y)]
    pc.to_csv(f"{OUT}/per_class_val.csv", index=False)

    # 2. confusion
    cm = confusion_matrix(y, pred, labels=LAB)
    cmd = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cmp_ = (100 * cmd.div(cmd.sum(1).replace(0, np.nan), axis=0)).round(2)
    cmd.insert(0, "satuan", "jumlah")
    cmp_.insert(0, "satuan", "persen_baris")
    pd.concat([cmd, cmp_]).rename_axis("label_sebenarnya").to_csv(f"{OUT}/confusion_val.csv")

    # 3. top confusions
    off = [(CLASSES[i], CLASSES[j], int(cm[i, j]), 100 * cm[i, j] / cm[i].sum())
           for i in range(N) for j in range(N) if i != j and cm[i, j] > 0]
    tc = pd.DataFrame(off, columns=["label_sebenarnya", "prediksi", "jumlah_salah", "persen_dari_kelas_sebenarnya"])
    tc = tc.sort_values("jumlah_salah", ascending=False).head(10)
    tc.to_csv(f"{OUT}/top_confusions.csv", index=False)

    # 4. per sesi
    meta = meta.assign(y=y, pred=pred, benar=(y == pred))
    ps = meta.groupby("session_id").agg(kelas=("kelas_window", "first"), n_window=("y", "size"),
                                        akurasi_window=("benar", "mean")).reset_index()
    ps.to_csv(f"{OUT}/per_session_val.csv", index=False)
    summ = ps.groupby("kelas").agg(n_sesi=("session_id", "size"), median_akurasi_sesi=("akurasi_window", "median"),
                                   min_akurasi_sesi=("akurasi_window", "min"),
                                   n_sesi_akurasi_lt_0_5=("akurasi_window", lambda s: int((s < 0.5).sum())))
    summ.reindex(CLASSES).reset_index().to_csv(f"{OUT}/per_session_val_ringkasan_kelas.csv", index=False)

    # 5. tingkat sesi (suara mayoritas)
    si = pd.read_pickle("artifacts/stage_a/session_index.pkl").set_index("session_id")
    sv = meta.groupby("session_id").pred.apply(lambda s: int(np.argmax(np.bincount(s, minlength=N))))
    true_s = np.array([CLASSES.index(si.loc[s, "kelas_sesi"]) for s in sv.index])
    ps_, rs_, fs_, ss_ = precision_recall_fscore_support(true_s, sv.values, labels=LAB, zero_division=0)
    sl = pd.DataFrame(dict(kelas=CLASSES, n_sesi=ss_, precision=ps_, recall=rs_, F1=fs_))
    sl.loc[len(sl)] = ["F1-macro", int(ss_.sum()), np.nan, np.nan,
                       f1_score(true_s, sv.values, labels=LAB, average="macro", zero_division=0)]
    sl.to_csv(f"{OUT}/session_level_val.csv", index=False)

    # 6. distribusi prediksi
    pd.DataFrame(dict(kelas=CLASSES, n_label_sebenarnya=np.bincount(y, minlength=N),
                      n_prediksi=np.bincount(pred, minlength=N))).assign(
        selisih=lambda d: d.n_prediksi - d.n_label_sebenarnya).to_csv(f"{OUT}/pred_distribution_val.csv", index=False)

    # 7. bobot kelas (dari log pelatihan seed 42)
    cw = json.load(open("results/stage_b/train_log.json"))["class_weight"]
    ytr = np.load(f"{B}/y_train.npy").astype(int)
    pd.DataFrame(dict(kelas=CLASSES, n_window_latih=np.bincount(ytr, minlength=N),
                      bobot=[cw[str(i)] for i in range(N)],
                      sumber="results/stage_b/train_log.json (compute_class_weight balanced)")).to_csv(
        f"{OUT}/class_weights_used.csv", index=False)

    # 8. keyakinan
    pmax = P.max(1)
    rows = []
    for i, c in enumerate(CLASSES):
        for name, m in (("benar", (y == i) & (pred == i)), ("salah", (y == i) & (pred != i))):
            rows.append(dict(kelas=c, prediksi=name, n=int(m.sum()),
                             mean_pmax=pmax[m].mean() if m.any() else np.nan,
                             median_pmax=np.median(pmax[m]) if m.any() else np.nan))
    pd.DataFrame(rows).to_csv(f"{OUT}/confidence_val.csv", index=False)
    print("selesai")


if __name__ == "__main__":
    main()
