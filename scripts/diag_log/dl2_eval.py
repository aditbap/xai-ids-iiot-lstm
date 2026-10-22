"""
Evaluasi VALIDASI model log1p (artifacts/diag_log/lstm_main_seed42_log.h5)
dan perbandingan dengan model lama. Data uji tidak dimuat.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diag_val"))
from diag_val2 import row_hash  # noqa: E402

SB, ART, RES = "artifacts/stage_b", "artifacts/diag_log", "results/diag_log"
FJ = json.load(open(f"{SB}/features.json"))
CLASSES, FEATS = FJ["classes"], FJ["features"]
N = len(CLASSES)
LAB = np.arange(N)
OLD_MACRO = 0.6326


def main():
    # 1. cek presisi src_ip_bytes
    f = FEATS.index("src_ip_bytes")
    new = pickle.load(open(f"{ART}/scaler_log.pkl", "rb"))
    vals = np.array([40, 44, 52, 60, 1500], dtype=np.float64)
    old_s = vals * FJ["scale_"][f] + FJ["min_"][f]
    new_s = np.log1p(vals) * new.scale_[f] + new.min_[f]
    pd.DataFrame(dict(src_ip_bytes=vals.astype(int), minmax_lama=old_s, minmax_lama_float32=old_s.astype(np.float32),
                      log1p_minmax_baru=new_s, log1p_minmax_baru_float32=new_s.astype(np.float32))
                 ).to_csv(f"{RES}/precision_check.csv", index=False)

    X = np.load(f"{ART}/X_val_log.npy")
    y = np.load(f"{SB}/y_val.npy").astype(int)
    meta = pd.read_csv(f"{SB}/meta_val.csv")
    tf.compat.v1.set_random_seed(42)
    model = tf.keras.models.load_model(f"{ART}/lstm_main_seed42_log.h5", compile=False)
    P = model.predict(X, batch_size=2048, verbose=0)
    pred = P.argmax(1)

    # 3. per kelas + confusion (format = results/diag_val)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    n_sess = meta.groupby("kelas_window").session_id.nunique().reindex(CLASSES, fill_value=0).values
    pc = pd.DataFrame(dict(kelas=CLASSES, n_window_validasi=sup, n_sesi_validasi=n_sess,
                           precision=pr, recall=rc, F1=f1, support=sup))
    macro = f1_score(y, pred, labels=LAB, average="macro", zero_division=0)
    for avg in ("macro", "weighted"):
        pc.loc[len(pc)] = [f"F1-{avg}", len(y), int(n_sess.sum()), np.nan, np.nan,
                           f1_score(y, pred, labels=LAB, average=avg, zero_division=0), len(y)]
    pc.to_csv(f"{RES}/per_class_val_log.csv", index=False)
    cm = confusion_matrix(y, pred, labels=LAB)
    cmd = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cmp_ = (100 * cmd.div(cmd.sum(axis=1).replace(0, np.nan), axis=0)).round(2)
    cmd.insert(0, "satuan", "jumlah")
    cmp_.insert(0, "satuan", "persen_baris")
    pd.concat([cmd, cmp_]).rename_axis("label_sebenarnya").to_csv(f"{RES}/confusion_val_log.csv")

    # tingkat sesi (definisi sama dengan diag_val)
    si = pd.read_pickle("artifacts/stage_a/session_index.pkl").set_index("session_id")
    sv = meta.assign(pred=pred).groupby("session_id").pred.apply(lambda s: int(np.argmax(np.bincount(s, minlength=N))))
    true_s = np.array([CLASSES.index(si.loc[s, "kelas_sesi"]) for s in sv.index])
    sess_macro = f1_score(true_s, sv.values, labels=LAB, average="macro", zero_division=0)

    # 4. perbandingan
    old = pd.read_csv("results/diag_val/per_class_val.csv").set_index("kelas")
    old_sess = pd.read_csv("results/diag_val/session_level_val.csv").set_index("kelas").loc["F1-macro", "F1"]
    cmpdf = pd.DataFrame(dict(kelas=CLASSES, F1_val_lama=old.loc[CLASSES, "F1"].values, F1_val_baru=f1))
    cmpdf["selisih"] = cmpdf.F1_val_baru - cmpdf.F1_val_lama
    cmpdf.loc[len(cmpdf)] = ["F1-macro (window)", old.loc["F1-macro", "F1"], macro, macro - old.loc["F1-macro", "F1"]]
    cmpdf.loc[len(cmpdf)] = ["F1-macro (sesi)", old_sess, sess_macro, sess_macro - old_sess]
    cmpdf.to_csv(f"{RES}/compare_val.csv", index=False)

    # 5. keluaran identik
    hp, h_in = row_hash(P), row_hash(X)
    df = pd.DataFrame(dict(hp=hp, y=y, pred=pred, hin=h_in, pmax=P.max(1)))
    rows = []
    for k, (hk, n) in enumerate(df.groupby("hp").size().sort_values(ascending=False).head(15).items(), 1):
        s = df[df.hp == hk]
        dist = s.y.value_counts()
        rows.append(dict(peringkat=k, ukuran_kelompok=int(n), pmax=round(float(s.pmax.iloc[0]), 6),
                         kelas_prediksi=CLASSES[int(s.pred.iloc[0])],
                         distribusi_label_sebenarnya="; ".join(f"{CLASSES[i]}={c}" for i, c in dist.items()),
                         n_label_berbeda=int(len(dist)), n_window_unik_hash_masukan=int(s.hin.nunique())))
    pd.DataFrame(rows).to_csv(f"{RES}/identical_outputs_val_log.csv", index=False)

    tl = json.load(open(f"{RES}/train_log.json"))
    tl["hasil_aturan_keputusan"] = dict(f1_macro_val_baru=float(macro), ambang=OLD_MACRO,
                                        log1p_dipakai=bool(macro > OLD_MACRO))
    json.dump(tl, open(f"{RES}/train_log.json", "w"), indent=1)
    print("selesai")


if __name__ == "__main__":
    main()
