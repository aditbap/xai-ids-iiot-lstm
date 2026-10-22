"""
C4 kontrol shortcut: model kontrol seed 42 (10, 23; src_port di indeks 0).
DeepSHAP (background primer = window latih yang sama dengan model utama) dan
LIME flat (230 dimensi; kategorikal = kode asli) pada 450 instance yang sama.
Per kelas (label sebenarnya): peringkat src_port pada I_{i,f} (1 = teratas,
seri -> indeks fitur terkecil), proporsi I_src_port / sum I_f, dan rerata
penurunan probabilitas kelas c_i (prediksi model kontrol) ketika src_port
diganti median latih (satuan asli lalu Min-Max) di seluruh time-step.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import shap  # noqa: E402,F401

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

tf.config.set_visible_devices([], "GPU")
_sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, device_count={"GPU": 0}))
tf.compat.v1.keras.backend.set_session(_sess)

sys.path.insert(0, os.path.dirname(__file__))
from common_c import B2, C_ART, C_RES, CLASSES, MODELS, ROWS, build_inputs, load_rows, run_log, sha256  # noqa: E402
from c4_core import Ctx, agg_f  # noqa: E402

OUT = f"{C_RES}/c4_shortcut_control.csv"
SEED = 42


def main():
    if os.path.exists(OUT):
        return run_log("C4 shortcut sudah ada")
    freeze = json.load(open(f"{C_RES}/freeze_after_c2.json"))["hash_sha256"]
    mpath = f"{MODELS}/control_s42.h5"
    assert sha256(mpath) == freeze[mpath]
    FJ = json.load(open(f"{B2}/features.json"))
    rows = load_rows()
    X_tr, y_tr, _ = build_inputs("control", 10, "train", None, rows)
    X_te, _, _ = build_inputs("control", 10, "test", None, rows)
    at = np.load(f"{C_ART}/attributions_450.npz")
    IDX, YP = at["test_index"], at["y_true"]
    XP = X_te[IDX]
    del X_te
    sc = pickle.load(open(f"{C_ART}/scaler_srcport.pkl", "rb"))
    scale = np.r_[sc.scale_, FJ["scale_"]]
    mn = np.r_[sc.min_, FJ["min_"]]
    cat_f = [f + 1 for f in FJ["categorical_index"]]
    codes_len = [len(FJ["codes"][FJ["categorical_columns"][j]]) for j in range(len(cat_f))]
    tf.compat.v1.set_random_seed(SEED)
    model = tf.keras.models.load_model(mpath, compile=False)
    ctx = Ctx(model, X_tr, y_tr, cat_f, scale, mn, codes_len)
    P0 = ctx.predict(XP)
    C = P0.argmax(1)

    bg_idx = np.load(f"{C_ART}/c4_parts/bg_primary_idx.npy")
    ex, _ = ctx.deepshap(bg_idx)
    a_ds, sv, ev = ctx.ds_explain(ex, XP, C)
    e_c, _ = ctx.additivity(sv, ev, P0, C)
    feats = ["src_port"] + FJ["features"]
    cat_names = {f + 1: [k for k, _ in sorted(FJ["codes"][FJ["categorical_columns"][j]].items(), key=lambda kv: kv[1])]
                 for j, f in enumerate(FJ["categorical_index"])}
    ctx.build_lime(SEED, feats, cat_names)
    ctx.rec["active"] = True
    a_lime = np.array([ctx.lime_explain(XP[i], C[i], SEED) for i in range(len(XP))])
    ctx.rec["active"] = False

    med = json.load(open(f"{ROWS}/rows_info.json"))["src_port_median_ruang_model"]
    Xr = XP.copy()
    Xr[:, :, 0] = med
    P1 = ctx.predict(Xr)
    drop = P0[np.arange(len(XP)), C] - P1[np.arange(len(XP)), C]

    rows_out, per_inst = [], []
    for mth, att in (("deepshap", a_ds), ("lime_flat", a_lime)):
        IF = agg_f(att)
        order = np.argsort(-IF, axis=1, kind="stable")
        rank = np.argmax(order == 0, axis=1) + 1
        prop = IF[:, 0] / np.where(IF.sum(1) == 0, 1, IF.sum(1))
        per_inst.append(pd.DataFrame(dict(metode=mth, instance=np.arange(len(XP)), kelas=[CLASSES[y] for y in YP],
                                          c_pred_kontrol=[CLASSES[c] for c in C], rank_src_port=rank,
                                          proporsi_src_port=prop, penurunan_p_ci=drop)))
        for ci, c in enumerate(CLASSES + ["SEMUA"]):
            m = np.ones(len(YP), bool) if c == "SEMUA" else YP == ci
            rows_out.append(dict(kelas=c, metode=mth, n=int(m.sum()), rank_src_port_mean=rank[m].mean(),
                                 rank_src_port_median=np.median(rank[m]), persen_rank1=100 * (rank[m] == 1).mean(),
                                 proporsi_src_port_mean=prop[m].mean(),
                                 penurunan_p_ci_mean=drop[m].mean(), penurunan_p_ci_sd=drop[m].std(ddof=1),
                                 eksploratori=(c == "backdoor")))
    pd.concat(per_inst).to_csv(f"{C_RES}/c4_shortcut_control_per_instance.csv", index=False)
    pd.DataFrame(rows_out).to_csv(OUT, index=False)
    np.savez_compressed(f"{C_ART}/attributions_450_control.npz", att_deepshap=a_ds, att_lime_flat=a_lime, c_pred=C,
                        P0=P0, additivity_err=e_c, features=np.array(feats))
    run_log(f"C4 shortcut selesai; additivity maks {e_c.max():.2e}; LIME kode tidak valid {ctx.rec['invalid']}")


if __name__ == "__main__":
    main()
