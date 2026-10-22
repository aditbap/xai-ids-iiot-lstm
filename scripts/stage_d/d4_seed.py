"""
D4: kepekaan seed model untuk XAI. 90 instance = 10 pertama per kelas dari
artifacts/stage_d/main_450_index_v2.csv. Untuk model utama seed <s>
(main_s42/43/44 Tahap C, hash diverifikasi): DeepSHAP (background primer
Tahap C), LIME flat (seed 42), kustom-flat (background primer, seed 42), dan
M2 lokal (penghapusan fitur, pengganti median/modus B2, k = 0,1,3,5,10,15,
5 urutan acak dengan seed per instance yang SAMA dengan D3 deletion.npz).
Semua memakai c4_core.Ctx tanpa perubahan; c_i = prediksi model seed itu.
Seed 42 dihitung ulang dengan cara yang sama dan dicocokkan dengan D3.

Pemakaian: python d4_seed.py <seed>   |   python d4_seed.py report
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
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

tf.config.set_visible_devices([], "GPU")
_sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, device_count={"GPU": 0}))
tf.compat.v1.keras.backend.set_session(_sess)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_c"))
sys.path.insert(0, os.path.dirname(__file__))
from c4_core import Ctx, agg_f, norm_rows, spear  # noqa: E402
from common_d import B2, C_RES_C, CLASSES, D_ART, D_RES, run_log, sha256  # noqa: E402

SEED, KS_F, REP = 42, [0, 1, 3, 5, 10, 15], 5
METH = ["deepshap", "lime_flat", "kustom_flat"]
OUT = f"{D_ART}/d4"
os.makedirs(OUT, exist_ok=True)


def subset_positions(idx):
    return idx.reset_index().groupby("kelas_window", sort=False).head(10)["index"].to_numpy()


def run_seed(seed):
    out = f"{OUT}/seed{seed}.npz"
    if os.path.exists(out):
        return run_log(f"D4 seed {seed} sudah ada")
    freeze = json.load(open(f"{C_RES_C}/freeze_after_c2.json"))["hash_sha256"]
    mpath = f"artifacts/stage_c/models/main_s{seed}.h5"
    assert sha256(mpath) == freeze[mpath]
    FJ = json.load(open(f"{B2}/features.json"))
    X_tr = np.load(f"{B2}/X_train.npy")
    y_tr = np.load(f"{B2}/y_train.npy").astype(int)
    X_te = np.load(f"{B2}/X_test.npy", mmap_mode="r")
    meta_te = pd.read_csv(f"{B2}/meta_test.csv")
    idx = pd.read_csv(f"{D_ART}/main_450_index_v2.csv")
    pos = subset_positions(idx)
    pm = idx.iloc[pos].merge(meta_te.reset_index(), on=["session_id", "posisi_akhir_window"], how="left")
    assert pm["index"].notna().all() and len(pm) == 90
    tf.compat.v1.set_random_seed(SEED)
    model = tf.keras.models.load_model(mpath, compile=False)
    cat_f = FJ["categorical_index"]
    codes_len = [len(FJ["codes"][FJ["categorical_columns"][j]]) for j in range(len(cat_f))]
    ctx = Ctx(model, X_tr, y_tr, cat_f, FJ["scale_"], FJ["min_"], codes_len)
    XP = np.asarray(X_te[pm["index"].to_numpy()], dtype=np.float32)
    P0 = ctx.predict(XP)
    C = P0.argmax(1)
    NI, NF = len(XP), ctx.NF
    bg_idx = np.load("artifacts/stage_c/c4_parts/bg_primary_idx.npy")
    bg = X_tr[bg_idx]
    ex, _ = ctx.deepshap(bg_idx)
    att = {"deepshap": ctx.ds_explain(ex, XP, C)[0]}
    cat_names = {f: [k for k, _ in sorted(FJ["codes"][FJ["categorical_columns"][j]].items(), key=lambda kv: kv[1])]
                 for j, f in enumerate(cat_f)}
    ctx.build_lime(SEED, FJ["features"], cat_names)
    att["lime_flat"] = np.array([ctx.lime_explain(XP[i], C[i], SEED) for i in range(NI)])
    att["kustom_flat"] = np.array([ctx.custom(XP[i], C[i], "kustom_flat", bg, SEED) for i in range(NI)])
    IF = {m: agg_f(att[m]) for m in METH}

    base = pd.read_csv(f"{B2}/deletion_baseline.csv").nilai_ruang_model.to_numpy()
    rand_seeds = np.load(f"{D_ART}/c4_parts/deletion.npz")["rand_seeds"][pos]

    def run(orders):
        batch = []
        for i in range(NI):
            for k in KS_F:
                x = XP[i].copy()
                sel = orders[i][:k]
                x[:, sel] = base[sel]
                batch.append(x)
        P = ctx.predict(np.stack(batch)).reshape(NI, len(KS_F), -1)
        return P[np.arange(NI)[:, None], np.arange(len(KS_F))[None, :], C[:, None]]

    R = {f"p_{m}": run(np.argsort(-IF[m], axis=1, kind="stable")) for m in METH}
    for r in range(REP):
        R[f"p_acak{r}"] = run(np.stack([np.random.default_rng(rand_seeds[i, r]).permutation(NF) for i in range(NI)]))
    np.savez_compressed(out, **{f"att_{m}": att[m] for m in METH}, **{f"IF_{m}": IF[m] for m in METH}, **R,
                        P0=P0, c_pred=C, pos_in_450=pos, test_index=pm["index"].to_numpy(),
                        y_true=np.array([CLASSES.index(c) for c in pm.kelas_window_x]))
    run_log(f"D4 seed {seed} selesai")


def report():
    auc = lambda c: np.trapz(c, x=KS_F, axis=-1)  # noqa: E731
    Z = {s: np.load(f"{OUT}/seed{s}.npz") for s in (42, 43, 44)}
    # cek: seed 42 == D3 (subset)
    at = np.load(f"{D_ART}/attributions_450_v2.npz")
    pos = Z[42]["pos_in_450"]
    chk = {m: float(np.abs(Z[42][f"IF_{m}"] - at[f"IF_{m}"][pos]).max()) for m in METH}
    rows = []
    G = {s: {m: norm_rows(Z[s][f"IF_{m}"]).mean(0) for m in METH} for s in Z}
    for s, z in Z.items():
        Ar = np.mean([auc(z[f"p_acak{r}"]) for r in range(REP)], axis=0)
        d = {m: Ar - auc(z[f"p_{m}"]) for m in METH}
        for m in METH:
            rows.append(dict(seed=s, metrik="m2_lokal_dauc_mean", metode=m, nilai=d[m].mean(), sd=d[m].std(ddof=1), n=len(Ar)))
        p1 = d["deepshap"] - d["lime_flat"]
        rows.append(dict(seed=s, metrik="P1_selisih_deepshap_minus_lime_flat", metode="deepshap-lime_flat",
                         nilai=p1.mean(), sd=p1.std(ddof=1), n=len(p1)))
        for m in METH:
            rows.append(dict(seed=s, metrik="spearman_global_If_vs_seed42", metode=m,
                             nilai=spear(G[42][m], G[s][m]), sd=np.nan, n=len(G[s][m])))
        rows.append(dict(seed=s, metrik="persen_c_pred_sama_dgn_seed42", metode="model",
                         nilai=100 * (z["c_pred"] == Z[42]["c_pred"]).mean(), sd=np.nan, n=len(Ar)))
    df = pd.DataFrame(rows)
    df["catatan"] = ""
    df.loc[df.seed == 42, "catatan"] = f"seed 42 dihitung ulang; selisih maks I_f vs D3: {chk}"
    df.to_csv(f"{D_RES}/d4_seed_sensitivity.csv", index=False)
    run_log(f"D4 laporan selesai; cek seed 42 vs D3 {chk}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        run_seed(int(sys.argv[1]))
