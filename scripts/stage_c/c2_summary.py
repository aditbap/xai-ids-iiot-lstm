"""
C2 ringkasan: c2_training.csv, c2_t6.csv, c2_search_<model>.csv,
c2_ablation_val.csv, freeze_after_c2.json. Hanya data validasi.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import glob  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common_c import C_RES, MODELS, N_CLS, RUNS, build_inputs, load_rows, run_log, save_json, sha256  # noqa: E402


def rec(name):
    return json.load(open(f"{RUNS}/{name}.json"))


def main():
    runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{RUNS}/*.json"))}

    # c2_training.csv (tanpa run pencarian)
    rows = []
    for n, r in runs.items():
        if n.startswith("search_"):
            continue
        sp = r["spec"]
        fam = n.rsplit("_s", 1)[0]
        rows.append(dict(model=fam, seed=sp["seed"], konfigurasi=json.dumps(sp.get("params", {})),
                         w=sp.get("w", 10), epoch_terpilih=r.get("epoch_terpilih"), f1_macro_val=r["val_f1_macro"],
                         waktu_latih_detik=r["waktu_latih_detik"], berkas_model=r.get("model")))
    pd.DataFrame(rows).sort_values(["model", "seed"]).to_csv(f"{C_RES}/c2_training.csv", index=False)

    # T6
    seq = {42: rec("main_s42"), 43: rec("main_s43"), 44: rec("main_s44"), 45: rec("t6_urut_s45"), 46: rec("t6_urut_s46")}
    t6 = []
    for s in range(42, 47):
        a, b = seq[s]["val_f1_macro"], rec(f"t6_acak_s{s}")["val_f1_macro"]
        t6.append(dict(seed=s, f1_val_urut=a, f1_val_acak=b, delta_f1=a - b))
    t6 = pd.DataFrame(t6)
    d = t6.delta_f1.to_numpy()
    obs = d.mean()
    perms = [np.mean(np.array(sg) * np.abs(d)) for sg in itertools.product([1, -1], repeat=5)]
    p = float(np.mean(np.array(perms) >= obs - 1e-12))
    lolos = bool(p <= 0.05 and obs >= 0.01)
    t6.loc[len(t6)] = dict(seed="RINGKASAN", f1_val_urut=t6.f1_val_urut.mean(), f1_val_acak=t6.f1_val_acak.mean(),
                           delta_f1=obs)
    t6["p_permutasi_satu_sisi"] = [np.nan] * 5 + [p]
    t6["lolos"] = [np.nan] * 5 + [lolos]
    t6.to_csv(f"{C_RES}/c2_t6.csv", index=False)

    # pencarian
    for model in ("rf", "xgb", "mlp", "lstm1"):
        cfgs = json.load(open(f"{C_RES}/c2_search_{model}_configs.json"))
        sr = []
        for i, c in enumerate(cfgs):
            r = rec(f"search_{model}_{i:02d}")
            sr.append(dict(indeks=i, **c, f1_macro_val=r["val_f1_macro"], epoch_terpilih=r.get("epoch_terpilih"),
                           waktu_latih_detik=r["waktu_latih_detik"]))
        sr = pd.DataFrame(sr)
        sr["terpilih"] = sr.indeks == int(sr.f1_macro_val.idxmax())
        sr.to_csv(f"{C_RES}/c2_search_{model}.csv", index=False)

    # ablasi (w=10 = model utama; subset dihitung di sini)
    rows_ = load_rows()
    meta = rows_[2]
    blen = meta.groupby("session_id").pos.transform("max").to_numpy() + 1
    pos = meta.pos.to_numpy()
    X_va, y_va, end = build_inputs("main", 10, "val", None, rows_)
    m = (blen[end] >= 20) & (pos[end] >= 19)
    ab = []
    for s in (42, 43, 44):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(s)
        model = tf.keras.models.load_model(f"{MODELS}/main_s{s}.h5", compile=False)
        pred = model.predict(X_va[m], batch_size=2048, verbose=0).argmax(1)
        ab.append(dict(w=10, seed=s, f1_macro_val=rec(f"main_s{s}")["val_f1_macro"], n_val=len(y_va),
                       f1_macro_val_subset20=f1_score(y_va[m], pred, average="macro", labels=np.arange(N_CLS),
                                                      zero_division=0), n_val_subset20=int(m.sum())))
        tf.compat.v1.keras.backend.clear_session()
    for w in (1, 5, 20):
        for s in (42, 43, 44):
            r = rec(f"abl_w{w}_s{s}")
            ab.append(dict(w=w, seed=s, f1_macro_val=r["val_f1_macro"], n_val=r["n_val"],
                           f1_macro_val_subset20=r["val_f1_macro_subset20"], n_val_subset20=r["n_val_subset20"]))
    ab = pd.DataFrame(ab).sort_values(["w", "seed"])
    agg = ab.groupby("w").agg(f1_macro_val_mean=("f1_macro_val", "mean"), f1_macro_val_sd=("f1_macro_val", "std"),
                              f1_subset20_mean=("f1_macro_val_subset20", "mean"),
                              f1_subset20_sd=("f1_macro_val_subset20", "std")).reset_index()
    agg["seed"] = "rerata_sd"
    pd.concat([ab, agg], ignore_index=True).to_csv(f"{C_RES}/c2_ablation_val.csv", index=False)

    # freeze
    files = sorted(glob.glob(f"{MODELS}/*"))
    save_json(dict(catatan="Mulai titik ini tidak ada pelatihan lagi.",
                   hash_sha256={f: sha256(f) for f in files}), f"{C_RES}/freeze_after_c2.json")
    run_log(f"C2 ringkasan selesai; T6 rerata dF1={obs:.4f}, p={p:.4f}, lolos={lolos}; {len(files)} model dibekukan")


if __name__ == "__main__":
    main()
