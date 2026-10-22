"""
D1 & D2: tabel dari prediksi tersimpan (artifacts/stage_d/preds/) dan nilai
asli (artifacts/stage_d/rows_raw.npy). Tanpa TensorFlow.

Pemeriksaan awal (d_predict_check.csv): F1-macro uji penuh dari prediksi
tersimpan dibandingkan dengan results/stage_c/c3_*.csv (harus sama).

D2 hash: blake2b 8 byte atas byte float64 NILAI ASLI (window 10x22 atau
time-step terakhir 22), tanpa pembulatan; kolisi diperiksa terhadap identitas
eksak (np.unique pada nilai) dan dilaporkan di run_log.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(__file__))
from common_d import C_RES_C, CLASSES, D_ART, D_RES, D_RUNS, N_CLS, load_meta, run_log, window_ends  # noqa: E402

PRED = f"{D_ART}/preds"
LAB = np.arange(N_CLS)
FAMS = ["main", "lstm1", "mlp", "rf", "xgb", "control"]


def load_pred(p, name):
    z = np.load(f"{PRED}/{p}_{name}.npz")
    return z["pred"].astype(int), z["end"], z["y"].astype(int)


def scores(y, pred):
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    d = dict(f1_macro=f1_score(y, pred, labels=LAB, average="macro", zero_division=0),
             f1_weighted=f1_score(y, pred, labels=LAB, average="weighted", zero_division=0))
    for i, c in enumerate(CLASSES):
        d[f"recall_{c}"], d[f"f1_{c}"], d[f"n_{c}"] = rc[i], f1[i], int(sup[i])
    d["n"] = len(y)
    return d


def hashes(M):
    M = np.ascontiguousarray(np.asarray(M, dtype=np.float64).reshape(len(M), -1) + 0.0)
    h = np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in M],
                 dtype=np.uint64)
    n_exact = len(np.unique(M.view(np.dtype((np.void, M.shape[1] * 8))).ravel()))
    return h, n_exact


# ---------------------------------------------------------------- pemeriksaan
def check_preds():
    c3 = pd.read_csv(f"{C_RES_C}/c3_test_performance.csv")
    ct = pd.read_csv(f"{C_RES_C}/c3_control_test.csv")
    ab = pd.read_csv(f"{C_RES_C}/c3_ablation_test.csv")
    ref = {(r.model, int(r.seed)): r.f1_macro for r in c3.itertuples()}
    ref.update({("control", int(r.seed)): r.f1_macro for r in ct.itertuples() if r.seed != "rerata_sd"})
    ref.update({(f"abl_w{int(r.w)}", int(r.seed)): r.f1_macro for r in ab.itertuples()
                if r.seed != "rerata_sd" and int(r.w) != 10})
    rows = []
    for (fam, s), v in ref.items():
        pred, _, y = load_pred("test", f"{fam}_s{s}")
        f = f1_score(y, pred, labels=LAB, average="macro", zero_division=0)
        rows.append(dict(model=fam, seed=s, f1_macro_c3=v, f1_macro_prediksi_d=f, selisih=f - v))
    df = pd.DataFrame(rows)
    df.to_csv(f"{D_RES}/d_predict_check.csv", index=False)
    run_log(f"D cek prediksi vs C3: selisih maks {df.selisih.abs().max():.2e} ({len(df)} model)")


# ---------------------------------------------------------------- D1
def d1():
    tr_rec = {("lama", 42): json.load(open("results/stage_b/train_log.json"))["val_f1_macro_terpilih"],
              ("log1p", 42): json.load(open("results/diag_log/train_log.json")).get("val_f1_macro_terpilih")}
    for s in (43, 44):
        tr_rec[("lama", s)] = json.load(open(f"{D_RUNS}/old_s{s}.json"))["val_f1_macro"]
        tr_rec[("log1p", s)] = json.load(open(f"{C_RES_C}/c2_runs/main_s{s}.json"))["val_f1_macro"]
    rows = []
    for cfg, fam in (("lama", "old"), ("log1p", "main")):
        for s in (42, 43, 44):
            pred, _, y = load_pred("val", f"{fam}_s{s}")
            d = scores(y, pred)
            rows.append(dict(baris="per_seed", konfigurasi=cfg, seed=s, f1_macro_val_saat_latih=tr_rec[(cfg, s)],
                             f1_macro_val=d["f1_macro"],
                             **{f"recall_{c}": d[f"recall_{c}"] for c in CLASSES},
                             **{f"f1_{c}": d[f"f1_{c}"] for c in CLASSES}, n_val=d["n"]))
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("baris", "konfigurasi", "seed")]
    out = [df]
    for cfg in ("lama", "log1p"):
        g = df[df.konfigurasi == cfg][num]
        out.append(pd.DataFrame([dict(baris="rerata", konfigurasi=cfg, seed="42-44", **g.mean().to_dict()),
                                 dict(baris="sd", konfigurasi=cfg, seed="42-44", **g.std(ddof=1).to_dict())]))
    a = df[df.konfigurasi == "log1p"].set_index("seed")[num]
    b = df[df.konfigurasi == "lama"].set_index("seed")[num]
    dlt = (a - b).reset_index()
    dlt.insert(0, "baris", "selisih_log1p_minus_lama")
    dlt.insert(1, "konfigurasi", "log1p-lama")
    out.append(dlt)
    dd = dlt[num]
    out.append(pd.DataFrame([dict(baris="selisih_rerata", konfigurasi="log1p-lama", seed="42-44", **dd.mean().to_dict()),
                             dict(baris="selisih_sd", konfigurasi="log1p-lama", seed="42-44", **dd.std(ddof=1).to_dict())]))
    pd.concat(out, ignore_index=True).drop(columns=["n_val"]).assign(
        n_val=df.n_val.iloc[0]).to_csv(f"{D_RES}/d1_log1p_multiseed.csv", index=False)
    run_log("D1 tabel selesai")


# ---------------------------------------------------------------- D2
def d2():
    raw = np.load(f"{D_ART}/rows_raw.npy")
    meta = load_meta()
    lab = meta.label.to_numpy().astype(int)
    E = {p: window_ends(meta, p) for p in ("train", "val", "test")}
    H = {}
    for p, e in E.items():
        Hw, nw = hashes(raw[e[:, None] + np.arange(-9, 1)])
        Hl, nl = hashes(raw[e])
        H[("window", p)], H[("timestep_akhir", p)] = Hw, Hl
        run_log(f"D2 hash {p}: window unik hash={len(np.unique(Hw))} eksak={nw}; "
                f"time-step unik hash={len(np.unique(Hl))} eksak={nl}")
    Y = {p: lab[e] for p, e in E.items()}

    # 1. di dalam partisi
    rows = []
    for lvl in ("window", "timestep_akhir"):
        for p in ("train", "val", "test"):
            h, y = H[(lvl, p)], Y[p]
            df = pd.DataFrame(dict(h=h, y=y))
            cnt = df.groupby("h").h.transform("size").to_numpy()
            nlab = df.groupby("h").y.transform("nunique").to_numpy()
            for ci, c in enumerate(CLASSES + ["SEMUA"]):
                m = np.ones(len(y), bool) if c == "SEMUA" else y == ci
                rows.append(dict(tingkat=lvl, partisi=p, bagian="ringkasan", kelas=c, kelas_lain="",
                                 n_window=int(m.sum()), persen_hash_muncul_lebih_sekali=100 * (cnt[m] > 1).mean() if m.any() else np.nan,
                                 persen_hash_lebih_satu_label=100 * (nlab[m] > 1).mean() if m.any() else np.nan,
                                 n_hash=np.nan))
            # pasangan label dominan (hash dengan >1 label)
            ml = df[nlab > 1]
            sets = ml.groupby("h").y.agg(lambda s: tuple(sorted(set(s))))
            sizes = ml.groupby("h").size()
            pc = {}
            for hh, labs in sets.items():
                for i in range(len(labs)):
                    for j in range(i + 1, len(labs)):
                        k = (labs[i], labs[j])
                        a = pc.setdefault(k, [0, 0])
                        a[0] += 1
                        a[1] += int(sizes[hh])
            top = sorted(pc.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0]))[:10]
            for (a, b), (nh, nw_) in top:
                rows.append(dict(tingkat=lvl, partisi=p, bagian="pasangan_label_dominan", kelas=CLASSES[a],
                                 kelas_lain=CLASSES[b], n_window=nw_, persen_hash_muncul_lebih_sekali=np.nan,
                                 persen_hash_lebih_satu_label=np.nan, n_hash=nh))
    pd.DataFrame(rows).to_csv(f"{D_RES}/d2_within_split.csv", index=False)
    run_log("D2 within selesai")

    # 2. lintas partisi
    def ref_table(parts, lvl):
        h = np.concatenate([H[(lvl, p)] for p in parts])
        y = np.concatenate([Y[p] for p in parts])
        return pd.DataFrame(dict(h=h, y=y)).drop_duplicates()

    rows = []
    pairs = [(("train",), "val"), (("train",), "test"), (("val",), "test"), (("train", "val"), "test")]
    for lvl in ("window", "timestep_akhir"):
        for A_, B_ in pairs:
            ref = ref_table(A_, lvl)
            labsets = ref.groupby("h").y.agg(set)
            hb, yb = H[(lvl, B_)], Y[B_]
            inA = np.isin(hb, labsets.index.to_numpy())
            ls = labsets.reindex(hb).to_numpy()
            same_only = np.array([bool(i) and s == {yy} for i, s, yy in zip(inA, ls, yb)])
            conflict = inA & ~same_only
            name = "+".join(A_) + "->" + B_
            for ci, c in enumerate(CLASSES + ["SEMUA"]):
                m = np.ones(len(yb), bool) if c == "SEMUA" else yb == ci
                rows.append(dict(tingkat=lvl, pasangan=name, bagian="ringkasan", kelas=c, kelas_lain="",
                                 n_window=int(m.sum()), persen_ada_di_partisi_pertama=100 * inA[m].mean(),
                                 persen_hanya_label_sama=100 * same_only[m].mean(),
                                 persen_konflik_label_berbeda=100 * conflict[m].mean(), n_window_konflik=np.nan))
            pc = {}
            for i in np.flatnonzero(conflict):
                for o in ls[i]:
                    if o != yb[i]:
                        pc[(yb[i], o)] = pc.get((yb[i], o), 0) + 1
            for (a, b), n in sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
                rows.append(dict(tingkat=lvl, pasangan=name, bagian="pasangan_konflik_dominan", kelas=CLASSES[a],
                                 kelas_lain=CLASSES[b], n_window=np.nan, persen_ada_di_partisi_pertama=np.nan,
                                 persen_hanya_label_sama=np.nan, persen_konflik_label_berbeda=np.nan,
                                 n_window_konflik=n))
    pd.DataFrame(rows).to_csv(f"{D_RES}/d2_cross_split.csv", index=False)
    run_log("D2 cross selesai")

    # 3. novel-input uji (window penuh tidak ada di latih maupun validasi)
    seen = np.concatenate([H[("window", "train")], H[("window", "val")]])
    novel_te = ~np.isin(H[("window", "test")], seen)
    novel_va = ~np.isin(H[("window", "val")], H[("window", "train")])
    np.savez_compressed(f"{D_ART}/d2_novel_masks.npz", novel_test=novel_te, novel_val=novel_va,
                        end_test=E["test"], end_val=E["val"])

    def novel_table(p, mask, models, path):
        rows = []
        for fam, s in models:
            pred, end, y = load_pred(p, f"{fam}_s{s}")
            assert np.array_equal(end, E[p])
            for sub, m in (("novel", mask), ("pelengkap_punya_padanan", ~mask), ("semua", np.ones(len(y), bool))):
                rows.append(dict(model=fam, seed=s, subset=sub, **scores(y[m], pred[m])))
        df = pd.DataFrame(rows)
        num = [c for c in df.columns if c not in ("model", "seed", "subset")]
        agg = df.groupby(["model", "subset"], sort=False)[num].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index().assign(seed="rerata_sd")
        pd.concat([df, agg], ignore_index=True).to_csv(path, index=False)

    novel_table("test", novel_te, [(f, s) for f in FAMS for s in (42, 43, 44)], f"{D_RES}/d2_novel_input.csv")
    novel_table("val", novel_va, [(f, s) for f in ("old", "main") for s in (42, 43, 44)], f"{D_RES}/d2_novel_val.csv")
    run_log(f"D2 novel selesai: uji novel {novel_te.sum()}/{len(novel_te)}, val novel {novel_va.sum()}/{len(novel_va)}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("check", "all"):
        check_preds()
    if step in ("d1", "all"):
        d1()
    if step in ("d2", "all"):
        d2()
