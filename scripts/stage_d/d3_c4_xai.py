"""
D3.4: SALINAN scripts/stage_c/c4_xai.py untuk 450 instance BARU
(artifacts/stage_d/main_450_index_v2.csv, dari partisi uji final tanpa sesi
pilot). Prosedur, konstanta, dan c4_core TIDAK diubah. Perubahan terhadap
c4_xai.py (hanya ini):
  1. Keluaran: artifacts/stage_d/c4_parts/, artifacts/stage_d/attributions_450_v2.npz,
     results/stage_d/d3_c4_*.csv; run_log ke results/stage_d/run_log.txt.
     Model & freeze dibaca dari Tahap C (beku, hash diverifikasi).
  2. setup(): indeks 450 v2; X_te/y_te/meta_te untuk M2 tingkat SISTEM dibatasi
     ke partisi uji FINAL (sesi pilot dikeluarkan). IDX tetap menunjuk ke
     artifacts/stage_b2/X_test.npy (uji penuh), seperti C4.
  3. primer: background primer diverifikasi identik dengan
     artifacts/stage_c/c4_parts/bg_primary_idx.npy.
  4. m4: tepat sebelum M4, `uptime` dan 10 proses teratas menurut CPU dicatat
     ke results/stage_d/d3_m4_kondisi.txt. Instance M4 = 5 pertama per kelas
     (45), sama dengan C4.

Bagian (satu per proses): primer, m4, m3_proc, m3_pert_<s>, sens_bg, deletion.
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
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

tf.config.set_visible_devices([], "GPU")
_sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, device_count={"GPU": 0}))
tf.compat.v1.keras.backend.set_session(_sess)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_c"))
sys.path.insert(0, os.path.dirname(__file__))
from common_c import A, B2, C_ART, C_RES, CLASSES, MODELS, run_log, save_json, sha256  # noqa: E402
# ---- Tahap D: keluaran ke artifacts/stage_d & results/stage_d; model & freeze Tahap C (beku)
C_ART, C_RES, C_RES_C = "artifacts/stage_d", "results/stage_d", "results/stage_c"
from common_d import run_log  # noqa: E402,F811
from c4_core import Ctx, agg_f, agg_t, level_vec  # noqa: E402

PARTS = f"{C_ART}/c4_parts"
os.makedirs(PARTS, exist_ok=True)
SEED = 42
SEEDS_PROC = [101, 102, 103, 104, 105]
KS_F, KS_T, REP = [0, 1, 3, 5, 10, 15], [0, 1, 2, 3, 5], 5
METHODS_F = ["deepshap", "lime_flat", "kustom_flat", "per_fitur"]
METHODS_T = ["deepshap", "lime_flat", "kustom_flat", "per_timestep"]
ALL5 = ["deepshap", "lime_flat", "kustom_flat", "per_fitur", "per_timestep"]


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def setup():
    freeze = json.load(open(f"{C_RES_C}/freeze_after_c2.json"))["hash_sha256"]
    mpath = f"{MODELS}/main_s42.h5"
    assert sha256(mpath) == freeze[mpath] == sha256(f"{B2}/lstm_main_seed42.h5")
    FJ = json.load(open(f"{B2}/features.json"))
    X_tr = np.load(f"{B2}/X_train.npy")
    y_tr = np.load(f"{B2}/y_train.npy").astype(int)
    X_te = np.load(f"{B2}/X_test.npy", mmap_mode="r")
    meta_te = pd.read_csv(f"{B2}/meta_test.csv")
    idx = pd.read_csv(f"{C_ART}/main_450_index_v2.csv")
    pilot = set(pd.read_csv(f"{A}/pilot_40_index.csv").session_id)
    final = ~meta_te.session_id.isin(pilot).to_numpy()              # partisi uji final
    pm = idx.merge(meta_te.reset_index(), on=["session_id", "posisi_akhir_window"], how="left")
    assert pm["index"].notna().all() and (pm.kelas_window_x == pm.kelas_window_y).all() and len(pm) == 450
    if os.environ.get("STAGE_C_SMOKE_PER_CLASS"):               # uji asap saja
        k = int(os.environ["STAGE_C_SMOKE_PER_CLASS"])
        pm = pm.groupby("kelas_window_x", sort=False).head(k).reset_index(drop=True)
    tf.compat.v1.set_random_seed(SEED)
    model = tf.keras.models.load_model(mpath, compile=False)
    cat_f = FJ["categorical_index"]
    codes_len = [len(FJ["codes"][FJ["categorical_columns"][j]]) for j in range(len(cat_f))]
    ctx = Ctx(model, X_tr, y_tr, cat_f, FJ["scale_"], FJ["min_"], codes_len)
    IDX = pm["index"].to_numpy()
    XP = np.asarray(X_te[IDX], dtype=np.float32)
    YP = np.array([CLASSES.index(c) for c in pm.kelas_window_x])
    P0 = ctx.predict(XP)
    C = P0.argmax(1)
    cat_names = {f: [k for k, _ in sorted(FJ["codes"][FJ["categorical_columns"][j]].items(), key=lambda kv: kv[1])]
                 for j, f in enumerate(cat_f)}
    return dict(FJ=FJ, ctx=ctx, IDX=IDX, XP=XP, YP=YP, P0=P0, C=C, SESS=pm.session_id.to_numpy(),
                cat_names=cat_names, X_te=np.asarray(X_te[np.flatnonzero(final)]),
                y_te=np.load(f"{B2}/y_test.npy").astype(int)[final], meta_te=meta_te[final].reset_index(drop=True))


def part_primer(S):
    out = f"{C_ART}/attributions_450_v2.npz"
    if os.path.exists(out):
        return run_log("C4 primer sudah ada")
    ctx, XP, C, P0, NI = S["ctx"], S["XP"], S["C"], S["P0"], len(S["XP"])
    bg_idx = ctx.empirical_bg(SEED)
    assert np.array_equal(bg_idx, np.load("artifacts/stage_c/c4_parts/bg_primary_idx.npy"))
    np.save(f"{PARTS}/bg_primary_idx.npy", bg_idx)
    ex, t_init = ctx.deepshap(bg_idx)
    att = {}
    att["deepshap"], sv, ev = ctx.ds_explain(ex, XP, C)
    e_c, e_all = ctx.additivity(sv, ev, P0, C)
    pd.DataFrame(dict(instance=np.arange(NI), session_id=S["SESS"], kelas=[CLASSES[y] for y in S["YP"]],
                      c_pred=[CLASSES[c] for c in C], galat_kelas_ci=e_c, galat_maks_semua_kelas=e_all,
                      lolos_1e2=e_c < 1e-2)).to_csv(f"{C_RES}/d3_c4_additivity.csv", index=False)
    ctx.build_lime(SEED, S["FJ"]["features"], S["cat_names"])
    pw = ctx.wrapper(ctx.to_mixed(XP))
    ctx.rec["active"] = True
    att["lime_flat"] = np.array([ctx.lime_explain(XP[i], C[i], SEED) for i in range(NI)])
    ctx.rec["active"] = False
    bg = ctx.X_tr[bg_idx]
    for v in ctx.groups:
        att[v] = np.array([ctx.custom(XP[i], C[i], v, bg, SEED) for i in range(NI)])
    IF = {m: agg_f(att[m]) for m in ("deepshap", "lime_flat", "kustom_flat")}
    IT = {m: agg_t(att[m]) for m in ("deepshap", "lime_flat", "kustom_flat")}
    IF["per_fitur"], IT["per_timestep"] = att["per_fitur"], att["per_timestep"]
    checks = dict(additivity_lolos=int((e_c < 1e-2).sum()), additivity_galat_maks=float(e_c.max()),
                  lime_kode_tidak_valid=ctx.rec["invalid"], lime_total_nilai_kategorikal=ctx.rec["total"],
                  lime_pembungkus_selisih_maks=float(np.abs(pw - P0).max()),
                  bentuk={k: list(v.shape) for k, v in att.items()},
                  nan=int(sum((~np.isfinite(v)).sum() for v in att.values())), deepshap_init_detik=t_init)
    save_json(checks, f"{PARTS}/primer_checks.json")
    np.savez_compressed(out, **{f"att_{k}": v for k, v in att.items()}, **{f"IF_{k}": v for k, v in IF.items()},
                        **{f"IT_{k}": v for k, v in IT.items()}, c_pred=C, y_true=S["YP"], P0=P0,
                        test_index=S["IDX"], session_id=S["SESS"].astype(str), expected_value=ev,
                        bg_primary_idx=bg_idx, features=np.array(S["FJ"]["features"]))
    run_log(f"C4 primer selesai: {checks}")


def load_att():
    return np.load(f"{C_ART}/attributions_450_v2.npz")


def part_m4(S):
    out = f"{C_RES}/d3_c4_m4.csv"
    if os.path.exists(out):
        return run_log("C4 m4 sudah ada")
    ctx, XP, C, YP = S["ctx"], S["XP"], S["C"], S["YP"]
    sel = np.concatenate([np.flatnonzero(YP == c)[:5] for c in range(9)])       # 45 instance
    bg_idx = np.load(f"{PARTS}/bg_primary_idx.npy")
    bg = ctx.X_tr[bg_idx]
    import subprocess
    with open(f"{C_RES}/d3_m4_kondisi.txt", "w") as fh:
        fh.write(f"waktu: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n$ uptime\n")
        fh.write(subprocess.run(["uptime"], capture_output=True, text=True).stdout)
        fh.write("\n$ ps -Ao pid,pcpu,pmem,rss,etime,comm -r | head -11\n")
        fh.write(subprocess.run("ps -Ao pid,pcpu,pmem,rss,etime,comm -r | head -11", shell=True,
                                capture_output=True, text=True).stdout)
    ex, t_ds = ctx.deepshap(bg_idx)
    _, t_lime = ctx.build_lime(SEED, S["FJ"]["features"], S["cat_names"])
    init = dict(deepshap=t_ds, lime_flat=t_lime, kustom_flat=0.0, per_fitur=0.0, per_timestep=0.0)
    runners = {
        "deepshap": lambda i: ex.shap_values(XP[i:i + 1], check_additivity=False),
        "lime_flat": lambda i: ctx.lime_explain(XP[i], C[i], SEED),
        "kustom_flat": lambda i: ctx.custom(XP[i], C[i], "kustom_flat", bg, SEED),
        "per_fitur": lambda i: ctx.custom(XP[i], C[i], "per_fitur", bg, SEED),
        "per_timestep": lambda i: ctx.custom(XP[i], C[i], "per_timestep", bg, SEED),
    }
    rows, raw = [], []
    for m, fn in runners.items():
        for i in sel[:5]:                        # pemanasan, dibuang
            fn(i)
        tt = np.zeros((len(sel), 10))
        for a, i in enumerate(sel):
            for r in range(10):
                t = time.perf_counter()
                fn(i)
                tt[a, r] = time.perf_counter() - t
        raw.append(pd.DataFrame(tt, columns=[f"ulang_{r}" for r in range(10)]).assign(metode=m, instance=sel))
        per = tt.mean(1)
        q1, q3 = np.percentile(per, [25, 75])
        rows.append(dict(metode=m, init_detik=init[m], per_instance_mean=per.mean(), sd=per.std(ddof=1),
                         median=np.median(per), iqr=q3 - q1, n_instance=len(sel), n_ukur=tt.size,
                         puncak_memori_mb=round(peak_mb(), 1)))
    pd.concat(raw).to_csv(f"{C_RES}/d3_c4_m4_mentah.csv", index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    run_log("C4 m4 selesai")


def part_m3_proc(S):
    out = f"{PARTS}/m3_proc.npz"
    if os.path.exists(out):
        return run_log("C4 m3_proc sudah ada")
    ctx, XP, C, NI = S["ctx"], S["XP"], S["C"], len(S["XP"])
    bg = ctx.X_tr[np.load(f"{PARTS}/bg_primary_idx.npy")]
    ctx.build_lime(SEED, S["FJ"]["features"], S["cat_names"])
    res = {m: [] for m in ALL5}
    bgs = []
    for s in SEEDS_PROC:
        bi = ctx.empirical_bg(s)
        bgs.append(bi)
        ex, _ = ctx.deepshap(bi)
        res["deepshap"].append(level_vec("deepshap", ctx.ds_explain(ex, XP, C)[0]))
        res["lime_flat"].append(level_vec("lime_flat", np.array([ctx.lime_explain(XP[i], C[i], s) for i in range(NI)])))
        for v in ctx.groups:
            res[v].append(level_vec(v, np.array([ctx.custom(XP[i], C[i], v, bg, s) for i in range(NI)])))
        run_log(f"C4 m3_proc seed {s} selesai")
    np.savez_compressed(out, **{m: np.stack(v) for m, v in res.items()}, bg_idx=np.stack(bgs))
    run_log("C4 m3_proc selesai")


def part_m3_pert(S, sigma):
    out = f"{PARTS}/m3_pert_{sigma}.npz"
    if os.path.exists(out):
        return run_log(f"C4 m3_pert {sigma} sudah ada")
    ctx, XP, C, NI = S["ctx"], S["XP"], S["C"], len(S["XP"])
    bg_idx = np.load(f"{PARTS}/bg_primary_idx.npy")
    bg = ctx.X_tr[bg_idx]
    ex, _ = ctx.deepshap(bg_idx)
    ctx.build_lime(SEED, S["FJ"]["features"], S["cat_names"])
    rng = np.random.default_rng(SEED)
    num = np.ones(ctx.NF, bool)
    num[ctx.cat_f] = False
    res = {m: [] for m in ALL5}
    for r in range(REP):
        Xn = XP.copy()
        Xn[:, :, num] = np.clip(Xn[:, :, num] + rng.normal(0, sigma, Xn[:, :, num].shape), 0, 1)
        Xn = Xn.astype(np.float32)
        res["deepshap"].append(level_vec("deepshap", ctx.ds_explain(ex, Xn, C)[0]))
        res["lime_flat"].append(level_vec("lime_flat", np.array([ctx.lime_explain(Xn[i], C[i], SEED) for i in range(NI)])))
        for v in ctx.groups:
            res[v].append(level_vec(v, np.array([ctx.custom(Xn[i], C[i], v, bg, SEED) for i in range(NI)])))
        run_log(f"C4 m3_pert {sigma} salinan {r} selesai")
    np.savez_compressed(out, **{m: np.stack(v) for m, v in res.items()})
    run_log(f"C4 m3_pert {sigma} selesai")


def part_sens_bg(S):
    out = f"{PARTS}/sens_bg.npz"
    if os.path.exists(out):
        return run_log("C4 sens_bg sudah ada")
    ctx, XP, C, P0, NI = S["ctx"], S["XP"], S["C"], S["P0"], len(S["XP"])
    res = {}
    for name, bi in (("seimbang100", ctx.balanced_bg(SEED)), ("empiris50", ctx.empirical_bg(SEED, 50)),
                     ("empiris200", ctx.empirical_bg(SEED, 200))):
        ex, _ = ctx.deepshap(bi)
        a, sv, ev = ctx.ds_explain(ex, XP, C)
        e_c, _ = ctx.additivity(sv, ev, P0, C)
        bg = ctx.X_tr[bi]
        res[f"{name}_bg_idx"] = bi
        res[f"{name}_IF_deepshap"] = agg_f(a)
        res[f"{name}_additivity_maks"] = e_c.max()
        res[f"{name}_IF_kustom_flat"] = agg_f(np.array([ctx.custom(XP[i], C[i], "kustom_flat", bg, SEED) for i in range(NI)]))
        res[f"{name}_IF_per_fitur"] = np.array([ctx.custom(XP[i], C[i], "per_fitur", bg, SEED) for i in range(NI)])
        run_log(f"C4 sens_bg {name} selesai")
    np.savez_compressed(out, **res)
    run_log("C4 sens_bg selesai")


def part_deletion(S):
    out = f"{PARTS}/deletion.npz"
    if os.path.exists(out):
        return run_log("C4 deletion sudah ada")
    ctx, XP, C, NI, NF = S["ctx"], S["XP"], S["C"], len(S["XP"]), S["ctx"].NF
    at = load_att()
    IF = {m: at[f"IF_{m}"] for m in METHODS_F}
    IT = {m: at[f"IT_{m}"] for m in METHODS_T}
    sens = np.load(f"{PARTS}/sens_bg.npz")
    base = pd.read_csv(f"{B2}/deletion_baseline.csv").nilai_ruang_model.to_numpy()
    bg = ctx.X_tr[np.load(f"{PARTS}/bg_primary_idx.npy")]
    rand_seeds = np.random.SeedSequence(SEED).generate_state(NI * REP).reshape(NI, REP)
    sys_seeds = np.random.SeedSequence([SEED, 1]).generate_state(REP)
    emp_pick = np.random.default_rng(SEED).integers(0, len(bg), size=10)
    # fitur >= 99% nilai sama (data latih, window x time-step)
    flat = ctx.X_tr.reshape(-1, NF)
    const99 = [f for f in range(NF) if np.unique(flat[:, f], return_counts=True)[1].max() / len(flat) >= 0.99]
    keep_f = np.array([f for f in range(NF) if f not in const99])
    del flat
    R = {}

    def run(X, orders, ks, repl):
        """orders: (n, >=max k) urutan; repl(x, sel) -> x. Return p_ci (n,k), pred (n,k)."""
        batch = []
        for i in range(len(X)):
            for k in ks:
                batch.append(repl(X[i].copy(), orders[i][:k]))
        P = ctx.predict(np.stack(batch)).reshape(len(X), len(ks), -1)
        p_ci = P[np.arange(len(X))[:, None], np.arange(len(ks))[None, :], C[:, None]]
        return p_ci, P.argmax(2).astype(np.int8)

    def rf(x, sel):
        x[:, sel] = base[sel]
        return x

    def rt(x, sel):
        x[sel, :] = base[None, :]
        return x

    ordF = {m: np.argsort(-IF[m], axis=1, kind="stable") for m in METHODS_F}
    ordT = {m: np.argsort(-IT[m], axis=1, kind="stable") for m in METHODS_T}
    randF = [np.stack([np.random.default_rng(rand_seeds[i, r]).permutation(NF) for i in range(NI)]) for r in range(REP)]
    randT = [np.stack([np.random.default_rng(rand_seeds[i, r]).permutation(10) for i in range(NI)]) for r in range(REP)]

    # M2 lokal + kohort (primer)
    for m in METHODS_F:
        R[f"F_{m}_p"], R[f"F_{m}_pred"] = run(XP, ordF[m], KS_F, rf)
    for r in range(REP):
        R[f"F_acak{r}_p"], R[f"F_acak{r}_pred"] = run(XP, randF[r], KS_F, rf)
    run_log("C4 deletion: lokal/kohort primer selesai")
    # temporal
    for m in METHODS_T:
        R[f"T_{m}_p"], _ = run(XP, ordT[m], KS_T, rt)
    for r in range(REP):
        R[f"T_acak{r}_p"], _ = run(XP, randT[r], KS_T, rt)
    # pengganti empiris (10 ulangan)
    for j, b in enumerate(emp_pick):
        def re(x, sel, b=b):
            x[:, sel] = bg[b][:, sel]
            return x
        for m in METHODS_F:
            R[f"E{j}_{m}_p"], R[f"E{j}_{m}_pred"] = run(XP, ordF[m], KS_F, re)
        for r in range(REP):
            R[f"E{j}_acak{r}_p"], R[f"E{j}_acak{r}_pred"] = run(XP, randF[r], KS_F, re)
    run_log("C4 deletion: empiris lokal/kohort selesai")
    # kepekaan background (M2 lokal)
    for bgname in ("seimbang100", "empiris50", "empiris200"):
        for m in ("deepshap", "kustom_flat", "per_fitur"):
            o = np.argsort(-sens[f"{bgname}_IF_{m}"], axis=1, kind="stable")
            R[f"B_{bgname}_{m}_p"], _ = run(XP, o, KS_F, rf)
    # kontrol tanpa fitur >=99% sama (M2 lokal): urutan & acak hanya atas fitur tersisa
    ks_c = [k for k in KS_F if k <= len(keep_f)]
    for m in METHODS_F:
        o = keep_f[np.argsort(-IF[m][:, keep_f], axis=1, kind="stable")]
        R[f"K_{m}_p"], _ = run(XP, o, ks_c, rf)
    for r in range(REP):
        o = np.stack([keep_f[np.random.default_rng(rand_seeds[i, r]).permutation(len(keep_f))] for i in range(NI)])
        R[f"K_acak{r}_p"], _ = run(XP, o, ks_c, rf)
    run_log("C4 deletion: kepekaan bg & kontrol selesai")

    # sistem: urutan global I_f pada SELURUH window uji
    Xt = np.asarray(S["X_te"], dtype=np.float32)
    gord = {m: np.argsort(-(IF[m] / np.where(IF[m].sum(1, keepdims=True) == 0, 1, IF[m].sum(1, keepdims=True))).mean(0),
                          kind="stable") for m in METHODS_F}
    for r in range(REP):
        gord[f"acak{r}"] = np.random.default_rng(sys_seeds[r]).permutation(NF)

    def sys_pred(order, k, src):
        Xk = Xt.copy()
        Xk[:, :, order[:k]] = src[:, order[:k]] if src.ndim == 2 else src[order[:k]]
        return ctx.predict(Xk).argmax(1).astype(np.int8)

    R["S_k0_pred"] = ctx.predict(Xt).argmax(1).astype(np.int8)
    for name, o in gord.items():
        R[f"S_{name}_order"] = o
        for k in KS_F[1:]:
            R[f"S_{name}_k{k}_pred"] = sys_pred(o, k, base)
    run_log("C4 deletion: sistem primer selesai")
    for j, b in enumerate(emp_pick):
        for name, o in gord.items():
            for k in KS_F[1:]:
                R[f"SE{j}_{name}_k{k}_pred"] = sys_pred(o, k, bg[b])
    run_log("C4 deletion: sistem empiris selesai")
    np.savez_compressed(out, **R, rand_seeds=rand_seeds, sys_seeds=sys_seeds, emp_pick=emp_pick,
                        const99=np.array(const99), keep_f=keep_f, ks_kontrol=np.array(ks_c),
                        y_test=S["y_te"], sess_test=S["meta_te"].session_id.to_numpy().astype(str))
    run_log("C4 deletion selesai")


def main():
    part = sys.argv[1]
    t0 = time.time()
    S = setup()
    if part == "primer":
        part_primer(S)
    elif part == "m4":
        part_m4(S)
    elif part == "m3_proc":
        part_m3_proc(S)
    elif part.startswith("m3_pert_"):
        part_m3_pert(S, float(part.split("_")[-1]))
    elif part == "sens_bg":
        part_sens_bg(S)
    elif part == "deletion":
        part_deletion(S)
    run_log(f"C4 bagian {part}: {time.time() - t0:.0f}s, puncak memori {peak_mb():.0f} MB")


if __name__ == "__main__":
    main()
