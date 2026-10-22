"""
D3.4 laporan: SALINAN scripts/stage_c/c4_report.py. Perubahan hanya jalur:
artefak dari artifacts/stage_d/ (c4_parts, attributions_450_v2.npz), tabel ke
results/stage_d/d3_c4_*.csv. M2 sistem otomatis memakai partisi uji final
(y_test/sess_test di deletion.npz berasal dari d3_c4_xai.py). Isi asli di bawah.

C4 laporan: tabel akhir dari artefak c4_xai.py (tanpa TensorFlow).

Definisi (rev8 + pilot B2):
- AUC = trapesium atas k. dAUC = AUC(acak, rerata 5 ulangan) - AUC(metode).
- Lokal: per instance, probabilitas kelas c_i (dikunci sejak k=0).
- Kohort: F1-macro pada 450 instance; sistem: F1-macro pada seluruh window uji
  dengan urutan global I_f (I_{i,f} dinormalisasi per instance lalu dirata).
- M3: Stab@k = rerata Jaccard@k berpasangan (10 pasang dari R=5); k=10,
  per time-step k=3; Spearman berpasangan dilaporkan.
- Bootstrap: 1.000 ulangan, rng default_rng(42); sesi diambil ulang dengan
  pengembalian di dalam tiap kelas (kelas instance = label sebenarnya; untuk
  tingkat sistem: label sesi), bobot r; bobot yang sama dipakai untuk kedua
  metode. Primer P1-P4: persentil 98,75%; eksploratori: 95%. Bootstrap
  tingkat instance (instance diambil ulang di dalam kelas) sebagai pembanding.
"""

import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_c"))
sys.path.insert(0, os.path.dirname(__file__))
from common_c import C_ART, C_RES, CLASSES, run_log  # noqa: E402
# ---- Tahap D: keluaran ke artifacts/stage_d & results/stage_d; model & freeze Tahap C (beku)
C_ART, C_RES, C_RES_C = "artifacts/stage_d", "results/stage_d", "results/stage_c"
from common_d import run_log  # noqa: E402,F811
from c4_core import jacc, norm_rows, spear  # noqa: E402

PARTS = f"{C_ART}/c4_parts"
KS_F, KS_T, REP, NB = [0, 1, 3, 5, 10, 15], [0, 1, 2, 3, 5], 5, 1000
MF = ["deepshap", "lime_flat", "kustom_flat", "per_fitur"]
MT = ["deepshap", "lime_flat", "kustom_flat", "per_timestep"]
ALL5 = MF + ["per_timestep"]
PRIMARY = {"P1": ("m2_lokal", "deepshap", "lime_flat"), "P2": ("m2_lokal", "kustom_flat", "per_fitur"),
           "P3": ("m3_perturbasi", "deepshap", "lime_flat"), "P4": ("m3_perturbasi", "kustom_flat", "per_fitur")}


def auc(c, ks):
    return np.trapz(c, x=ks, axis=-1)


def f1w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    cm = np.bincount(y * 9 + p, weights=w, minlength=81).reshape(9, 9)
    tp = np.diag(cm)
    pr = np.divide(tp, cm.sum(0), out=np.zeros(9), where=cm.sum(0) > 0)
    rc = np.divide(tp, cm.sum(1), out=np.zeros(9), where=cm.sum(1) > 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(9), where=(pr + rc) > 0)
    return f.mean()


def stab(runs, k):
    R_, n = runs.shape[0], runs.shape[1]
    J, S = np.zeros(n), np.zeros(n)
    for i in range(n):
        js, ss = [], []
        for a, b in combinations(range(R_), 2):
            js.append(jacc(runs[a, i], runs[b, i], k))
            ss.append(spear(runs[a, i], runs[b, i]))
        J[i] = np.mean(js)
        S[i] = np.nanmean(ss) if np.isfinite(ss).any() else np.nan
    return J, S


def session_weights(cls, sess, rng):
    """Bobot r per elemen: sesi diambil ulang dengan pengembalian di dalam tiap kelas."""
    w = np.zeros(len(cls))
    for c in np.unique(cls):
        m = np.flatnonzero(cls == c)
        us, inv = np.unique(sess[m], return_inverse=True)
        cnt = np.bincount(rng.integers(0, len(us), size=len(us)), minlength=len(us))
        w[m] = cnt[inv]
    return w


def instance_weights(cls, rng):
    w = np.zeros(len(cls))
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        w[idx] = np.bincount(rng.choice(len(idx), size=len(idx), replace=True), minlength=len(idx))
    return w


def ci(stats, level):
    a = (100 - level) / 2
    return np.percentile(stats, [a, 100 - a])


def main():
    at = np.load(f"{C_ART}/attributions_450_v2.npz")
    D = np.load(f"{PARTS}/deletion.npz")
    YP, C, SESS = at["y_true"], at["c_pred"], at["session_id"]
    NI = len(YP)
    IF = {m: at[f"IF_{m}"] for m in MF}
    feats = list(at["features"])

    # ---------------- M1
    m1 = []
    for other in ("lime_flat", "kustom_flat"):
        a_, b_ = IF["deepshap"], IF[other]
        ga, gb = norm_rows(a_).mean(0), norm_rows(b_).mean(0)
        for k in (5, 10, 15):
            per = np.array([jacc(a_[i], b_[i], k) for i in range(NI)])
            m1.append(dict(pasangan=f"deepshap_vs_{other}", metrik=f"jaccard@{k}", global_=jacc(ga, gb, k),
                           per_instance_mean=per.mean(), per_instance_sd=per.std(ddof=1), n_nan=0))
        per = np.array([spear(a_[i], b_[i]) for i in range(NI)])
        m1.append(dict(pasangan=f"deepshap_vs_{other}", metrik="spearman", global_=spear(ga, gb),
                       per_instance_mean=np.nanmean(per), per_instance_sd=np.nanstd(per, ddof=1),
                       n_nan=int(np.isnan(per).sum())))
    pd.DataFrame(m1).rename(columns={"global_": "global"}).to_csv(f"{C_RES}/d3_c4_m1.csv", index=False)

    # ---------------- M2 lokal / kohort / sistem
    def local_dauc(prefix, methods, ks):
        A = {m: auc(D[f"{prefix}_{m}_p"], ks) for m in methods}
        Ar = np.mean([auc(D[f"{prefix}_acak{r}_p"], ks) for r in range(REP)], axis=0)
        return {m: Ar - A[m] for m in methods}, A, Ar

    dl, Al, Arl = local_dauc("F", MF, KS_F)
    pd.DataFrame([dict(metode=m, auc_mean=Al[m].mean(), auc_acak_mean=Arl.mean(), dauc_mean=dl[m].mean(),
                       dauc_sd=dl[m].std(ddof=1), n=NI) for m in MF]).to_csv(f"{C_RES}/d3_c4_m2_local.csv", index=False)
    pd.DataFrame(dict(instance=np.arange(NI), session_id=SESS, kelas=[CLASSES[y] for y in YP],
                      **{f"dauc_{m}": dl[m] for m in MF})).to_csv(f"{C_RES}/d3_c4_m2_local_per_instance.csv", index=False)

    def cohort_curves(prefix, w=None):
        cur = {m: np.array([f1w(YP, D[f"{prefix}_{m}_pred"][:, j], w) for j in range(len(KS_F))]) for m in MF}
        for r in range(REP):
            cur[f"acak{r}"] = np.array([f1w(YP, D[f"{prefix}_acak{r}_pred"][:, j], w) for j in range(len(KS_F))])
        return cur

    def dauc_from_curves(cur):
        ar = np.mean([auc(cur[f"acak{r}"], KS_F) for r in range(REP)])
        return {m: ar - auc(cur[m], KS_F) for m in MF}, ar

    coh = cohort_curves("F")
    dk, ark = dauc_from_curves(coh)
    pd.DataFrame([dict(metode=m, **{f"f1_k{k}": v for k, v in zip(KS_F, coh[m])}, auc=auc(coh[m], KS_F),
                       auc_acak_mean=ark, dauc=dk.get(m, np.nan)) for m in coh]).to_csv(
        f"{C_RES}/d3_c4_m2_cohort.csv", index=False)

    yt, st = D["y_test"].astype(int), D["sess_test"]
    si = pd.read_pickle("artifacts/stage_a/session_index.pkl").set_index("session_id")
    cls_t = np.array([CLASSES.index(si.loc[s, "kelas_sesi"]) for s in st])

    def sys_curves(prefix, w=None):
        cur = {}
        for name in MF + [f"acak{r}" for r in range(REP)]:
            cur[name] = np.array([f1w(yt, D["S_k0_pred"].astype(int), w)] +
                                 [f1w(yt, D[f"{prefix}_{name}_k{k}_pred"].astype(int), w) for k in KS_F[1:]])
        return cur

    sc_ = sys_curves("S")
    ds_, ars = dauc_from_curves(sc_)
    pd.DataFrame([dict(metode=m, urutan_global=",".join(feats[j] for j in D[f"S_{m}_order"][:15]),
                       **{f"f1_k{k}": v for k, v in zip(KS_F, sc_[m])}, auc=auc(sc_[m], KS_F), auc_acak_mean=ars,
                       dauc=ds_.get(m, np.nan)) for m in sc_]).to_csv(f"{C_RES}/d3_c4_m2_system.csv", index=False)

    dt, At, Art = local_dauc("T", MT, KS_T)
    pd.DataFrame([dict(metode=m, auc_mean=At[m].mean(), auc_acak_mean=Art.mean(), dauc_mean=dt[m].mean(),
                       dauc_sd=dt[m].std(ddof=1), n=NI) for m in MT]).to_csv(f"{C_RES}/d3_c4_temporal.csv", index=False)

    # ---------------- M3
    m3, m3i, stabs = [], [], {}
    sources = [("prosedural", f"{PARTS}/m3_proc.npz")] + \
              [(f"perturbasi_{s}", f"{PARTS}/m3_pert_{s}.npz") for s in ("0.01", "0.005", "0.02")]
    for jenis, path in sources:
        Z = np.load(path)
        for m in ALL5:
            k = 3 if m == "per_timestep" else 10
            J, S = stab(Z[m], k)
            stabs[(jenis, m)] = J
            m3.append(dict(jenis=jenis, metode=m, k=k, stab_mean=J.mean(), stab_sd=J.std(ddof=1),
                           spearman_mean=np.nanmean(S), spearman_sd=np.nanstd(S, ddof=1), n_nan_spearman=int(np.isnan(S).sum())))
            m3i.append(pd.DataFrame(dict(jenis=jenis, metode=m, instance=np.arange(NI), stab=J, spearman=S)))
    pd.DataFrame(m3).to_csv(f"{C_RES}/d3_c4_m3.csv", index=False)
    pd.concat(m3i).to_csv(f"{C_RES}/d3_c4_m3_per_instance.csv", index=False)

    # ---------------- bootstrap
    per_inst = {("m2_lokal", m): dl[m] for m in MF}
    per_inst.update({("m3_perturbasi", m): stabs[("perturbasi_0.01", m)] for m in ALL5})
    per_inst.update({("temporal", m): dt[m] for m in MT})
    rng = np.random.default_rng(42)
    rng_i = np.random.default_rng(43)
    Ws = np.stack([session_weights(YP, SESS, rng) for _ in range(NB)])
    Wi = np.stack([instance_weights(YP, rng_i) for _ in range(NB)])

    prim = []
    for pid, (metric, a, b) in PRIMARY.items():
        d = per_inst[(metric, a)] - per_inst[(metric, b)]
        for unit, W in (("sesi", Ws), ("instance", Wi)):
            stats = (W * d).sum(1) / W.sum(1)
            lo, hi = ci(stats, 98.75)
            prim.append(dict(perbandingan=pid, metrik=metric, metode_a=a, metode_b=b, selisih_titik=d.mean(),
                             batas_bawah=lo, batas_atas=hi, tingkat=98.75, unit=unit,
                             satu_sisi_nol=bool(lo > 0 or hi < 0) if unit == "sesi" else np.nan))
    pd.DataFrame(prim).to_csv(f"{C_RES}/d3_c4_bootstrap_primary.csv", index=False)

    expl = []
    # temporal lokal (per instance)
    for a, b in combinations(MT, 2):
        d = dt[a] - dt[b]
        stats = (Ws * d).sum(1) / Ws.sum(1)
        lo, hi = ci(stats, 95)
        expl.append(dict(metrik="temporal_lokal_dauc", metode_a=a, metode_b=b, selisih_titik=d.mean(),
                         batas_bawah=lo, batas_atas=hi, tingkat=95, unit="sesi"))
    # kohort (F1 berbobot)
    boot_k = []
    for bi in range(NB):
        dk_b, _ = dauc_from_curves(cohort_curves("F", Ws[bi]))
        boot_k.append(dk_b)
    for a, b in combinations(MF, 2):
        stats = np.array([x[a] - x[b] for x in boot_k])
        lo, hi = ci(stats, 95)
        expl.append(dict(metrik="m2_kohort_dauc", metode_a=a, metode_b=b, selisih_titik=dk[a] - dk[b],
                         batas_bawah=lo, batas_atas=hi, tingkat=95, unit="sesi"))
    # sistem (sesi uji di dalam kelas sesi)
    rng_s = np.random.default_rng(44)
    boot_s = []
    for bi in range(NB):
        w = session_weights(cls_t, st, rng_s)
        dsb, _ = dauc_from_curves(sys_curves("S", w))
        boot_s.append(dsb)
    for a, b in combinations(MF, 2):
        stats = np.array([x[a] - x[b] for x in boot_s])
        lo, hi = ci(stats, 95)
        expl.append(dict(metrik="m2_sistem_dauc", metode_a=a, metode_b=b, selisih_titik=ds_[a] - ds_[b],
                         batas_bawah=lo, batas_atas=hi, tingkat=95, unit="sesi_uji"))
    pd.DataFrame(expl).to_csv(f"{C_RES}/d3_c4_bootstrap_exploratory.csv", index=False)
    run_log("C4 laporan: bootstrap selesai")

    # ---------------- kepekaan
    sens = []
    SB = np.load(f"{PARTS}/sens_bg.npz")
    for bgname in ("seimbang100", "empiris50", "empiris200"):
        IFv = dict(IF, deepshap=SB[f"{bgname}_IF_deepshap"], kustom_flat=SB[f"{bgname}_IF_kustom_flat"],
                   per_fitur=SB[f"{bgname}_IF_per_fitur"])
        for other in ("lime_flat", "kustom_flat"):
            a_, b_ = IFv["deepshap"], IFv[other]
            ga, gb = norm_rows(a_).mean(0), norm_rows(b_).mean(0)
            sens.append(dict(skenario=f"bg_{bgname}", metrik="m1_jaccard@10_global", objek=f"deepshap_vs_{other}",
                             nilai=jacc(ga, gb, 10)))
            sens.append(dict(skenario=f"bg_{bgname}", metrik="m1_jaccard@10_per_instance_mean", objek=f"deepshap_vs_{other}",
                             nilai=np.mean([jacc(a_[i], b_[i], 10) for i in range(NI)])))
            sens.append(dict(skenario=f"bg_{bgname}", metrik="m1_spearman_per_instance_mean", objek=f"deepshap_vs_{other}",
                             nilai=np.nanmean([spear(a_[i], b_[i]) for i in range(NI)])))
        dv = {"lime_flat": dl["lime_flat"]}
        for m in ("deepshap", "kustom_flat", "per_fitur"):
            dv[m] = Arl - auc(D[f"B_{bgname}_{m}_p"], KS_F)
        for m in MF:
            sens.append(dict(skenario=f"bg_{bgname}", metrik="m2_lokal_dauc_mean", objek=m, nilai=dv[m].mean(),
                             catatan="lime_flat tidak memakai background (= primer)" if m == "lime_flat" else ""))
        for pid in ("P1", "P2"):
            _, a, b = PRIMARY[pid]
            sens.append(dict(skenario=f"bg_{bgname}", metrik=f"{pid}_selisih_titik", objek=f"{a}-{b}",
                             nilai=(dv[a] - dv[b]).mean()))
        sens.append(dict(skenario=f"bg_{bgname}", metrik="additivity_galat_maks", objek="deepshap",
                         nilai=float(SB[f"{bgname}_additivity_maks"])))
    # pengganti empiris
    emp_l = {m: [] for m in MF}
    emp_k = {m: [] for m in MF}
    emp_s = {m: [] for m in MF}
    for j in range(10):
        d_, _, _ = local_dauc(f"E{j}", MF, KS_F)
        for m in MF:
            emp_l[m].append(d_[m].mean())
        dk_j, _ = dauc_from_curves(cohort_curves(f"E{j}"))
        ds_j, _ = dauc_from_curves(sys_curves(f"SE{j}"))
        for m in MF:
            emp_k[m].append(dk_j[m])
            emp_s[m].append(ds_j[m])
    for lvl, prim_v, emp in (("lokal", {m: dl[m].mean() for m in MF}, emp_l), ("kohort", dk, emp_k), ("sistem", ds_, emp_s)):
        e_mean = {m: float(np.mean(emp[m])) for m in MF}
        for m in MF:
            sens.append(dict(skenario="pengganti_empiris_10", metrik=f"m2_{lvl}_dauc", objek=m, nilai=e_mean[m],
                             catatan=f"primer (median/modus) = {prim_v[m]:.6f}; sd antar-ulangan = {np.std(emp[m], ddof=1):.6f}"))
        o_p = sorted(MF, key=lambda m: -prim_v[m])
        o_e = sorted(MF, key=lambda m: -e_mean[m])
        sens.append(dict(skenario="pengganti_empiris_10", metrik=f"m2_{lvl}_urutan_metode_sama", objek=">".join(o_e),
                         nilai=float(o_p == o_e), catatan=f"urutan primer: {'>'.join(o_p)}"))
    # sigma
    for s in ("0.005", "0.01", "0.02"):
        for m in ALL5:
            sens.append(dict(skenario=f"sigma_{s}", metrik="m3_perturbasi_stab_mean", objek=m,
                             nilai=stabs[(f"perturbasi_{s}", m)].mean()))
        for pid in ("P3", "P4"):
            _, a, b = PRIMARY[pid]
            sens.append(dict(skenario=f"sigma_{s}", metrik=f"{pid}_selisih_titik", objek=f"{a}-{b}",
                             nilai=(stabs[(f"perturbasi_{s}", a)] - stabs[(f"perturbasi_{s}", b)]).mean()))
    # kontrol tanpa fitur >= 99% sama
    keep = D["keep_f"]
    ksc = list(D["ks_kontrol"])
    excl = ",".join(feats[f] for f in D["const99"])
    for other in ("lime_flat", "kustom_flat"):
        a_, b_ = IF["deepshap"][:, keep], IF[other][:, keep]
        ga, gb = norm_rows(a_).mean(0), norm_rows(b_).mean(0)
        for k in (5, 10, 15):
            if k > len(keep):
                continue
            sens.append(dict(skenario="kontrol_tanpa_fitur_99", metrik=f"m1_jaccard@{k}_global", objek=f"deepshap_vs_{other}",
                             nilai=jacc(ga, gb, k), catatan=f"dikeluarkan: {excl}"))
            sens.append(dict(skenario="kontrol_tanpa_fitur_99", metrik=f"m1_jaccard@{k}_per_instance_mean",
                             objek=f"deepshap_vs_{other}", nilai=np.mean([jacc(a_[i], b_[i], k) for i in range(NI)])))
        sens.append(dict(skenario="kontrol_tanpa_fitur_99", metrik="m1_spearman_global", objek=f"deepshap_vs_{other}",
                         nilai=spear(ga, gb)))
        sens.append(dict(skenario="kontrol_tanpa_fitur_99", metrik="m1_spearman_per_instance_mean",
                         objek=f"deepshap_vs_{other}", nilai=np.nanmean([spear(a_[i], b_[i]) for i in range(NI)])))
    dK, _, _ = local_dauc("K", MF, ksc)
    for m in MF:
        sens.append(dict(skenario="kontrol_tanpa_fitur_99", metrik="m2_lokal_dauc_mean", objek=m, nilai=dK[m].mean(),
                         catatan=f"k={ksc}; dikeluarkan: {excl}"))
    pd.DataFrame(sens).to_csv(f"{C_RES}/d3_c4_sensitivity.csv", index=False)

    # ---------------- per kelas
    pc = []
    for ci_, c in enumerate(CLASSES):
        m = YP == ci_
        row = dict(kelas=c, n=int(m.sum()), n_sesi=len(np.unique(SESS[m])), eksploratori=(c == "backdoor"))
        for other in ("lime_flat", "kustom_flat"):
            row[f"m1_jaccard@10_deepshap_vs_{other}"] = np.mean([jacc(IF["deepshap"][i], IF[other][i], 10)
                                                                  for i in np.flatnonzero(m)])
            row[f"m1_spearman_deepshap_vs_{other}"] = np.nanmean([spear(IF["deepshap"][i], IF[other][i])
                                                                   for i in np.flatnonzero(m)])
        for mth in MF:
            row[f"m2_lokal_dauc_{mth}"] = dl[mth][m].mean()
        for mth in ALL5:
            row[f"m3_prosedural_{mth}"] = stabs[("prosedural", mth)][m].mean()
            row[f"m3_perturbasi_{mth}"] = stabs[("perturbasi_0.01", mth)][m].mean()
        rng_c = np.random.default_rng(42)
        Wc = np.stack([session_weights(YP[m], SESS[m], rng_c) for _ in range(NB)])
        for pid, (metric, a, b) in PRIMARY.items():
            d = (per_inst[(metric, a)] - per_inst[(metric, b)])[m]
            lo, hi = ci((Wc * d).sum(1) / Wc.sum(1), 95)
            row.update({f"{pid}_selisih": d.mean(), f"{pid}_ci95_bawah": lo, f"{pid}_ci95_atas": hi})
        pc.append(row)
    pd.DataFrame(pc).to_csv(f"{C_RES}/d3_c4_per_class.csv", index=False)
    run_log("C4 laporan selesai")


if __name__ == "__main__":
    main()
