"""
Tahap A (rancangan paper rev6): sesionisasi final TANPA LABEL + hitung ulang
Tabel 2-5. TIDAK melatih model apa pun. Kode lama hanya dibaca.

Keluaran: results/stage_a/*.csv + summary.json, artifacts/stage_a/*.

Definisi yang dipakai (dicatat juga di summary.json):
- Sesi: kunci (src_ip, dst_ip, dst_port, proto), urut ts (tie -> urutan baris
  asli, sort stabil), potong bila jeda > 15 detik. Label TIDAK dipakai.
- Label sesi = modus label baris; seri -> kelas urutan alfabetis pertama.
- Cakupan kelas c = baris berlabel-baris c yang berada di sesi >=10 baris
  / seluruh baris berlabel c setelah pembersihan.
- Jumlah sesi per kelas dihitung menurut label sesi.
- Split per kelas (label sesi): urut t_mulai_asli (tie -> session_id),
  n_train = floor(0.70 n), n_val = floor(0.85 n) - n_train, sisanya uji.
- Purging: sesi partisi awal dibuang bila t_akhir_asli > t_mulai_asli sesi
  pertama partisi berikutnya (strict >).
"""

import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline_common import CHUNKSIZE, sorted_network_files  # noqa: E402

SEED = 42
GAP = 15
MIN_LEN = 10
W = 10
Q = 120_000
K_MIN = 20
N_SAMPLE_BIG = 6_000
DASH_THRESHOLD = 0.95
GAPS_TABLE4 = [10, 15, 30, 60]

KEYS = {
    "3_atribut": ["src_ip", "dst_ip", "dst_port"],
    "4_atribut": ["src_ip", "dst_ip", "dst_port", "proto"],
    "5_atribut_src_port": ["src_ip", "src_port", "dst_ip", "dst_port", "proto"],
}
PROTECTED = {"src_ip", "dst_ip", "dst_port", "proto", "src_port", "ts", "label", "type"}
NON_FEATURE = {"ts", "src_ip", "dst_ip", "dst_port", "label", "type"}
PARTS = ["train", "val", "test"]

RES = os.environ.get("STAGE_A_RES", "results/stage_a")
ART = os.environ.get("STAGE_A_ART", "artifacts/stage_a")
SMOKE_FILES = int(os.environ.get("STAGE_A_SMOKE_FILES", "0"))  # uji asap: tiap berkas ke-k; 0 = semua (run resmi)
os.makedirs(RES, exist_ok=True)
os.makedirs(ART, exist_ok=True)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def raw_chunks():
    """Yield (file_idx, chunk) mentah (uid sudah dibuang) dari 23 berkas."""
    files = sorted_network_files()
    if SMOKE_FILES:
        files = files[::SMOKE_FILES]
    for fi, path in enumerate(files):
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype={"src_bytes": "string"},
                                 low_memory=False):
            if "uid" in chunk.columns:
                chunk = chunk.drop(columns=["uid"])
            yield fi, chunk


def clean_chunk(chunk):
    bad = (chunk["src_ip"] == "0") | (chunk["src_bytes"] == "0.0.0.0")
    c2 = chunk.loc[~bad]
    c3 = c2.loc[c2["type"] != "mitm"].copy()
    c3["src_bytes"] = c3["src_bytes"].astype("int64")
    return c2, c3


def key_hash(chunk, cols):
    return pd.util.hash_pandas_object(chunk[cols].astype(str), index=False).to_numpy()


# --------------------------------------------------------------------------- PASS 1
def pass1():
    log("PASS 1: baca 23 berkas, hitung invarian, hash kunci sesi")
    st1, st2, st3 = Counter(), Counter(), Counter()
    columns, dash, is_cat = None, None, None
    H = {k: [] for k in KEYS}
    TS, LAB = [], []
    n3 = 0
    raw_code = {}
    for fi, chunk in raw_chunks():
        if columns is None:
            columns = list(chunk.columns)
            dash = Counter()
            is_cat = {c: False for c in columns}
        st1.update(chunk["type"].value_counts().to_dict())
        c2, c3 = clean_chunk(chunk)
        st2.update(c2["type"].value_counts().to_dict())
        if c3.empty:
            continue
        st3.update(c3["type"].value_counts().to_dict())
        n3 += len(c3)
        for c in columns:
            if pd.api.types.is_string_dtype(c3[c]):
                is_cat[c] = True
                dash[c] += int((c3[c] == "-").sum())
        for k, cols in KEYS.items():
            H[k].append(key_hash(c3, cols))
        TS.append(c3["ts"].to_numpy(np.int64))
        for c in c3["type"].unique():
            raw_code.setdefault(c, len(raw_code))
        LAB.append(c3["type"].map(raw_code).to_numpy(np.int8))
    log(f"PASS 1 selesai: {n3:,} baris bersih")
    classes = sorted(st3)
    cls_idx = {c: i for i, c in enumerate(classes)}
    remap = np.zeros(len(raw_code), dtype=np.int8)
    for c, r in raw_code.items():
        remap[r] = cls_idx[c]
    lab = remap[np.concatenate(LAB)]
    del LAB
    ts = np.concatenate(TS)
    del TS
    H = {k: np.concatenate(v) for k, v in H.items()}
    dash_pct = {c: dash[c] / n3 for c in columns}
    dropped = [c for c in columns if c not in PROTECTED and dash_pct[c] >= DASH_THRESHOLD]
    kept = [c for c in columns if c not in dropped]
    return dict(st1=st1, st2=st2, st3=st3, classes=classes, lab=lab, ts=ts, H=H,
                columns=columns, dropped=dropped, kept=kept, is_cat=is_cat,
                dash_pct=dash_pct, n3=n3)


# --------------------------------------------------------------------------- sesionisasi
def sort_by_key(h, ts):
    return np.lexsort((ts, h))  # stabil: tie ts -> urutan baris asli


def sessionize(h_s, ts_s, gap):
    n = len(h_s)
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = (h_s[1:] != h_s[:-1]) | (np.diff(ts_s) > gap)
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, n))
    return starts, sizes


def session_labels(lab_s, starts, sizes, ncls, sel_mask=None):
    """Modus label per sesi (seri -> indeks kelas terkecil) dan flag campuran.
    sel_mask: subset sesi yang dihitung (None = semua)."""
    sid = np.repeat(np.arange(len(starts)), sizes)
    if sel_mask is not None:
        rows = sel_mask[sid]
        sid_r, lab_r = sid[rows], lab_s[rows]
    else:
        sid_r, lab_r = sid, lab_s
    key = sid_r.astype(np.int64) * ncls + lab_r
    u, cnt = np.unique(key, return_counts=True)
    s, l = u // ncls, (u % ncls).astype(np.int8)
    order = np.lexsort((l, -cnt, s))
    s_o = s[order]
    first = np.ones(len(s_o), dtype=bool)
    first[1:] = s_o[1:] != s_o[:-1]
    mode = np.full(len(starts), -1, dtype=np.int8)
    mode[s_o[first]] = l[order][first]
    nlab = np.bincount(s, minlength=len(starts))
    return mode, nlab > 1


def coverage(lab_s, starts, sizes, ncls):
    big = sizes >= MIN_LEN
    rowbig = np.repeat(big, sizes)
    rows_cov = np.bincount(lab_s[rowbig], minlength=ncls)
    rows_all = np.bincount(lab_s, minlength=ncls)
    mode, mixed = session_labels(lab_s, starts, sizes, ncls, sel_mask=big)
    sess_big = np.bincount(mode[big], minlength=ncls)
    n_win = int((sizes[big] - (W - 1)).sum())
    return dict(rows_cov=rows_cov, rows_all=rows_all, sess_big=sess_big, big=big,
                mode=mode, mixed=mixed, n_win=n_win)


# --------------------------------------------------------------------------- utama
def main():
    P = pass1()
    classes, ncls, lab, ts = P["classes"], len(P["classes"]), P["lab"], P["ts"]
    summary = {"seed": SEED, "classes": classes}

    # ---------------- cleaning log + invarian
    cl_rows = []
    all_cls = sorted(set(P["st1"]) | set(P["st2"]) | set(P["st3"]))
    for c in all_cls:
        cl_rows.append(dict(kelas=c, t1_mentah=P["st1"].get(c, 0), t2_buang_baris_rusak=P["st2"].get(c, 0),
                            t3_buang_mitm=P["st3"].get(c, 0), t4_buang_kolom_dash=P["st3"].get(c, 0)))
    cl = pd.DataFrame(cl_rows)
    cl.loc[len(cl)] = ["TOTAL"] + [int(cl[c].sum()) for c in cl.columns[1:]]
    cl.to_csv(f"{RES}/cleaning_log.csv", index=False)
    tot1, tot2, tot3 = (sum(P[k].values()) for k in ("st1", "st2", "st3"))
    summary["total_baris_per_tahap"] = {
        "1_mentah_gabungan": tot1, "2_buang_baris_rusak": tot2,
        "baris_rusak_dibuang": tot1 - tot2, "3_buang_mitm": tot3,
        "mitm_dibuang": P["st2"].get("mitm", 0),
        "4_buang_kolom_dash": tot3,
    }
    summary["invarian"] = {
        "t1_sama_22339021": tot1 == 22_339_021,
        "baris_rusak_sama_869": (tot1 - tot2) == 869,
        "t1_minus_rusak_eq_t2": tot1 - (tot1 - tot2) == tot2,
        "t2_minus_mitm_eq_t3": tot2 - P["st2"].get("mitm", 0) == tot3,
        "jumlah_baris_array_eq_t3": int(len(lab)) == tot3,
        "per_kelas_t2_eq_t3_kecuali_mitm": all(P["st2"].get(c, 0) == P["st3"].get(c, 0)
                                                for c in classes),
    }
    summary["kolom"] = {"total_setelah_buang_uid": len(P["columns"]),
                        "dibuang_dash95": P["dropped"], "n_dibuang_dash95": len(P["dropped"]),
                        "n_dibuang_sama_17": len(P["dropped"]) == 17, "kept": P["kept"]}
    log(f"Invarian: {summary['invarian']}; kolom dash dibuang = {len(P['dropped'])}")

    # ---------------- Tabel 3 (skema kunci, gap 15) dan Tabel 4 (gap, 4 atribut)
    def _ci(name):
        return classes.index(name) if name in classes else None

    def _get(arr, i):
        return float("nan") if i is None else arr[i]

    t3_rows, t4_rows = [], []
    main_state = None
    for scheme in ["3_atribut", "5_atribut_src_port", "4_atribut"]:
        h = P["H"][scheme]
        order = sort_by_key(h, ts)
        h_s, ts_s, lab_s = h[order], ts[order], lab[order]
        gaps = GAPS_TABLE4 if scheme == "4_atribut" else [GAP]
        for g in gaps:
            starts, sizes = sessionize(h_s, ts_s, g)
            cov = coverage(lab_s, starts, sizes, ncls)
            pct = 100 * cov["rows_cov"] / cov["rows_all"]
            if g == GAP:
                lo = int(np.argmin(pct))
                t3_rows.append(dict(
                    skema=scheme, kelas_cakupan_terendah=classes[lo],
                    cakupan_terendah_persen=round(float(pct[lo]), 4),
                    kelas_tanpa_sesi_ge10=";".join(classes[i] for i in range(ncls) if cov["sess_big"][i] == 0) or "-",
                    **{f"cakupan_{c}": round(float(pct[i]), 4) for i, c in enumerate(classes)},
                    **{f"sesi_ge10_{c}": int(cov["sess_big"][i]) for i, c in enumerate(classes)},
                ))
            if scheme == "4_atribut":
                t4_rows.append(dict(
                    ambang=g, cakupan_scanning=round(float(_get(pct, _ci("scanning"))), 4),
                    cakupan_normal=round(float(_get(pct, _ci("normal"))), 4),
                    sesi_backdoor_ge10=_get(cov["sess_big"], _ci("backdoor")),
                    total_window_sebelum_kuota=cov["n_win"],
                    total_sesi_ge10=int(cov["big"].sum()),
                ))
            if scheme == "4_atribut" and g == GAP:
                mode_all, mixed_all = session_labels(lab_s, starts, sizes, ncls)
                main_state = dict(order=order, ts_s=ts_s, lab_s=lab_s, starts=starts, sizes=sizes,
                                  cov=cov, mode_all=mode_all, mixed_all=mixed_all, pct=pct)
            log(f"skema {scheme} gap {g}: sesi={len(starts):,} sesi>=10={int(cov['big'].sum()):,}")
        del h_s
        if scheme != "4_atribut":
            del order, ts_s, lab_s
    pd.DataFrame(t3_rows).to_csv(f"{RES}/table3_key_schemes.csv", index=False)
    pd.DataFrame(t4_rows).to_csv(f"{RES}/table4_gap.csv", index=False)
    del P["H"]

    M = main_state
    starts, sizes, ts_s, lab_s, order = M["starts"], M["sizes"], M["ts_s"], M["lab_s"], M["order"]
    cov = M["cov"]

    # ---------------- Tabel 2
    t2 = pd.DataFrame(dict(
        kelas=classes,
        baris_setelah_pembersihan=cov["rows_all"],
        sesi_terbentuk=np.bincount(M["mode_all"], minlength=ncls),
        baris_dalam_sesi_ge10=cov["rows_cov"],
        sesi_ge10=cov["sess_big"],
        cakupan_persen=np.round(M["pct"], 4),
    ))
    t2.loc[len(t2)] = ["TOTAL", int(t2.baris_setelah_pembersihan.sum()), int(len(starts)),
                       int(t2.baris_dalam_sesi_ge10.sum()), int(cov["big"].sum()),
                       round(100 * t2.baris_dalam_sesi_ge10.sum() / t2.baris_setelah_pembersihan.sum(), 4)]
    t2.to_csv(f"{RES}/table2_sessions.csv", index=False)

    # ---------------- sesi >=10, session_id berurut (t_mulai, indeks)
    big_idx = np.flatnonzero(cov["big"])
    t_start_all = ts_s[starts]
    t_end_all = ts_s[starts + sizes - 1]
    big_idx = big_idx[np.lexsort((big_idx, t_start_all[big_idx]))]
    S = pd.DataFrame(dict(
        sidx=big_idx, session_id=np.arange(len(big_idx)),
        kelas_idx=M["mode_all"][big_idx], start_sorted=starts[big_idx], n_asli=sizes[big_idx],
        t_mulai_asli=t_start_all[big_idx], t_akhir_asli=t_end_all[big_idx],
        campuran=M["mixed_all"][big_idx],
    ))
    S["kelas_sesi"] = [classes[i] for i in S.kelas_idx]
    S["session_id"] = ["S%07d" % i for i in S.session_id]

    # ---------------- sampling kuota + crop
    rng = np.random.default_rng(SEED)
    picked, quota_info = [], {}
    for ci, c in enumerate(classes):
        sub = S[S.kelas_idx == ci]
        Nc = len(sub)
        if Nc == 0:
            quota_info[c] = dict(N_c=0, K_c=None, mode="kosong")
            continue
        Kc = max(K_MIN, Q // Nc)
        if Nc * K_MIN > Q:
            take = np.sort(rng.choice(Nc, size=N_SAMPLE_BIG, replace=False))
            sub = sub.iloc[take]
            mode = f"acak_{N_SAMPLE_BIG}_sesi"
        else:
            mode = "semua_sesi"
        sub = sub.copy()
        sub["K_c"] = Kc
        sub["panjang_blok"] = np.minimum(sub.n_asli.to_numpy(), Kc)
        sub["offset_blok"] = [int(rng.integers(0, n - b + 1)) for n, b in zip(sub.n_asli, sub.panjang_blok)]
        picked.append(sub)
        quota_info[c] = dict(N_c=Nc, K_c=int(Kc), mode=mode, sesi_terpilih=len(sub))
    S = pd.concat(picked).sort_values("session_id").reset_index(drop=True)
    summary["kuota"] = quota_info

    crop = S.groupby("kelas_sesi").agg(baris_sebelum_crop=("n_asli", "sum"),
                                       baris_sesudah_crop=("panjang_blok", "sum")).reindex(classes)
    crop["baris_terbuang"] = crop.baris_sebelum_crop - crop.baris_sesudah_crop
    crop["persen_terbuang"] = (100 * crop.baris_terbuang / crop.baris_sebelum_crop).round(4)
    crop.index.name = "kelas"
    crop.to_csv(f"{RES}/crop_loss.csv")

    # ---------------- split kronologis per kelas
    S["partisi_awal"] = ""
    for c in classes:
        sub = S[S.kelas_sesi == c].sort_values(["t_mulai_asli", "session_id"])
        n = len(sub)
        ntr = int(np.floor(0.70 * n))
        nva = int(np.floor(0.85 * n)) - ntr
        lab_p = ["train"] * ntr + ["val"] * nva + ["test"] * (n - ntr - nva)
        S.loc[sub.index, "partisi_awal"] = lab_p
    S["n_window"] = S.panjang_blok - (W - 1)

    # ---------------- purging (val|test dulu, lalu train|val)
    S["partisi"] = S["partisi_awal"]
    purge_rows = []
    for c in classes:
        m = S.kelas_sesi == c
        for early, late, name in (("val", "test", "val_test"), ("train", "val", "train_val")):
            late_m = m & (S.partisi == late)
            if late == "val" and not late_m.any():
                late_m = m & (S.partisi == "test")
            t_first = S.loc[late_m, "t_mulai_asli"].min() if late_m.any() else np.inf
            drop = m & (S.partisi == early) & (S.t_akhir_asli > t_first)
            purge_rows.append(dict(kelas=c, batas=name, sesi_dibuang=int(drop.sum()),
                                   baris_dibuang=int(S.loc[drop, "panjang_blok"].sum()),
                                   window_dibuang=int(S.loc[drop, "n_window"].sum()),
                                   t_mulai_pertama_partisi_berikut=t_first))
            S.loc[drop, "partisi"] = "purged"
    pd.DataFrame(purge_rows).to_csv(f"{RES}/purging_log.csv", index=False)

    # ---------------- Tabel 5
    pre = S.pivot_table(index="kelas_sesi", columns="partisi_awal", values="session_id",
                        aggfunc="count", fill_value=0).reindex(index=classes, columns=PARTS, fill_value=0)
    post = S[S.partisi != "purged"].pivot_table(index="kelas_sesi", columns="partisi", values="session_id",
                                                aggfunc="count", fill_value=0).reindex(index=classes, columns=PARTS, fill_value=0)
    postw = S[S.partisi != "purged"].pivot_table(index="kelas_sesi", columns="partisi", values="n_window",
                                                 aggfunc="sum", fill_value=0).reindex(index=classes, columns=PARTS, fill_value=0)
    t5 = pd.DataFrame(dict(kelas=classes,
                           sesi_latih=pre["train"].values, sesi_validasi=pre["val"].values, sesi_uji=pre["test"].values,
                           sesi_latih_purged=post["train"].values, sesi_validasi_purged=post["val"].values,
                           sesi_uji_purged=post["test"].values,
                           window_latih_purged=postw["train"].values, window_validasi_purged=postw["val"].values,
                           window_uji_purged=postw["test"].values))
    t5.loc[len(t5)] = ["TOTAL"] + [int(t5[c].sum()) for c in t5.columns[1:]]
    t5.to_csv(f"{RES}/table5_partitions.csv", index=False)

    # ---------------- baris & window per blok
    blk_pos = np.concatenate([np.arange(b) for b in S.panjang_blok])
    blk_sorted = np.repeat(S.start_sorted.to_numpy() + S.offset_blok.to_numpy(), S.panjang_blok) + blk_pos
    R = pd.DataFrame(dict(
        session_id=np.repeat(S.session_id.to_numpy(), S.panjang_blok),
        pos=blk_pos, rowid=order[blk_sorted], ts=ts_s[blk_sorted], lab=lab_s[blk_sorted],
        partisi=np.repeat(S.partisi.to_numpy(), S.panjang_blok),
    ))
    # blok campuran
    blk_mixed = R.groupby("session_id", sort=True)["lab"].nunique() > 1
    S["blok_campuran"] = blk_mixed.reindex(S.session_id).to_numpy()

    is_end = R.pos.to_numpy() >= (W - 1)
    Wn = pd.DataFrame(dict(
        session_id=R.session_id[is_end].to_numpy(), posisi_akhir_window=R.pos[is_end].to_numpy(),
        label_window_idx=R.lab[is_end].to_numpy(), ts_akhir_window=R.ts[is_end].to_numpy(),
        ts_awal_window=R.ts.to_numpy()[np.flatnonzero(is_end) - (W - 1)],
        rowid_akhir=R.rowid[is_end].to_numpy(), partisi=R.partisi[is_end].to_numpy(),
    ))
    Wn.insert(0, "window_id", np.arange(len(Wn)))
    Wn = Wn.merge(S[["session_id", "kelas_sesi", "t_mulai_asli", "t_akhir_asli"]], on="session_id", how="left")
    Wn["label_window"] = [classes[i] for i in Wn.label_window_idx]
    assert len(Wn) == int(S.n_window.sum())

    Wf = Wn[Wn.partisi != "purged"]
    Sf = S[S.partisi != "purged"]

    # ---------------- mixed_label
    ml = pd.DataFrame(dict(kelas=classes))
    ml["n_sesi"] = Sf.groupby("kelas_sesi").size().reindex(classes, fill_value=0).values
    ml["n_sesi_campuran"] = Sf.groupby("kelas_sesi").campuran.sum().reindex(classes, fill_value=0).values.astype(int)
    ml["persen_sesi_campuran"] = (100 * ml.n_sesi_campuran / ml.n_sesi.replace(0, np.nan)).round(4)
    ml["n_blok_campuran"] = Sf.groupby("kelas_sesi").blok_campuran.sum().reindex(classes, fill_value=0).values.astype(int)
    ml["n_window"] = Wf.groupby("kelas_sesi").size().reindex(classes, fill_value=0).values
    diff = Wf[Wf.label_window != Wf.kelas_sesi]
    ml["n_window_label_beda_dari_sesi"] = diff.groupby("kelas_sesi").size().reindex(classes, fill_value=0).values
    ml["persen_window_label_beda"] = (100 * ml.n_window_label_beda_dari_sesi / ml.n_window.replace(0, np.nan)).round(4)
    ml.to_csv(f"{RES}/mixed_label.csv", index=False)
    # versi sebelum kuota (seluruh sesi >=10) sebagai pelengkap
    allbig = M["cov"]["big"]
    ml_pre = pd.DataFrame(dict(
        kelas=classes,
        n_sesi_ge10=np.bincount(M["mode_all"][allbig], minlength=ncls),
        n_sesi_campuran=np.bincount(M["mode_all"][allbig & M["mixed_all"]], minlength=ncls)))
    ml_pre["persen_sesi_campuran"] = (100 * ml_pre.n_sesi_campuran / ml_pre.n_sesi_ge10.replace(0, np.nan)).round(4)
    ml_pre.to_csv(f"{RES}/mixed_label_sebelum_kuota.csv", index=False)

    # ---------------- tabel silang window
    xt = Wn.groupby(["partisi", "kelas_sesi", "label_window"]).size().rename("n_window").reset_index()
    xt = xt.rename(columns={"kelas_sesi": "label_sesi"})
    xt.to_csv(f"{RES}/window_label_by_partition.csv", index=False)

    # ---------------- T3 audit
    t3 = []
    for c in classes:
        sc = Sf[Sf.kelas_sesi == c]
        for early, late, name in (("train", "val", "train_val"), ("val", "test", "val_test")):
            e, l_ = sc[sc.partisi == early], sc[sc.partisi == late]
            a = e.t_akhir_asli.max() if len(e) else np.nan
            b = l_.t_mulai_asli.min() if len(l_) else np.nan
            ok = bool(a <= b) if len(e) and len(l_) else True
            nviol = int(e.loc[e.t_akhir_asli > b, "n_window"].sum()) if len(l_) else 0
            t3.append(dict(tingkat="label_sesi", kelas_atau_label=c, batas=name,
                           akhir_terlambat_partisi_awal=a, mulai_terawal_partisi_berikut=b,
                           lolos=ok, n_window_melanggar=nviol,
                           akhir_terlambat_window_ts=np.nan, mulai_terawal_window_ts=np.nan,
                           lolos_window_ts=np.nan, n_window_melanggar_window_ts=np.nan))
    for c in classes:
        wc = Wf[Wf.label_window == c]
        for early, late, name in (("train", "val", "train_val"), ("val", "test", "val_test")):
            e, l_ = wc[wc.partisi == early], wc[wc.partisi == late]
            if len(e) and len(l_):
                a, b = e.t_akhir_asli.max(), l_.t_mulai_asli.min()
                aw, bw = e.ts_akhir_window.max(), l_.ts_awal_window.min()
                row = dict(akhir_terlambat_partisi_awal=a, mulai_terawal_partisi_berikut=b, lolos=bool(a <= b),
                           n_window_melanggar=int((e.t_akhir_asli > b).sum()),
                           akhir_terlambat_window_ts=aw, mulai_terawal_window_ts=bw, lolos_window_ts=bool(aw <= bw),
                           n_window_melanggar_window_ts=int((e.ts_akhir_window > bw).sum()))
            else:
                row = dict(akhir_terlambat_partisi_awal=e.t_akhir_asli.max() if len(e) else np.nan,
                           mulai_terawal_partisi_berikut=l_.t_mulai_asli.min() if len(l_) else np.nan,
                           lolos=True, n_window_melanggar=0, akhir_terlambat_window_ts=np.nan,
                           mulai_terawal_window_ts=np.nan, lolos_window_ts=np.nan,
                           n_window_melanggar_window_ts=np.nan)
            t3.append(dict(tingkat="label_window", kelas_atau_label=c, batas=name, **row))
    t3 = pd.DataFrame(t3)
    t3.to_csv(f"{RES}/t3_audit.csv", index=False)

    # ---------------- pemilihan 450 utama + 40 pilot dari partisi uji
    rng_sel = np.random.default_rng(SEED)
    Wt = Wf[Wf.partisi == "test"]

    def round_robin(cand, k, rng):
        """Ambil k window, disebar ke sebanyak mungkin sesi berbeda."""
        groups = {s: g.window_id.to_numpy() for s, g in cand.groupby("session_id")}
        sess = np.array(sorted(groups))
        rng.shuffle(sess)
        pools = {s: list(rng.permutation(groups[s])) for s in sess}
        out = []
        while len(out) < k and any(pools.values()):
            for s in sess:
                if pools[s] and len(out) < k:
                    out.append(pools[s].pop())
        return out

    main_ids, pilot_ids, sel_info = [], [], {}
    pilot_extra = set(rng_sel.choice(classes, size=min(ncls, 40 - 4 * ncls), replace=False)) if 40 - 4 * ncls > 0 else set()
    for c in classes:
        cand = Wt[Wt.label_window == c]
        mids = round_robin(cand, 50, rng_sel)
        main_ids += mids
        msel = cand[cand.window_id.isin(mids)]
        # pilot: tidak boleh sama dan tidak boleh berbagi time-step dgn window utama
        rest = cand[~cand.window_id.isin(mids)]
        occ = msel.groupby("session_id").posisi_akhir_window.apply(np.array).to_dict()
        ok = np.array([s not in occ or np.min(np.abs(occ[s] - p)) >= W
                       for s, p in zip(rest.session_id, rest.posisi_akhir_window)], dtype=bool)
        rest = rest[ok] if len(rest) else rest
        kp = 5 if c in pilot_extra else 4
        fresh = rest[~rest.session_id.isin(occ)]
        pids = round_robin(fresh, kp, rng_sel)
        if len(pids) < kp:
            pids += round_robin(rest[~rest.window_id.isin(pids)], kp - len(pids), rng_sel)
        pilot_ids += pids
        sel_info[c] = dict(window_uji_tersedia=int(len(cand)), sesi_uji_tersedia=int(cand.session_id.nunique()),
                           utama_terpilih=len(mids), utama_sesi_berbeda=int(msel.session_id.nunique()),
                           pilot_terpilih=len(pids),
                           pilot_sesi_berbeda=int(cand[cand.window_id.isin(pids)].session_id.nunique()))
    cols_idx = ["window_id", "session_id", "label_window", "posisi_akhir_window", "rowid_akhir"]
    main_df = Wn.set_index("window_id").loc[main_ids].reset_index()[cols_idx].rename(columns={"label_window": "kelas_window"})
    pilot_df = Wn.set_index("window_id").loc[pilot_ids].reset_index()[cols_idx].rename(columns={"label_window": "kelas_window"})
    assert not set(main_ids) & set(pilot_ids)
    main_df.to_csv(f"{ART}/main_450_index.csv", index=False)
    pilot_df.to_csv(f"{ART}/pilot_40_index.csv", index=False)
    summary["seleksi_instance"] = sel_info
    summary["pilot_kelas_dapat_5"] = sorted(pilot_extra)

    # ---------------- session index
    si = S[["session_id", "kelas_sesi", "partisi", "t_mulai_asli", "t_akhir_asli", "offset_blok",
            "panjang_blok", "campuran", "partisi_awal", "n_asli", "K_c", "blok_campuran", "n_window"]]
    si.to_csv(f"{ART}/session_index.csv", index=False)
    si.to_pickle(f"{ART}/session_index.pkl")

    # ---------------- PASS 2: ambil fitur baris terpilih, fit encoder & scaler di latih
    log("PASS 2: ambil fitur baris terpilih")
    sel_mask = np.zeros(P["n3"], dtype=bool)
    sel_mask[R.rowid.to_numpy()] = True
    kept = P["kept"]
    parts_df, g0 = [], 0
    for fi, chunk in raw_chunks():
        _, c3 = clean_chunk(chunk)
        if c3.empty:
            continue
        m = sel_mask[g0:g0 + len(c3)]
        if m.any():
            sub = c3.loc[m, kept].copy()
            sub["rowid"] = np.arange(g0, g0 + len(c3))[m]
            sub["file_idx"] = fi
            sub["raw_row"] = c3.index.to_numpy()[m]
            parts_df.append(sub)
        g0 += len(c3)
    assert g0 == P["n3"]
    D = pd.concat(parts_df, ignore_index=True)
    assert len(D) == len(R)
    D = D.merge(R[["rowid", "session_id", "pos", "partisi"]], on="rowid", how="left")
    assert (D.type.map({c: i for i, c in enumerate(classes)}).to_numpy() ==
            R.set_index("rowid").loc[D.rowid, "lab"].to_numpy()).all()

    feats = [c for c in kept if c not in NON_FEATURE]
    cat_cols = [c for c in feats if P["is_cat"][c]]
    for c in cat_cols:
        D[c] = D[c].astype(str).replace("-", "none")
    tr = D.partisi == "train"
    encoders, codes = {}, {}
    for c in cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(D.loc[tr, [c]])
        encoders[c] = enc
        codes[c] = {str(v): i for i, v in enumerate(enc.categories_[0])}
        D[c] = enc.transform(D[[c]]).ravel()
    X = D[feats].astype("float64").to_numpy()
    scaler = MinMaxScaler().fit(X[tr.to_numpy()])
    with open(f"{ART}/encoder.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(f"{ART}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{ART}/categorical_codes.json", "w") as f:
        json.dump({"feature_order": feats, "categorical_columns": cat_cols, "unknown_code": -1,
                   "codes": codes}, f, indent=1)
    D[["rowid", "file_idx", "raw_row", "session_id", "pos", "partisi"]].sort_values(
        ["session_id", "pos"]).to_csv(f"{ART}/block_rows.csv.gz", index=False)
    summary["fitur"] = {"n_features": len(feats), "feature_order": feats, "kategorikal": cat_cols,
                        "scaler_fit_pada_baris_latih": int(tr.sum())}

    # ---------------- summary
    t3_ok = t3.groupby("tingkat").lolos.all().to_dict()
    over5 = ml[ml.persen_window_label_beda > 5].kelas.tolist()
    summary.update({
        "baris_dalam_sesi_ge10": int(cov["rows_cov"].sum()),
        "sesi_ge10": int(cov["big"].sum()),
        "baris_sesudah_kuota_crop": int(S.panjang_blok.sum()),
        "baris_sesudah_purging": int(Sf.panjang_blok.sum()),
        "window_sesudah_purging": {p: int(Wf[Wf.partisi == p].shape[0]) for p in PARTS},
        "sesi_uji_backdoor_sesudah_purging": int(((Sf.kelas_sesi == "backdoor") & (Sf.partisi == "test")).sum()),
        "persen_sesi_campuran_tertinggi": {"kelas": ml.loc[ml.persen_sesi_campuran.idxmax(), "kelas"],
                                           "persen": float(ml.persen_sesi_campuran.max())},
        "ada_kelas_window_label_beda_gt5": bool(over5), "kelas_window_label_beda_gt5": over5,
        "t3_lolos_semua": {k: bool(v) for k, v in t3_ok.items()},
        "definisi": __doc__.strip(),
        "catatan": "session_index disimpan CSV+pickle (pyarrow tidak terpasang; aturan larang install).",
        "versi": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                  "sklearn": sklearn.__version__},
        "waktu_eksekusi_detik": round(time.time() - T0, 1),
    })
    with open(f"{RES}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("SELESAI")


if __name__ == "__main__":
    main()
