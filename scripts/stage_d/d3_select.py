"""
D3.1, D3.3, D3.5 (tanpa TensorFlow).

D3.1 Partisi uji final = window uji yang sesinya TIDAK termasuk sesi asal 40
     instance pilot (artifacts/stage_a/pilot_40_index.csv).
D3.3 Pilih ulang 450 instance dari partisi uji final dengan prosedur Tahap A.
     Fungsi round_robin dan urutan langkah DISALIN dari scripts/stage_a/stage_a.py
     (baris 471-512) tanpa perubahan: rng default_rng(42); pilot_extra diambil
     dulu; per kelas (urutan alfabetis) 50 window utama lalu pilot (mengonsumsi
     rng yang sama). Hasil pilot dibuang (tidak dipakai). Prosedur diverifikasi
     lebih dulu: pada partisi uji PENUH harus mereproduksi main_450_index.csv
     dan pilot_40_index.csv Tahap A persis.
     Tabel window (window_id, rowid_akhir) direkonstruksi dari rows_meta +
     block_rows (window_id = urutan window pos>=9 atas baris terurut
     (session_id, pos), sama dengan Wn Tahap A; diverifikasi pada indeks lama).
D3.5 Hitung instance dengan kode kategori -1 pada 450 instance baru.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common_d import A, CLASSES, D_ART, D_RES, load_meta, pilot_sessions, run_log  # noqa: E402

SEED, W = 42, 10


def window_table():
    meta = load_meta()
    br = pd.read_csv(f"{A}/block_rows.csv.gz").sort_values(["session_id", "pos"]).reset_index(drop=True)
    assert (br.session_id.to_numpy() == meta.session_id.to_numpy()).all()
    end = np.flatnonzero(meta.pos.to_numpy() >= W - 1)
    Wn = pd.DataFrame(dict(window_id=np.arange(len(end)), session_id=meta.session_id.to_numpy()[end],
                           posisi_akhir_window=meta.pos.to_numpy()[end], rowid_akhir=br.rowid.to_numpy()[end],
                           partisi=meta.partisi.to_numpy()[end],
                           label_window=[CLASSES[i] for i in meta.label.to_numpy()[end]], end_row=end))
    for f in ("main_450_index", "pilot_40_index"):
        ix = pd.read_csv(f"{A}/{f}.csv").merge(Wn, on="window_id", suffixes=("", "_r"))
        assert (ix.session_id == ix.session_id_r).all() and (ix.rowid_akhir == ix.rowid_akhir_r).all()
        assert (ix.kelas_window == ix.label_window).all()
    return Wn


def select(Wt, classes):
    """Salinan langkah pemilihan Tahap A (stage_a.py baris 471-512)."""
    ncls = len(classes)
    rng_sel = np.random.default_rng(SEED)

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
                           utama_terpilih=len(mids), utama_sesi_berbeda=int(msel.session_id.nunique()))
    return main_ids, pilot_ids, sel_info


def main():
    out_idx = f"{D_ART}/main_450_index_v2.csv"
    if os.path.exists(out_idx):
        return run_log("D3 seleksi sudah ada")
    Wn = window_table()
    cols_idx = ["window_id", "session_id", "label_window", "posisi_akhir_window", "rowid_akhir"]
    Wt = Wn[Wn.partisi == "test"]

    # verifikasi prosedur pada uji penuh
    m_old, p_old, _ = select(Wt, CLASSES)
    ok_m = list(pd.read_csv(f"{A}/main_450_index.csv").window_id) == list(m_old)
    ok_p = list(pd.read_csv(f"{A}/pilot_40_index.csv").window_id) == list(p_old)
    assert ok_m and ok_p, (ok_m, ok_p)
    run_log("D3 prosedur seleksi Tahap A tereproduksi persis pada uji penuh (450 & 40)")

    # D3.1 eksklusi
    P = pilot_sessions()
    si = pd.read_pickle(f"{A}/session_index.pkl").set_index("session_id")
    excl = Wt.session_id.isin(P).to_numpy()
    rows = []
    for c in CLASSES + ["SEMUA"]:
        mw = np.ones(len(Wt), bool) if c == "SEMUA" else (Wt.label_window == c).to_numpy()
        sess_c = Wt.session_id[mw].unique()
        sess_cls = si.loc[sess_c, "kelas_sesi"]
        rows.append(dict(kelas=c, sesi_uji_sebelum=len(sess_c), sesi_dikeluarkan=int(np.isin(sess_c, list(P)).sum()),
                         sesi_uji_final=int((~np.isin(sess_c, list(P))).sum()),
                         window_uji_sebelum=int(mw.sum()), window_dikeluarkan=int((mw & excl).sum()),
                         window_uji_final=int((mw & ~excl).sum()),
                         persen_window_dikeluarkan=100 * (mw & excl).sum() / max(mw.sum(), 1),
                         catatan="kelas = label window; sesi dihitung bila punya window berlabel kelas ini"
                         if c != "SEMUA" else f"{len(P)} sesi pilot; kelas_sesi: "
                         + ", ".join(f"{k}={v}" for k, v in si.loc[sorted(P), 'kelas_sesi'].value_counts().sort_index().items())))
        del sess_cls
    pd.DataFrame(rows).to_csv(f"{D_RES}/d3_pilot_exclusion.csv", index=False)
    np.savez_compressed(f"{D_ART}/test_final_mask.npz", end_row=Wt.end_row.to_numpy(), final=~excl,
                        pilot_sessions=np.array(sorted(P)))

    # D3.3 seleksi baru
    Wf = Wt[~excl]
    m_new, _, info = select(Wf, CLASSES)
    new = Wn.set_index("window_id").loc[m_new].reset_index()[cols_idx].rename(columns={"label_window": "kelas_window"})
    assert not new.session_id.isin(P).any() and len(new) == 450
    new.to_csv(out_idx, index=False)
    old = pd.read_csv(f"{A}/main_450_index.csv")
    rows = []
    for c in CLASSES + ["SEMUA"]:
        a = new if c == "SEMUA" else new[new.kelas_window == c]
        b = old if c == "SEMUA" else old[old.kelas_window == c]
        rows.append(dict(kelas=c, n_baru=len(a), sesi_berbeda_baru=a.session_id.nunique(),
                         sesi_berbeda_lama=b.session_id.nunique(),
                         sesi_uji_final_tersedia=info[c]["sesi_uji_tersedia"] if c != "SEMUA" else int(Wf.session_id.nunique()),
                         window_sama_dgn_lama=int(a.window_id.isin(b.window_id).sum()),
                         sesi_sama_dgn_lama=int(len(set(a.session_id) & set(b.session_id)))))
    pd.DataFrame(rows).to_csv(f"{D_RES}/d3_selection_v2.csv", index=False)

    # D3.5 kode -1
    raw = np.load(f"{D_ART}/rows_raw.npy")
    import json
    FJ = json.load(open("artifacts/stage_b2/features.json"))
    cat_f, cat_c = FJ["categorical_index"], FJ["categorical_columns"]
    rows = []
    for nm, ix in (("450_baru", new), ("450_lama", old)):
        e = Wn.set_index("window_id").loc[ix.window_id, "end_row"].to_numpy()
        Xw = raw[e[:, None] + np.arange(-(W - 1), 1)][:, :, cat_f]
        neg = Xw == -1
        inst = neg.any(axis=(1, 2))
        rows.append(dict(indeks=nm, fitur="SEMUA", n_instance_dgn_kode_minus1=int(inst.sum()), n_sel=int(neg.sum()),
                         instance_posisi=";".join(map(str, np.flatnonzero(inst)))))
        for j, c in enumerate(cat_c):
            ij = neg[:, :, j].any(1)
            rows.append(dict(indeks=nm, fitur=c, n_instance_dgn_kode_minus1=int(ij.sum()), n_sel=int(neg[:, :, j].sum()),
                             instance_posisi=";".join(map(str, np.flatnonzero(ij)))))
    pd.DataFrame(rows).to_csv(f"{D_RES}/d3_unknown_codes.csv", index=False)
    run_log(f"D3 seleksi v2 selesai; dikeluarkan {excl.sum()} window uji dari {len(P)} sesi pilot")


if __name__ == "__main__":
    main()
