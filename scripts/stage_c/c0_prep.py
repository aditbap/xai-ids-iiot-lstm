"""
C0: tulis config_frozen.json (sekali, tidak pernah ditimpa) dan siapkan
baris prapemrosesan final untuk seluruh Tahap C.

- rows_main: identik dengan b2_0_tensors_log (diverifikasi: window w=10 dari
  baris ini harus sama persis dengan artifacts/stage_b2/X_{train,val,test}.npy).
- rows_srcport: src_port Min-Max TANPA log, scaler di-fit pada baris latih
  (model kontrol). 22 fitur lain pada model kontrol = ruang model utama.
- Nilai pengganti src_port untuk kontrol shortcut: median latih (satuan asli,
  dihitung atas window latih x time-step), lalu Min-Max.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from pipeline_common import CHUNKSIZE, sorted_network_files  # noqa: E402
from common_c import (A, B2, C_ART, C_RES, DL, LOG_FEATS, ROWS, run_log, save_json, sha256,  # noqa: E402
                      windows)

CONFIG = f"{C_RES}/config_frozen.json"


def write_config():
    if os.path.exists(CONFIG):
        run_log("config_frozen.json sudah ada, tidak ditimpa")
        return
    cfg = dict(
        dibekukan_pada=pd.Timestamp.now().isoformat(),
        naskah="Paper_RABIT_Benediktus_rev8 (rev10 tidak tersedia di mesin; Tabel 6-8 dibaca dari rev8)",
        prapemrosesan=dict(kategorikal="kode bilangan bulat encoder Tahap A", log1p=LOG_FEATS,
                           minmax="artifacts/diag_log/scaler_log.pkl", src_port_kontrol="Min-Max tanpa log, fit latih"),
        partisi_indeks="artifacts/stage_a/",
        window=10, epoch=12, batch=64, optimizer="Adam lr 0.001 (model utama & kontrol)",
        loss="categorical_crossentropy", class_weight="balanced dari y latih",
        pemilihan_epoch="F1-macro validasi terbaik",
        seed=dict(utama=[42, 43, 44], t6=[42, 43, 44, 45, 46], kontrol=[42, 43, 44], ablasi=[42, 43, 44],
                  pembanding_final=[42, 43, 44], random_search=42, xai=42, bootstrap=42,
                  m3_prosedural=[101, 102, 103, 104, 105]),
        model_utama_seed42=f"{B2}/lstm_main_seed42.h5",
        t6=dict(kriteria="uji permutasi berpasangan eksak satu sisi (32 tanda) p <= 0.05 dan rerata dF1 >= 0.01",
                acak="urutan baris di dalam sesi diacak (seed = seed model) pada latih & validasi sebelum window"),
        ablasi=dict(w=[1, 5, 10, 20], subset="window yang sesinya (blok) >= 20 baris DAN posisi akhir >= 19 "
                    "(target time-step identik untuk semua w)"),
        pembanding=dict(
            random_search="20 konfigurasi per model, rng numpy default_rng(42), seed latih 42; terpilih = F1-macro val "
                          "tertinggi, lalu dilatih ulang seed 42, 43, 44",
            rf=dict(n_estimators=[50, 500], max_depth=[5, 50], min_samples_leaf=[1, 20], lain="class_weight balanced, n_jobs 4"),
            xgb=dict(n_estimators=[50, 500], max_depth=[3, 15], learning_rate="uniform [0.01, 0.3]",
                     subsample=[0.5, 1.0], lain="tree_method hist, n_jobs 4, sample_weight balanced"),
            mlp=dict(lapis=[1, 3], unit=[32, 256], dropout=[0.1, 0.5], learning_rate="log-uniform [1e-4, 1e-2]",
                     arsitektur="(Dense unit ReLU -> Dropout) x lapis -> Dense 9 softmax; masukan 220"),
            lstm1=dict(unit=[32, 128], dropout=[0.1, 0.5], learning_rate="log-uniform [1e-4, 1e-2]",
                       arsitektur="LSTM unit -> Dropout -> Dense 64 ReLU -> Dense 9 softmax (model utama tanpa lapis LSTM kedua)"),
            protokol_saraf="12 epoch, batch 64, Adam, categorical CE, class_weight balanced, epoch terbaik F1-macro val"),
        mode_pelatihan=("Pelatihan Keras dijalankan dalam mode eager (tanpa disable_v2_behavior), satu model per proses, "
                        "identik dengan skrip yang menghasilkan model seed 42 (scripts/diag_log/dl1_prep_train.py), "
                        "agar protokol antar-seed sama. Semua XAI dan evaluasi memakai urutan impor wajib (graph mode)."),
        threads=dict(intra=4, inter=4, omp=4), deterministic_ops=True,
        xai=dict(rincian="results/stage_b2/pilot_log.json -> keputusan_implementasi (berlaku tanpa perubahan)",
                 n_instance=450, indeks="artifacts/stage_a/main_450_index.csv", bg_primer="empiris 100 proporsional, seed 42",
                 lime_n=5000, ks_fitur=[0, 1, 3, 5, 10, 15], ks_time=[0, 1, 2, 3, 5], acak_ulangan=5,
                 m3=dict(R=5, k=10, k_per_timestep=3, sigma=0.01, sigma_kepekaan=[0.005, 0.02]),
                 m4=dict(instance="45 (5 pertama per kelas dari indeks 450)", pemanasan=5, ulangan=10, batch=1, thread=4),
                 m2_dAUC="AUC(acak, rerata 5 ulangan) - AUC(metode); positif = lebih setia",
                 bootstrap=dict(ulangan=1000, unit="sesi di dalam kelas, bobot r", primer_tingkat=98.75,
                                eksploratori_tingkat=95),
                 kepekaan=["bg seimbang 100", "bg empiris 50", "bg empiris 200", "pengganti empiris 10 ulangan",
                           "sigma 0.005", "sigma 0.02", "kontrol tanpa fitur >=99% nilai sama"]),
        hash_sha256={p: sha256(p) for p in [f"{DL}/scaler_log.pkl", f"{A}/encoder.pkl",
                                            f"{B2}/lstm_main_seed42.h5", f"{A}/main_450_index.csv",
                                            f"{A}/block_rows.csv.gz", f"{A}/session_index.pkl"]},
    )
    save_json(cfg, CONFIG)
    run_log("config_frozen.json ditulis")


def prep_rows():
    if os.path.exists(f"{ROWS}/rows_meta.csv"):
        run_log("rows sudah ada, dilewati")
        return
    FJ = json.load(open(f"{B2}/features.json"))
    classes, feats = FJ["classes"], FJ["features"]
    cat_cols = FJ["categorical_columns"]
    log_idx = [feats.index(f) for f in LOG_FEATS]
    encoders = pickle.load(open(f"{A}/encoder.pkl", "rb"))
    scaler = pickle.load(open(f"{DL}/scaler_log.pkl", "rb"))
    keys = ["src_ip", "dst_ip", "dst_port", "proto"]

    br = pd.read_csv(f"{A}/block_rows.csv.gz")
    want = {fi: set(g.raw_row) for fi, g in br.groupby("file_idx")}
    parts = []
    for fi, path in enumerate(sorted_network_files()):
        if fi not in want:
            continue
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype={"src_bytes": "string"}, low_memory=False):
            m = chunk.index.isin(want[fi])
            if m.any():
                cols = ["src_port"] + feats + ["type", "src_ip", "dst_ip", "dst_port"]
                sub = chunk.loc[m, list(dict.fromkeys(cols))].copy()
                sub["file_idx"], sub["raw_row"] = fi, sub.index.to_numpy()
                parts.append(sub)
    D = pd.concat(parts, ignore_index=True)
    assert len(D) == len(br)
    D = br.merge(D, on=["file_idx", "raw_row"], how="left", validate="one_to_one")
    D = D.sort_values(["session_id", "pos"]).reset_index(drop=True)
    for c in cat_cols:
        D[c] = encoders[c].transform(D[[c]].astype(str).replace("-", "none")).ravel()
    Xr = D[feats].astype("float64").to_numpy()
    Xr[:, log_idx] = np.log1p(Xr[:, log_idx])
    Xm = scaler.transform(Xr).astype(np.float32)

    tr = (D.partisi == "train").to_numpy()
    sp_raw = D[["src_port"]].astype("float64").to_numpy()
    sc_sp = MinMaxScaler().fit(sp_raw[tr])
    pickle.dump(sc_sp, open(f"{C_ART}/scaler_srcport.pkl", "wb"))
    sp = sc_sp.transform(sp_raw).astype(np.float32).ravel()

    lab = D["type"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int8)
    # proto disimpan sebagai kode encoder (bijektif dgn nilai asli, cukup untuk kunci sesi)
    meta = pd.DataFrame(dict(session_id=D.session_id, pos=D.pos, partisi=D.partisi, label=lab,
                             src_ip=D.src_ip, dst_ip=D.dst_ip, dst_port=D.dst_port, proto=D.proto.astype(int)))

    sid, pos, part = meta.session_id.to_numpy(), meta.pos.to_numpy(), meta.partisi.to_numpy()
    ok = {}
    for p in ["train", "val", "test"]:
        Xw, yw, end = windows(Xm, lab, sid, pos, part, p, 10)
        ok[p] = bool(np.array_equal(Xw, np.load(f"{B2}/X_{p}.npy")) and np.array_equal(yw, np.load(f"{B2}/y_{p}.npy")))
        if p == "train":
            spw = sp_raw[:, 0][end[:, None] + np.arange(-9, 1)].ravel()     # window latih x time-step
            med_raw = float(np.median(spw))
    assert all(ok.values()), ok
    np.save(f"{ROWS}/rows_main.npy", Xm)
    np.save(f"{ROWS}/rows_srcport.npy", sp)
    meta.to_csv(f"{ROWS}/rows_meta.csv", index=False)
    save_json(dict(verifikasi_identik_stage_b2=ok, src_port_median_asli=med_raw,
                   src_port_median_ruang_model=float(sc_sp.transform([[med_raw]])[0, 0]),
                   n_baris=len(meta)), f"{ROWS}/rows_info.json")
    run_log(f"rows disiapkan: {len(meta):,} baris; identik dgn stage_b2: {ok}")


if __name__ == "__main__":
    write_config()
    prep_rows()
