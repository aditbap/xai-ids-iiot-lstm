"""
Tahap B2 (ulang pilot dengan prapemrosesan final log1p): salinan
scripts/stage_b/b2_pilot_xai.py dengan perubahan:
- model = salinan artifacts/diag_log/lstm_main_seed42_log.h5 (epoch 8), tidak
  dilatih ulang; tensor & features.json dari scripts/stage_b2/b2_0_tensors_log.py
  (kode kategorikal -> log1p 10 fitur -> Min-Max scaler_log.pkl).
- LIME flat: dimensi numerik di ruang masukan model; pembungkus hanya
  mengubah dimensi kategorikal (kode -> Min-Max).
- Pengganti M2 dari data latih dalam satuan asli lalu diprapemrosesan
  (artifacts/stage_b2/deletion_baseline.csv).
- M3 varian per time-step memakai k = 3; varian lain k = 10.
- M4: 5 instance pemanasan dibuang, lalu 40 instance x 10 pengulangan, batch 1;
  statistik dihitung atas rerata per instance (n = 40). M4 dijalankan SEBELUM
  M3 (urutan saja, untuk menghindari pembesaran graf TF dari explainer M3).
- M3 perturbasi DeepSHAP memakai ulang explainer primer (background sama).

Tahap B: pilot XAI pada 40 instance (artifacts/stage_a/pilot_40_index.csv).
Tujuan: memeriksa jalur pipeline, BUKAN membandingkan metode.

Metode: DeepSHAP; LIME flat (LimeTabularExplainer, 220 dim); LIME kustom
(Algoritma 1) varian flat (m=220), per fitur (m=22), per time-step (m=10).

Keputusan implementasi yang tidak dirinci rancangan (dicatat juga di log):
- Background "empiris": 100 window latih, alokasi proporsional per kelas
  (largest remainder), acak tanpa pengembalian di dalam kelas.
- Background seimbang: 100 dibagi rata ke 9 kelas (11 per kelas, sisa 1
  ke satu kelas acak).
- LIME flat: random_state explainer di-reset ke RandomState(seed) sebelum
  tiap instance, supaya penjelasan tidak bergantung pada urutan instance.
- Importance global I_f = rerata atas instance dari I_{i,f} yang dinormalisasi
  (jumlah = 1 per instance).
- Top-k: urut menurun, seri dipecah oleh indeks fitur terkecil (sort stabil).
- M2 "biner" = fitur numerik dengan <= 2 nilai unik di data latih.
- M3: Jaccard@k berpasangan (10 pasang dari R=5) di tingkat I_{i,f} (22
  fitur, k=10); varian per time-step di tingkat I_{i,t} (10 langkah, k=3;
  Spearman berpasangan dilaporkan juga). Pada M3 perturbasi, kelas yang
  dijelaskan tetap c_i dari instance asli.
- M4: lihat bagian perubahan B2 di atas.
- Data UJI: performa (F1) kohort pilot disimpan ke CSV tanpa dicetak; F1
  sistem disegel di _sealed_system_check.npz.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import shap  # noqa: E402

import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402

import lime  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

tf.config.set_visible_devices([], "GPU")
_sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, device_count={"GPU": 0}))
tf.compat.v1.keras.backend.set_session(_sess)

A, B, R = "artifacts/stage_a", "artifacts/stage_b2", "results/stage_b2"
SEED = 42
N_LIME = int(os.environ.get("STAGE_B_N_LIME", "5000"))      # uji asap boleh lebih kecil; run resmi 5000
N_INST = int(os.environ.get("STAGE_B_N_INST", "40"))        # uji asap boleh lebih kecil; run resmi 40
OUT_SUFFIX = os.environ.get("STAGE_B_SUFFIX", "")           # uji asap -> berkas terpisah
N_BG = 100
KS_F = [0, 1, 3, 5, 10, 15]
KS_T = [0, 1, 2, 3, 5]
REP = 5
SEEDS_PROC = [101, 102, 103, 104, 105]
N_BOOT = 1000
N_CLS = 9
T0 = time.time()
STEP_T = {}
CHECKS = []


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def check(name, ok, detail):
    CHECKS.append(dict(nama_cek=name, lolos=bool(ok), detail=detail))
    log(f"CEK {name}: {'LOLOS' if ok else 'GAGAL'} | {detail}")


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


class Step:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log(f"--- {self.name}")

    def __exit__(self, *a):
        STEP_T[self.name] = round(time.time() - self.t, 2)


# ------------------------------------------------------------------ data
FJ = json.load(open(f"{B}/features.json"))
CLASSES, FEATS = FJ["classes"], FJ["features"]
NF, T = len(FEATS), 10
CAT_F = FJ["categorical_index"]                            # 8 fitur kategorikal (indeks dlm 22)
SC_SCALE, SC_MIN = np.array(FJ["scale_"]), np.array(FJ["min_"])
CAT_D = np.array([t * NF + f for t in range(T) for f in CAT_F])   # 80 dimensi flat
CAT_D_F = np.array([f for t in range(T) for f in CAT_F])          # fitur asal tiap dimensi

X_tr = np.load(f"{B}/X_train.npy")
y_tr = np.load(f"{B}/y_train.npy").astype(int)
X_te = np.load(f"{B}/X_test.npy", mmap_mode="r")
y_te = np.load(f"{B}/y_test.npy").astype(int)
meta_te = pd.read_csv(f"{B}/meta_test.csv")
pilot = pd.read_csv(f"{A}/pilot_40_index.csv").head(N_INST)
pm = pilot.merge(meta_te.reset_index(), on=["session_id", "posisi_akhir_window"], how="left")
assert pm["index"].notna().all() and (pm.kelas_window_x == pm.kelas_window_y).all()
IDX = pm["index"].to_numpy()
XP = np.asarray(X_te[IDX], dtype=np.float32)
YP = np.array([CLASSES.index(c) for c in pm.kelas_window_x])
SESS_P = pm.session_id.to_numpy()
NI = len(XP)

tf.compat.v1.set_random_seed(SEED)   # wajib bila TF_DETERMINISTIC_OPS=1
model = tf.keras.models.load_model(f"{B}/lstm_main_seed42.h5", compile=False)


def predict(X):
    return model.predict(np.asarray(X, dtype=np.float32), batch_size=2048, verbose=0)


P0 = predict(XP)
C = P0.argmax(1)                                           # kelas yang dijelaskan (c_i)
NAN_POOL = {}


def keep(name, arr):
    NAN_POOL[name] = np.asarray(arr, dtype=float)
    return arr


# ------------------------------------------------------------------ util
def topk(v, k):
    return set(np.argsort(-np.asarray(v), kind="stable")[:k])


def jacc(a, b, k):
    A_, B_ = topk(a, k), topk(b, k)
    return len(A_ & B_) / len(A_ | B_)


def spear(a, b):
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else np.nan


def agg_f(att):   # (n,10,22) -> (n,22)
    return np.abs(att).mean(axis=1)


def agg_t(att):   # (n,10,22) -> (n,10)
    return np.abs(att).mean(axis=2)


def norm_rows(M):
    s = M.sum(1, keepdims=True)
    return M / np.where(s == 0, 1, s)


def empirical_bg(seed, n=N_BG):
    rng = np.random.default_rng(seed)
    cnt = np.bincount(y_tr, minlength=N_CLS)
    quota = n * cnt / cnt.sum()
    alloc = np.floor(quota).astype(int)
    rem = n - alloc.sum()
    alloc[np.argsort(-(quota - alloc), kind="stable")[:rem]] += 1
    idx = np.concatenate([rng.choice(np.flatnonzero(y_tr == c), alloc[c], replace=False) for c in range(N_CLS)])
    return np.sort(idx)


def balanced_bg(seed, n=N_BG):
    rng = np.random.default_rng(seed)
    alloc = np.full(N_CLS, n // N_CLS)
    alloc[rng.choice(N_CLS, n - alloc.sum(), replace=False)] += 1
    idx = np.concatenate([rng.choice(np.flatnonzero(y_tr == c), alloc[c], replace=False) for c in range(N_CLS)])
    return np.sort(idx)


# ------------------------------------------------------------------ DeepSHAP
def deepshap_explain(bg_idx, X, classes_):
    t = time.time()
    ex = shap.DeepExplainer(model, X_tr[bg_idx])
    t_init = time.time() - t
    sv = np.array(ex.shap_values(X, check_additivity=False))     # (9, n, 10, 22)
    ev = np.array(ex.expected_value).ravel()
    att = sv[classes_, np.arange(len(X))]
    return ex, sv, ev, att, t_init


def additivity(sv, ev, P):
    recon = sv.reshape(sv.shape[0], sv.shape[1], -1).sum(-1).T + ev    # (n, 9)
    err = np.abs(recon - P)
    return err[np.arange(len(P)), C[:len(P)]], err.max(1)


# ------------------------------------------------------------------ LIME flat
def to_mixed(X):
    """(n,10,22) Min-Max -> (n,220) campuran: kategorikal = kode bilangan bulat asli."""
    Z = np.array(X, dtype=np.float64).reshape(len(X), -1)
    Z[:, CAT_D] = np.rint((Z[:, CAT_D] - SC_MIN[CAT_D_F]) / SC_SCALE[CAT_D_F])
    return Z


def from_mixed(Z):
    """(n,220) campuran -> (n,10,22) Min-Max dengan scaler latih."""
    Z = np.array(Z, dtype=np.float64, copy=True)
    Z[:, CAT_D] = Z[:, CAT_D] * SC_SCALE[CAT_D_F] + SC_MIN[CAT_D_F]
    return Z.reshape(len(Z), T, NF).astype(np.float32)


VALID_CODES = {f: set(range(len(FJ["codes"][FJ["categorical_columns"][j]])))
               for j, f in enumerate(CAT_F)}
LIME_REC = dict(active=False, invalid=0, total=0, cat_samples=[], first_rows=[])


def lime_wrapper(Z):
    if LIME_REC["active"]:
        cats = np.asarray(Z)[:, CAT_D]
        bad = 0
        for j, f in enumerate(CAT_D_F):
            bad += int((~np.isin(cats[:, j], list(VALID_CODES[f]))).sum())
        LIME_REC["invalid"] += bad
        LIME_REC["total"] += cats.size
        LIME_REC["cat_samples"].append(cats.astype(np.int16))
        LIME_REC["first_rows"].append(np.asarray(Z[:4], dtype=np.float32))
    return predict(from_mixed(Z))


def build_lime(seed):
    Ztr = to_mixed(X_tr)
    cat_names = {int(d): [str(k) for k, _ in sorted(FJ["codes"][FJ["categorical_columns"][CAT_F.index(f)]].items(),
                                                    key=lambda kv: kv[1])]
                 for d, f in zip(CAT_D, CAT_D_F)}
    names = [f"{FEATS[f]}_t{t + 1}" for t in range(T) for f in range(NF)]
    t = time.time()
    ex = LimeTabularExplainer(Ztr, mode="classification", feature_names=names,
                              categorical_features=CAT_D.tolist(), categorical_names=cat_names,
                              discretize_continuous=False, feature_selection="none",
                              random_state=seed)
    t_init = time.time() - t
    del Ztr
    return ex, t_init


def lime_explain(ex, x, c, seed):
    ex.random_state = np.random.RandomState(seed)
    e = ex.explain_instance(to_mixed(x[None])[0], lime_wrapper, labels=(int(c),),
                            num_features=T * NF, num_samples=N_LIME)
    w = np.zeros(T * NF)
    for d, v in e.local_exp[int(c)]:
        w[d] = v
    return w.reshape(T, NF)


# ------------------------------------------------------------------ LIME kustom (Algoritma 1)
GROUPS = {"kustom_flat": np.arange(T * NF), "per_fitur": np.tile(np.arange(NF), T),
          "per_timestep": np.repeat(np.arange(T), NF)}
M_OF = {"kustom_flat": T * NF, "per_fitur": NF, "per_timestep": T}


def custom_lime(x, c, variant, bg, seed, n=None):
    n = n or N_LIME
    m, g = M_OF[variant], GROUPS[variant]
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, size=(n, m))
    Z[0] = 1
    bsel = rng.integers(0, len(bg), size=n)
    xf = x.reshape(-1)
    Xs = np.where(Z[:, g] == 1, xf[None, :], bg.reshape(len(bg), -1)[bsel])
    p = predict(Xs.reshape(n, T, NF))[:, c]
    d = np.sqrt((1 - Z).sum(1))
    sigma = 0.75 * np.sqrt(m)
    pi = np.exp(-d ** 2 / sigma ** 2)
    beta = Ridge(alpha=1.0).fit(Z, p, sample_weight=pi).coef_
    out = np.abs(beta)
    return out.reshape(T, NF) if variant == "kustom_flat" else out


def deletion_curve(X, order_fn, ks, replace):
    """X (n,10,22); order_fn(i) -> urutan fitur/langkah; replace(x, sel, i) -> x baru.
    Mengembalikan p_i(k) untuk kelas c_i yang dikunci sejak k=0."""
    batch, keys = [], []
    for i in range(len(X)):
        o = order_fn(i)
        for k in ks:
            batch.append(replace(X[i].copy(), o[:k], i))
            keys.append((i, k))
    P = predict(np.stack(batch))
    out = np.zeros((len(X), len(ks)))
    preds = np.zeros((len(X), len(ks)), dtype=int)
    for (i, k), p in zip(keys, P):
        out[i, ks.index(k)] = p[C[i]]
        preds[i, ks.index(k)] = p.argmax()
    return out, preds


def auc(curves, ks):
    return np.trapz(curves, x=ks, axis=1)


# ================================================================== MAIN
def main():
    errors_fixed = json.load(open(f"{R}/galat_diperbaiki.json")) if os.path.exists(f"{R}/galat_diperbaiki.json") else []
    train_log = json.load(open("results/diag_log/train_log.json"))
    h_new = hashlib.sha256(open(f"{B}/lstm_main_seed42.h5", "rb").read()).hexdigest()
    h_src = hashlib.sha256(open("artifacts/diag_log/lstm_main_seed42_log.h5", "rb").read()).hexdigest()
    check("training_selesai", h_new == h_src and len(train_log["curve"]) == 12 and train_log["epoch_terpilih"] == 8,
          f"model = salinan diag_log (sha256 {h_new[:16]}..., identik: {h_new == h_src}); 12 epoch, epoch terpilih "
          f"{train_log['epoch_terpilih']}, val F1-macro {train_log['val_f1_macro_terpilih']:.4f}; tidak dilatih ulang")
    check("bentuk_masukan_10x22", tuple(model.input_shape[1:]) == (10, 22) and X_tr.shape[1:] == (10, 22),
          f"model input {model.input_shape}, X_train {X_tr.shape}")
    check("src_port_tidak_ada", "src_port" not in FEATS, f"{NF} fitur: src_port absen")

    att, IF, IT, timing, init_t = {}, {}, {}, {}, {}

    # ---------------- DeepSHAP primer + seimbang
    with Step("deepshap"):
        bg_idx = empirical_bg(SEED)
        np.save(f"{B}/bg_primary_idx{OUT_SUFFIX}.npy", bg_idx)
        ex_ds, sv, ev, a, init_t["deepshap"] = deepshap_explain(bg_idx, XP, C)
        att["deepshap"] = a
        e_pred, e_all = additivity(sv, ev, P0)
        keep("additivity_err", e_all)
        n_ok = int((e_pred < 1e-2).sum())
        check("additivity_deepshap", n_ok == NI,
              f"{n_ok}/{NI} lolos (toleransi 1e-2, prob softmax kelas c_i); galat maks c_i={e_pred.max():.3e}, "
              f"galat maks semua kelas={e_all.max():.3e}")
        bal_idx = balanced_bg(SEED)
        np.save(f"{B}/bg_balanced_idx{OUT_SUFFIX}.npy", bal_idx)
        _, svb, evb, ab, _ = deepshap_explain(bal_idx, XP, C)
        eb, _ = additivity(svb, evb, P0)
        keep("deepshap_balanced", ab)
        check("deepshap_bg_seimbang_jalan", ab.shape == (NI, T, NF) and np.isfinite(ab).all(),
              f"bentuk {ab.shape}; additivity lolos {(eb < 1e-2).sum()}/{NI}, galat maks {eb.max():.3e}")

    # ---------------- LIME flat
    with Step("lime_flat"):
        lime_ex, init_t["lime_flat"] = build_lime(SEED)
        # (b) pembungkus identik
        pw = lime_wrapper(to_mixed(XP))
        diff = np.abs(pw - P0).max()
        check("lime_pembungkus_identik", np.allclose(pw, P0, atol=1e-6), f"selisih maks {diff:.3e} (atol 1e-6)")
        sc_log = pickle.load(open("artifacts/diag_log/scaler_log.pkl", "rb"))
        fsib = FEATS.index("src_ip_bytes")
        pc = pd.read_csv("results/diag_log/precision_check.csv").set_index("src_ip_bytes")
        got = {v: np.log1p(v) * sc_log.scale_[fsib] + sc_log.min_[fsib] for v in (44, 52)}
        same_ref = all(np.isclose(got[v], pc.loc[v, "log1p_minmax_baru"], rtol=0, atol=1e-12) for v in got)
        same_fj = np.allclose(SC_SCALE, sc_log.scale_) and np.allclose(SC_MIN, sc_log.min_)
        check("prapemrosesan_log1p_konsisten", np.array_equal(pw, P0) and same_ref and same_fj,
              f"prediksi pembungkus vs langsung pada {NI} instance identik persis: {np.array_equal(pw, P0)}; "
              f"src_ip_bytes 44 -> {got[44]:.8f}, 52 -> {got[52]:.8f} sama dgn precision_check.csv: {same_ref}; "
              f"parameter scaler features.json = scaler_log.pkl: {same_fj}")
        inst_cats = to_mixed(XP)[:, CAT_D]
        n_unknown = int((inst_cats < 0).sum())
        LIME_REC.update(active=True)
        lw = []
        ex_store = {}
        for i in range(NI):
            n_before = len(LIME_REC["cat_samples"])
            lw.append(lime_explain(lime_ex, XP[i], C[i], SEED))
            cls = CLASSES[YP[i]]
            if len(ex_store.setdefault(cls, [])) < 3:
                ex_store[cls].append(LIME_REC["first_rows"][n_before][1])   # baris ke-1: sampel perturbasi pertama
        LIME_REC["active"] = False
        att["lime_flat"] = np.array(lw)
        check("lime_kode_kategori_valid", LIME_REC["invalid"] == 0,
              f"{LIME_REC['invalid']} nilai tidak valid dari {LIME_REC['total']:,} nilai kategorikal "
              f"({len(LIME_REC['cat_samples'])} batch); kode -1 (unknown) pada instance asli: {n_unknown}")
        np.savez_compressed(f"{B}/lime_perturbation_examples{OUT_SUFFIX}.npz",
                            **{c: np.array(v) for c, v in ex_store.items()},
                            note=np.array("baris (220,) representasi campuran; urutan dim t*22+f"))
        np.savez_compressed(f"{B}/lime_perturbation_cat_codes{OUT_SUFFIX}.npz",
                            cats=np.concatenate(LIME_REC["cat_samples"]))
        LIME_REC["cat_samples"].clear()
        LIME_REC["first_rows"].clear()

    # ---------------- LIME kustom
    bg = X_tr[bg_idx]
    with Step("lime_kustom"):
        for v in GROUPS:
            att[v] = np.array([custom_lime(XP[i], C[i], v, bg, SEED) for i in range(NI)])

    # ---------------- agregasi + bentuk
    for mth in ["deepshap", "lime_flat", "kustom_flat"]:
        IF[mth], IT[mth] = agg_f(att[mth]), agg_t(att[mth])
    IF["per_fitur"] = att["per_fitur"]
    IT["per_timestep"] = att["per_timestep"]
    shapes = {"deepshap": (NI, T, NF), "lime_flat": (NI, T, NF), "kustom_flat": (NI, T, NF),
              "per_fitur": (NI, NF), "per_timestep": (NI, T)}
    for mth, s in shapes.items():
        check(f"bentuk_keluaran_{mth}", att[mth].shape == s and np.isfinite(att[mth]).all(),
              f"{att[mth].shape} (diharapkan {s})")
    np.savez_compressed(f"{B}/attributions_pilot{OUT_SUFFIX}.npz", **{f"att_{k}": v for k, v in att.items()},
                        **{f"IF_{k}": v for k, v in IF.items()}, **{f"IT_{k}": v for k, v in IT.items()},
                        deepshap_balanced=NAN_POOL["deepshap_balanced"], expected_value=ev,
                        c_pred=C, y_true=YP, test_index=IDX, session_id=SESS_P.astype(str),
                        features=np.array(FEATS))
    for k, v in {**{f"IF_{k}": v for k, v in IF.items()}, **{f"IT_{k}": v for k, v in IT.items()}}.items():
        keep(k, v)

    # ---------------- M1
    with Step("m1"):
        rows = []
        for other in ["lime_flat", "kustom_flat"]:
            a_, b_ = IF["deepshap"], IF[other]
            ga, gb = norm_rows(a_).mean(0), norm_rows(b_).mean(0)
            for k in (5, 10, 15):
                per = [jacc(a_[i], b_[i], k) for i in range(NI)]
                rows.append(dict(pasangan=f"deepshap_vs_{other}", metrik=f"jaccard@{k}", global_=jacc(ga, gb, k),
                                 per_instance_mean=np.mean(per), per_instance_sd=np.std(per, ddof=1)))
            per = [spear(a_[i], b_[i]) for i in range(NI)]
            rows.append(dict(pasangan=f"deepshap_vs_{other}", metrik="spearman", global_=spear(ga, gb),
                             per_instance_mean=np.nanmean(per), per_instance_sd=np.nanstd(per, ddof=1),
                             n_nan_per_instance=int(np.isnan(per).sum())))
        m1 = pd.DataFrame(rows).rename(columns={"global_": "global"})
        m1.to_csv(f"{R}/m1_agreement{OUT_SUFFIX}.csv", index=False)
        keep("m1", m1[["global", "per_instance_mean", "per_instance_sd"]].to_numpy())

    # ---------------- baseline deletion (median / modus dari data latih, semua time-step)
    bdf = pd.read_csv(f"{B}/deletion_baseline.csv")
    assert list(bdf.fitur) == FEATS
    base = bdf.nilai_ruang_model.to_numpy()
    bdf.to_csv(f"{R}/deletion_baseline{OUT_SUFFIX}.csv", index=False)

    def rep_feat(x, sel, i):
        x[:, sel] = base[sel]
        return x

    def rep_time(x, sel, i):
        x[sel, :] = base[None, :]
        return x

    order_f = {m: (lambda i, m=m: np.argsort(-IF[m][i], kind="stable")) for m in ["deepshap", "lime_flat", "kustom_flat", "per_fitur"]}
    order_t = {m: (lambda i, m=m: np.argsort(-IT[m][i], kind="stable")) for m in ["deepshap", "lime_flat", "kustom_flat", "per_timestep"]}

    # ---------------- M2 lokal + kohort
    with Step("m2_lokal_kohort"):
        curves, aucs, coh = {}, {}, []
        for m, of in order_f.items():
            cv, pr = deletion_curve(XP, of, KS_F, rep_feat)
            curves[m], aucs[m] = cv, auc(cv, KS_F)
            for j, k in enumerate(KS_F):
                coh.append(dict(metode=m, k=k, f1_macro_kohort=f1_score(YP, pr[:, j], average="macro",
                                                                         labels=np.arange(N_CLS), zero_division=0)))
        rand_seeds = np.random.SeedSequence(SEED).generate_state(NI * REP).reshape(NI, REP)
        np.save(f"{B}/deletion_random_seeds{OUT_SUFFIX}.npy", rand_seeds)
        rc = []
        for r in range(REP):
            cv, pr = deletion_curve(XP, lambda i, r=r: np.random.default_rng(rand_seeds[i, r]).permutation(NF), KS_F, rep_feat)
            rc.append(cv)
            for j, k in enumerate(KS_F):
                coh.append(dict(metode=f"acak_r{r}", k=k, f1_macro_kohort=f1_score(YP, pr[:, j], average="macro",
                                                                                   labels=np.arange(N_CLS), zero_division=0)))
        curves["acak"] = np.stack(rc)                 # (5, n, k)
        aucs["acak"] = np.stack([auc(c_, KS_F) for c_ in rc]).mean(0)
        loc = pd.DataFrame({m: v for m, v in aucs.items()})
        loc.insert(0, "session_id", SESS_P)
        loc.insert(0, "instance", np.arange(NI))
        loc.to_csv(f"{R}/m2_lokal_auc{OUT_SUFFIX}.csv", index=False)
        cohdf = pd.DataFrame(coh)
        cohdf.to_csv(f"{R}/m2_kohort_f1_TIDAK_DICETAK{OUT_SUFFIX}.csv", index=False)
        all_c = np.concatenate([curves[m] for m in order_f] + [curves["acak"].reshape(-1, len(KS_F))])
        keep("m2_curves", all_c)
        p0_ok = all(np.allclose(curves[m][:, 0], P0[np.arange(NI), C], atol=1e-6) for m in order_f)
        check("deletion_lokal", p0_ok and np.isfinite(all_c).all() and (all_c >= 0).all() and (all_c <= 1).all(),
              f"{len(order_f)} metode + acak {REP} ulangan; p_i(0)=prob asli: {p0_ok}; semua p di [0,1]")
        fv = cohdf.f1_macro_kohort.to_numpy()
        check("deletion_kohort", np.isfinite(fv).all() and ((fv >= 0) & (fv <= 1)).all(),
              f"{len(fv)} nilai F1 kohort terhitung, semua di [0,1] (nilai tidak dicetak: instance dari partisi uji)")

    # ---------------- M2 sistem (disegel)
    with Step("m2_sistem"):
        t = time.time()
        Xt = np.asarray(X_te, dtype=np.float32)
        sealed = {}
        pred0 = predict(Xt).argmax(1)
        sealed["k0"] = f1_score(y_te, pred0, average="macro", labels=np.arange(N_CLS), zero_division=0)
        for m in order_f:
            g = np.argsort(-norm_rows(IF[m]).mean(0), kind="stable")
            for k in KS_F[1:]:
                Xk = Xt.copy()
                Xk[:, :, g[:k]] = base[g[:k]]
                sealed[f"{m}_k{k}"] = f1_score(y_te, predict(Xk).argmax(1), average="macro",
                                               labels=np.arange(N_CLS), zero_division=0)
                del Xk
            sealed[f"{m}_urutan_global"] = g
        np.savez(f"{R}/_sealed_system_check{OUT_SUFFIX}.npz", **{k: np.asarray(v) for k, v in sealed.items()})
        t_sys = time.time() - t
        vals = np.array([v for k, v in sealed.items() if not k.endswith("urutan_global")], dtype=float)
        check("deletion_sistem_jalan", np.isfinite(vals).all() and len(vals) == 1 + len(order_f) * (len(KS_F) - 1),
              f"{len(Xt):,} window uji x {len(vals)} konfigurasi, {t_sys:.0f}s; nilai disegel, tidak dibuka")
        del Xt

    # ---------------- kepekaan pengganti empiris
    with Step("m2_empiris"):
        rng = np.random.default_rng(SEED)
        bg_pick = rng.integers(0, N_BG, size=10)
        np.save(f"{B}/deletion_empiris_bg_pick{OUT_SUFFIX}.npy", bg_pick)
        emp = []
        for m, of in order_f.items():
            for r, j in enumerate(bg_pick):
                def rep_emp(x, sel, i, j=j):
                    x[:, sel] = bg[j][:, sel]
                    return x
                cv, _ = deletion_curve(XP, of, KS_F, rep_emp)
                emp.append(pd.DataFrame(dict(metode=m, ulangan=r, bg_window=int(j), instance=np.arange(NI),
                                             auc=auc(cv, KS_F))))
        emp = pd.concat(emp)
        emp.to_csv(f"{R}/m2_empiris_auc{OUT_SUFFIX}.csv", index=False)
        keep("m2_emp", emp.auc.to_numpy())
        check("deletion_empiris", np.isfinite(emp.auc).all() and len(emp) == len(order_f) * 10 * NI,
              f"{len(order_f)} metode x 10 ulangan x {NI} instance")

    # ---------------- temporal deletion
    with Step("temporal_deletion"):
        tl = {}
        for m, of in order_t.items():
            cv, _ = deletion_curve(XP, of, KS_T, rep_time)
            tl[m] = auc(cv, KS_T)
            keep(f"tdel_{m}", cv)
        rc = []
        for r in range(REP):
            cv, _ = deletion_curve(XP, lambda i, r=r: np.random.default_rng(rand_seeds[i, r]).permutation(T), KS_T, rep_time)
            rc.append(auc(cv, KS_T))
        tl["acak"] = np.mean(rc, 0)
        tdf = pd.DataFrame(tl)
        tdf.insert(0, "instance", np.arange(NI))
        tdf.to_csv(f"{R}/temporal_deletion_auc{OUT_SUFFIX}.csv", index=False)
        check("temporal_deletion", np.isfinite(tdf.to_numpy()).all(), f"{len(order_t)} metode + acak, k={KS_T}")

    # ---------------- M4
    with Step("m4"):
        def t_once(fn):
            t = time.perf_counter()
            fn()
            return time.perf_counter() - t

        runners = {
            "deepshap": lambda i: ex_ds.shap_values(XP[i:i + 1], check_additivity=False),
            "lime_flat": lambda i: lime_explain(lime_ex, XP[i], C[i], SEED),
            "kustom_flat": lambda i: custom_lime(XP[i], C[i], "kustom_flat", bg, SEED),
            "per_fitur": lambda i: custom_lime(XP[i], C[i], "per_fitur", bg, SEED),
            "per_timestep": lambda i: custom_lime(XP[i], C[i], "per_timestep", bg, SEED),
        }
        init_t.update({v: 0.0 for v in GROUPS})   # Algoritma 1 tidak punya tahap inisialisasi terpisah
        rt = []
        raw_t = []
        for m, fn in runners.items():
            for i in range(5):                      # pemanasan, dibuang
                fn(i)
            tt = np.array([[t_once(lambda i=i: fn(i)) for _ in range(10)] for i in range(NI)])   # (40, 10)
            raw_t.append(pd.DataFrame(tt, columns=[f"ulang_{r}" for r in range(10)]).assign(metode=m, instance=np.arange(NI)))
            ts_ = tt.mean(1)                        # rerata per instance
            q1, q3 = np.percentile(ts_, [25, 75])
            rt.append(dict(metode=m, init_detik=round(init_t[m], 4), per_instance_mean=ts_.mean(),
                           sd=ts_.std(ddof=1), median=np.median(ts_), iqr=q3 - q1, n_ukur=tt.size,
                           n_instance=NI, puncak_memori_mb=round(peak_mb(), 1)))
        pd.concat(raw_t).to_csv(f"{R}/runtime_mentah{OUT_SUFFIX}.csv", index=False)
        rt = pd.DataFrame(rt)
        rt.to_csv(f"{R}/runtime{OUT_SUFFIX}.csv", index=False)
        keep("m4", rt[["per_instance_mean", "sd", "median", "iqr"]].to_numpy())
        check("m4_timing", len(rt) == 5 and (rt.n_ukur == NI * 10).all(),
              f"5 metode; 5 pemanasan dibuang, {NI} instance x 10 pengulangan, batch 1, thread intra=inter=4")

    # ---------------- M3
    def stab(runs, level):
        """runs: list R x (n, ...) -> per instance Jaccard@k & Spearman berpasangan
        (k = 3 utk per_timestep, 10 utk lainnya)."""
        kk = 3 if level == "per_timestep" else 10
        J, S = np.zeros(NI), np.zeros(NI)
        for i in range(NI):
            js, ss = [], []
            for a_, b_ in combinations(range(len(runs)), 2):
                va, vb = runs[a_][i], runs[b_][i]
                js.append(jacc(va, vb, kk))
                ss.append(spear(va, vb))
            J[i], S[i] = np.mean(js), np.nanmean(ss) if np.isfinite(ss).any() else np.nan
        return J, S

    def level_vec(m, a_):
        if m in ("deepshap", "lime_flat", "kustom_flat"):
            return agg_f(a_)
        return a_

    with Step("m3_prosedural"):
        proc = {m: [] for m in ["deepshap", "lime_flat", "kustom_flat", "per_fitur", "per_timestep"]}
        bg_proc = []
        for s in SEEDS_PROC:
            bi = empirical_bg(s)
            bg_proc.append(bi)
            proc["deepshap"].append(level_vec("deepshap", deepshap_explain(bi, XP, C)[3]))
            proc["lime_flat"].append(level_vec("lime_flat", np.array([lime_explain(lime_ex, XP[i], C[i], s) for i in range(NI)])))
            for v in GROUPS:
                proc[v].append(level_vec(v, np.array([custom_lime(XP[i], C[i], v, bg, s) for i in range(NI)])))
        np.save(f"{B}/bg_procedural_idx{OUT_SUFFIX}.npy", np.stack(bg_proc))
        m3 = []
        for m, runs in proc.items():
            J, S = stab(runs, m)
            m3.append(pd.DataFrame(dict(jenis="prosedural", metode=m, k=3 if m == "per_timestep" else 10,
                                        instance=np.arange(NI), jaccard_k=J, spearman=S)))

    with Step("m3_perturbasi"):
        rng = np.random.default_rng(SEED)
        num_mask = np.ones(NF, bool)
        num_mask[CAT_F] = False
        noisy = []
        for r in range(REP):
            Xn = XP.copy()
            Xn[:, :, num_mask] = np.clip(Xn[:, :, num_mask] + rng.normal(0, 0.01, Xn[:, :, num_mask].shape), 0, 1)
            noisy.append(Xn.astype(np.float32))
        pert = {m: [] for m in proc}
        for Xn in noisy:
            svn = np.array(ex_ds.shap_values(Xn, check_additivity=False))
            pert["deepshap"].append(level_vec("deepshap", svn[C, np.arange(NI)]))
            pert["lime_flat"].append(level_vec("lime_flat", np.array([lime_explain(lime_ex, Xn[i], C[i], SEED) for i in range(NI)])))
            for v in GROUPS:
                pert[v].append(level_vec(v, np.array([custom_lime(Xn[i], C[i], v, bg, SEED) for i in range(NI)])))
        for m, runs in pert.items():
            J, S = stab(runs, m)
            m3.append(pd.DataFrame(dict(jenis="perturbasi", metode=m, k=3 if m == "per_timestep" else 10,
                                        instance=np.arange(NI), jaccard_k=J, spearman=S)))
        m3 = pd.concat(m3)
        m3.to_csv(f"{R}/m3_stability{OUT_SUFFIX}.csv", index=False)
        for jn in ["prosedural", "perturbasi"]:
            sub = m3[m3.jenis == jn]
            # Spearman bisa NaN bila vektor konstan; dilaporkan, bukan kegagalan jalur
            ok = np.isfinite(sub.jaccard_k).all() and sub.jaccard_k.between(0, 1).all()
            check(f"m3_{jn}", ok, f"5 metode x {NI} instance, R={REP}, k=10 (per time-step k=3); Jaccard di [0,1]; "
                                  f"Spearman NaN: {int(sub.spearman.isna().sum())}")
        keep("m3_j", m3.jaccard_k.to_numpy())

    # ---------------- bootstrap sesi berpasangan
    with Step("bootstrap"):
        diff = aucs["deepshap"] - aucs["lime_flat"]
        rng = np.random.default_rng(SEED)
        cls_sess = {c: np.unique(SESS_P[YP == c]) for c in np.unique(YP)}
        stats = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            wsess = {}
            for c, ss in cls_sess.items():
                for s, n in zip(*np.unique(rng.choice(ss, size=len(ss), replace=True), return_counts=True)):
                    wsess[(c, s)] = n
            w = np.array([wsess.get((YP[i], SESS_P[i]), 0) for i in range(NI)], dtype=float)
            stats[b] = np.sum(w * diff) / w.sum()
        lo, hi = np.percentile(stats, [2.5, 97.5])
        pd.DataFrame(dict(ulangan=np.arange(N_BOOT), selisih_auc_deepshap_minus_limeflat=stats)).to_csv(
            f"{R}/bootstrap_contoh{OUT_SUFFIX}.csv", index=False)
        keep("boot", stats)
        check("bootstrap_jalan", np.isfinite(stats).all() and len(stats) == N_BOOT,
              f"{N_BOOT} ulangan, sesi diambil ulang per kelas ({len(np.unique(SESS_P))} sesi); "
              f"statistik tersimpan di bootstrap_contoh.csv (tidak ditafsirkan)")

    bad = {k: int((~np.isfinite(v)).sum()) for k, v in NAN_POOL.items() if (~np.isfinite(v)).any()}
    check("tidak_ada_NaN", not bad, f"array berisi NaN/inf: {bad or 'tidak ada'} "
                                    f"(Spearman NaN dari vektor konstan dihitung terpisah di m1/m3)")

    pd.DataFrame(CHECKS).to_csv(f"{R}/pilot_checks{OUT_SUFFIX}.csv", index=False)
    json.dump(dict(
        val_f1_macro_per_epoch=[c_["val_f1_macro"] for c_ in train_log["curve"]],
        train_curve=train_log["curve"], epoch_terpilih=train_log["epoch_terpilih"],
        val_f1_macro_terpilih=train_log["val_f1_macro_terpilih"], waktu_latih_detik=train_log["waktu_latih_detik"],
        puncak_memori_mb=round(peak_mb(), 1), waktu_per_langkah_detik=STEP_T,
        waktu_total_detik=round(time.time() - T0, 1),
        n_instance=NI, n_lime_samples=N_LIME, seeds=dict(utama=SEED, prosedural=SEEDS_PROC),
        galat_diperbaiki=errors_fixed, keputusan_implementasi=__doc__.strip(),
        versi=dict(python=sys.version.split()[0], tensorflow=tf.__version__, shap=shap.__version__,
                   lime="0.2.0.1", numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                   scipy=scipy.__version__),
        threads=dict(intra=4, inter=4, omp=4)), open(f"{R}/pilot_log{OUT_SUFFIX}.json", "w"), indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("SELESAI")


if __name__ == "__main__":
    main()
