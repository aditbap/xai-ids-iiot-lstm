"""
Inti XAI Tahap C (port langsung dari scripts/stage_b2/b2_pilot_xai_log.py,
rincian implementasi identik dengan pilot B2). Diimpor SETELAH skrip
pemanggil mengatur urutan impor TF (TF_USE_LEGACY_KERAS, disable_v2_behavior).
"""

import time

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import os

N_BG, N_CLS, T = 100, 9, 10
N_LIME = int(os.environ.get("STAGE_C_SMOKE_NLIME", "5000"))   # run resmi: 5000


class Ctx:
    """Konteks satu model: model Keras, data latih (n,10,F), fitur kategorikal,
    parameter Min-Max untuk kolom kategorikal (kode <-> ruang model)."""

    def __init__(self, model, X_tr, y_tr, cat_f, scale, mn, codes_len):
        self.model, self.X_tr, self.y_tr = model, X_tr, y_tr
        self.NF = X_tr.shape[2]
        self.cat_f = list(cat_f)
        self.cat_d = np.array([t * self.NF + f for t in range(T) for f in cat_f])
        self.cat_d_f = np.array([f for t in range(T) for f in cat_f])
        self.scale, self.mn = np.asarray(scale), np.asarray(mn)
        self.valid = {f: set(range(n)) for f, n in zip(cat_f, codes_len)}
        self.rec = dict(active=False, invalid=0, total=0)
        self.groups = {"kustom_flat": np.arange(T * self.NF), "per_fitur": np.tile(np.arange(self.NF), T),
                       "per_timestep": np.repeat(np.arange(T), self.NF)}
        self.m_of = {"kustom_flat": T * self.NF, "per_fitur": self.NF, "per_timestep": T}

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=np.float32), batch_size=2048, verbose=0)

    # ---------------- background
    def empirical_bg(self, seed, n=N_BG):
        rng = np.random.default_rng(seed)
        cnt = np.bincount(self.y_tr, minlength=N_CLS)
        quota = n * cnt / cnt.sum()
        alloc = np.floor(quota).astype(int)
        alloc[np.argsort(-(quota - alloc), kind="stable")[:n - alloc.sum()]] += 1
        idx = np.concatenate([rng.choice(np.flatnonzero(self.y_tr == c), alloc[c], replace=False) for c in range(N_CLS)])
        return np.sort(idx)

    def balanced_bg(self, seed, n=N_BG):
        rng = np.random.default_rng(seed)
        alloc = np.full(N_CLS, n // N_CLS)
        alloc[rng.choice(N_CLS, n - alloc.sum(), replace=False)] += 1
        idx = np.concatenate([rng.choice(np.flatnonzero(self.y_tr == c), alloc[c], replace=False) for c in range(N_CLS)])
        return np.sort(idx)

    # ---------------- DeepSHAP
    def deepshap(self, bg_idx):
        t = time.time()
        ex = shap.DeepExplainer(self.model, self.X_tr[bg_idx])
        return ex, time.time() - t

    @staticmethod
    def ds_explain(ex, X, C, chunk=50):
        svs = [np.array(ex.shap_values(X[i:i + chunk], check_additivity=False)) for i in range(0, len(X), chunk)]
        sv = np.concatenate(svs, axis=1)                         # (9, n, 10, F)
        ev = np.array(ex.expected_value).ravel()
        return sv[C, np.arange(len(X))], sv, ev

    @staticmethod
    def additivity(sv, ev, P, C):
        recon = sv.reshape(sv.shape[0], sv.shape[1], -1).sum(-1).T + ev
        err = np.abs(recon - P)
        return err[np.arange(len(P)), C], err.max(1)

    # ---------------- LIME flat
    def to_mixed(self, X):
        Z = np.array(X, dtype=np.float64).reshape(len(X), -1)
        Z[:, self.cat_d] = np.rint((Z[:, self.cat_d] - self.mn[self.cat_d_f]) / self.scale[self.cat_d_f])
        return Z

    def from_mixed(self, Z):
        Z = np.array(Z, dtype=np.float64, copy=True)
        Z[:, self.cat_d] = Z[:, self.cat_d] * self.scale[self.cat_d_f] + self.mn[self.cat_d_f]
        return Z.reshape(len(Z), T, self.NF).astype(np.float32)

    def wrapper(self, Z):
        if self.rec["active"]:
            cats = np.asarray(Z)[:, self.cat_d]
            for j, f in enumerate(self.cat_d_f):
                self.rec["invalid"] += int((~np.isin(cats[:, j], list(self.valid[f]))).sum())
            self.rec["total"] += cats.size
        return self.predict(self.from_mixed(Z))

    def build_lime(self, seed, feat_names, cat_names_by_f):
        Ztr = self.to_mixed(self.X_tr)
        names = [f"{feat_names[f]}_t{t + 1}" for t in range(T) for f in range(self.NF)]
        cat_names = {int(d): cat_names_by_f[int(f)] for d, f in zip(self.cat_d, self.cat_d_f)}
        t = time.time()
        ex = LimeTabularExplainer(Ztr, mode="classification", feature_names=names,
                                  categorical_features=self.cat_d.tolist(), categorical_names=cat_names,
                                  discretize_continuous=False, feature_selection="none", random_state=seed)
        t_init = time.time() - t
        del Ztr
        self.lime = ex
        return ex, t_init

    def lime_explain(self, x, c, seed):
        ex = self.lime
        ex.random_state = np.random.RandomState(seed)
        e = ex.explain_instance(self.to_mixed(x[None])[0], self.wrapper, labels=(int(c),),
                                num_features=T * self.NF, num_samples=N_LIME)
        w = np.zeros(T * self.NF)
        for d, v in e.local_exp[int(c)]:
            w[d] = v
        return w.reshape(T, self.NF)

    # ---------------- LIME kustom (Algoritma 1)
    def custom(self, x, c, variant, bg, seed):
        m, g = self.m_of[variant], self.groups[variant]
        rng = np.random.default_rng(seed)
        Z = rng.binomial(1, 0.5, size=(N_LIME, m))
        Z[0] = 1
        bsel = rng.integers(0, len(bg), size=N_LIME)
        xf = x.reshape(-1)
        Xs = np.where(Z[:, g] == 1, xf[None, :], bg.reshape(len(bg), -1)[bsel])
        p = self.predict(Xs.reshape(N_LIME, T, self.NF))[:, c]
        d = np.sqrt((1 - Z).sum(1))
        sigma = 0.75 * np.sqrt(m)
        pi = np.exp(-d ** 2 / sigma ** 2)
        beta = Ridge(alpha=1.0).fit(Z, p, sample_weight=pi).coef_
        out = np.abs(beta)
        return out.reshape(T, self.NF) if variant == "kustom_flat" else out


# ---------------- agregasi & metrik
def agg_f(att):
    return np.abs(att).mean(axis=1)


def agg_t(att):
    return np.abs(att).mean(axis=2)


def norm_rows(M):
    s = M.sum(1, keepdims=True)
    return M / np.where(s == 0, 1, s)


def topk(v, k):
    return set(np.argsort(-np.asarray(v), kind="stable")[:k])


def jacc(a, b, k):
    A_, B_ = topk(a, k), topk(b, k)
    return len(A_ & B_) / len(A_ | B_)


def spear(a, b):
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else np.nan


def level_vec(m, a_):
    return agg_f(a_) if m in ("deepshap", "lime_flat", "kustom_flat") else a_
