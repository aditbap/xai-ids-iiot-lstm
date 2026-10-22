"""
C2: latih SATU pembanding pohon (rf | xgb) pada window diratakan (220 dimensi)
ruang model utama. Bobot kelas balanced. Data uji tidak dimuat.
Pemakaian: python c2_tree_one.py '<json spec>'  (name, model, seed, params, save)
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(__file__))
from common_c import B2, MODELS, N_CLS, RUNS, run_log, save_json  # noqa: E402


def main():
    spec = json.loads(sys.argv[1])
    name = spec["name"]
    rec_path = f"{RUNS}/{name}.json"
    if os.path.exists(rec_path):
        run_log(f"C2 {name}: sudah ada, dilewati")
        return
    X_tr = np.load(f"{B2}/X_train.npy").reshape(-1, 220)
    y_tr = np.load(f"{B2}/y_train.npy").astype(int)
    X_va = np.load(f"{B2}/X_val.npy").reshape(-1, 220)
    y_va = np.load(f"{B2}/y_val.npy").astype(int)
    p, seed = spec["params"], spec["seed"]
    t0 = time.time()
    if spec["model"] == "rf":
        clf = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_leaf=p["min_samples_leaf"], class_weight="balanced",
                                     random_state=seed, n_jobs=4).fit(X_tr, y_tr)
    else:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                            learning_rate=p["learning_rate"], subsample=p["subsample"], tree_method="hist",
                            n_jobs=4, random_state=seed, objective="multi:softprob", num_class=N_CLS)
        clf.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    t_train = time.time() - t0
    f1 = f1_score(y_va, clf.predict(X_va), average="macro", labels=np.arange(N_CLS), zero_division=0)
    path = None
    if spec.get("save", True):
        if spec["model"] == "rf":
            path = f"{MODELS}/{name}.joblib"
            joblib.dump(clf, path, compress=3)
        else:
            path = f"{MODELS}/{name}.ubj"
            clf.save_model(path)
    save_json(dict(spec=spec, epoch_terpilih=None, val_f1_macro=float(f1), waktu_latih_detik=round(t_train, 1),
                   model=path, n_train=len(y_tr), n_val=len(y_va)), rec_path)
    run_log(f"C2 {name}: F1 val {f1:.4f}, {t_train:.0f}s")


if __name__ == "__main__":
    main()
