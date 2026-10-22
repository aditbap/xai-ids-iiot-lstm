"""
D1: latih SATU model konfigurasi LAMA (tanpa log1p; artifacts/stage_d/rows_old.npy)
dengan protokol identik model utama Tahap C. Kelas callback dan fungsi build
DIIMPOR dari scripts/stage_c/c2_train_one.py (bukan disalin), sehingga mode
eager, determinisme, thread 4/4, urutan penguncian seed, 12 epoch, batch 64,
Adam 0.001, class_weight balanced, dan pemilihan epoch F1-macro validasi sama
persis. Data uji tidak dimuat.

Pemakaian: python d1_train_old.py <seed>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_c"))
import c2_train_one as C2  # noqa: E402  (mengatur env & TF eager seperti Tahap C)

import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common_d import D_ART, D_MODELS, D_RUNS, N_CLS, load_meta, run_log, save_json, windows  # noqa: E402


def main():
    seed = int(sys.argv[1])
    name = f"old_s{seed}"
    rec = f"{D_RUNS}/{name}.json"
    if os.path.exists(rec):
        return run_log(f"D1 {name}: sudah ada, dilewati")
    X = np.load(f"{D_ART}/rows_old.npy")
    meta = load_meta()
    lab, sid = meta.label.to_numpy(), meta.session_id.to_numpy()
    pos, part = meta.pos.to_numpy(), meta.partisi.to_numpy()
    X_tr, y_tr, _ = windows(X, lab, sid, pos, part, "train", 10)
    X_va, y_va, _ = windows(X, lab, sid, pos, part, "val", 10)
    X_tr, X_va = X_tr.astype(np.float32), X_va.astype(np.float32)
    y_tr, y_va = y_tr.astype(int), y_va.astype(int)
    del X
    cw = compute_class_weight("balanced", classes=np.arange(N_CLS), y=y_tr)
    cw = {i: float(v) for i, v in enumerate(cw)}

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = C2.build("lstm2", X_tr.shape[1:], {})
    cb = C2.BestValF1(X_va, y_va)
    t0 = time.time()
    model.fit(X_tr, tf.keras.utils.to_categorical(y_tr, N_CLS), epochs=C2.EPOCHS, batch_size=C2.BATCH,
              class_weight=cw, callbacks=[cb], verbose=0, shuffle=True)
    t_train = time.time() - t0
    model.set_weights(cb.best_w)
    path = f"{D_MODELS}/{name}.h5"
    model.save(path)
    save_json(dict(name=name, seed=seed, konfigurasi="lama (tanpa log1p)", class_weight=cw, curve=cb.curve,
                   epoch_terpilih=cb.best_epoch, val_f1_macro=float(cb.best_f1), waktu_latih_detik=round(t_train, 1),
                   model=path, n_train=len(y_tr), n_val=len(y_va)), rec)
    run_log(f"D1 {name}: epoch {cb.best_epoch}, F1 val {cb.best_f1:.4f}, {t_train:.0f}s")


if __name__ == "__main__":
    main()
