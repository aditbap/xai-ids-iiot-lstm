"""
C2: latih SATU model Keras (satu model per proses). Protokol identik dengan
scripts/diag_log/dl1_prep_train.py (model utama seed 42): mode eager,
TF_DETERMINISTIC_OPS, op determinism, thread 4/4, seed (random, numpy, tf)
dikunci tepat sebelum model dibangun, Adam, categorical CE, batch 64,
12 epoch, class_weight balanced dari y latih, bobot epoch dengan F1-macro
validasi terbaik. Data uji tidak dimuat.

Pemakaian: python c2_train_one.py '<json spec>'
spec: name, arch (lstm2|lstm1|mlp), inputs (main|control), w, seed,
      shuffle (bool, T6), params (units, dropout, lr, layers), save (bool)
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

import json
import random
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.experimental.enable_op_determinism()

sys.path.insert(0, os.path.dirname(__file__))
from common_c import BATCH, EPOCHS, MODELS, N_CLS, RUNS, build_inputs, load_rows, run_log, save_json  # noqa: E402


class BestValF1(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.best_f1, self.best_epoch, self.best_w = -1.0, -1, None
        self.curve, self._t = [], None

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, batch_size=2048, verbose=0).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro", labels=np.arange(N_CLS), zero_division=0)
        self.curve.append(dict(epoch=epoch + 1, train_loss=float(logs["loss"]), val_f1_macro=float(f1),
                               detik=round(time.time() - self._t, 1)))
        if f1 > self.best_f1:
            self.best_f1, self.best_epoch = f1, epoch + 1
            self.best_w = [w.copy() for w in self.model.get_weights()]
        print(f"epoch {epoch + 1}/{EPOCHS} loss={logs['loss']:.4f} val_f1_macro={f1:.4f}", flush=True)


def build(arch, in_shape, p):
    L = tf.keras.layers
    if arch == "lstm2":
        layers = [L.LSTM(128, return_sequences=True), L.Dropout(0.3), L.LSTM(64), L.Dropout(0.3),
                  L.Dense(64, activation="relu")]
    elif arch == "lstm1":
        layers = [L.LSTM(p["units"]), L.Dropout(p["dropout"]), L.Dense(64, activation="relu")]
    elif arch == "mlp":
        layers = []
        for _ in range(p["layers"]):
            layers += [L.Dense(p["units"], activation="relu"), L.Dropout(p["dropout"])]
    model = tf.keras.Sequential([L.Input(shape=in_shape)] + layers + [L.Dense(N_CLS, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.get("lr", 0.001)), loss="categorical_crossentropy")
    return model


def main():
    spec = json.loads(sys.argv[1])
    name = spec["name"]
    rec_path = f"{RUNS}/{name}.json"
    if os.path.exists(rec_path):
        run_log(f"C2 {name}: sudah ada, dilewati")
        return
    rows = load_rows()
    w, shuf = spec.get("w", 10), spec.get("shuffle", False)
    seed_sh = spec["seed"] if shuf else None
    X_tr, y_tr, _ = build_inputs(spec["inputs"], w, "train", seed_sh, rows)
    X_va, y_va, end_va = build_inputs(spec["inputs"], w, "val", seed_sh, rows)
    if spec["arch"] == "mlp":
        X_tr, X_va = X_tr.reshape(len(X_tr), -1), X_va.reshape(len(X_va), -1)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLS), y=y_tr)
    cw = {i: float(v) for i, v in enumerate(cw)}

    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build(spec["arch"], X_tr.shape[1:], spec.get("params", {}))
    cb = BestValF1(X_va, y_va)
    t0 = time.time()
    model.fit(X_tr, tf.keras.utils.to_categorical(y_tr, N_CLS), epochs=EPOCHS, batch_size=BATCH,
              class_weight=cw, callbacks=[cb], verbose=0, shuffle=True)
    t_train = time.time() - t0
    model.set_weights(cb.best_w)

    extra = {}
    if spec.get("ablation_subset"):
        meta = rows[2]
        blen = meta.groupby("session_id").pos.transform("max").to_numpy() + 1
        pos = meta.pos.to_numpy()
        m = (blen[end_va] >= 20) & (pos[end_va] >= 19)
        pred = model.predict(X_va[m], batch_size=2048, verbose=0).argmax(1)
        extra = dict(val_f1_macro_subset20=float(f1_score(y_va[m], pred, average="macro", labels=np.arange(N_CLS),
                                                           zero_division=0)), n_val_subset20=int(m.sum()))
    path = None
    if spec.get("save", True):
        path = f"{MODELS}/{name}.h5"
        model.save(path)
    save_json(dict(spec=spec, class_weight=cw, curve=cb.curve, epoch_terpilih=cb.best_epoch,
                   val_f1_macro=float(cb.best_f1), waktu_latih_detik=round(t_train, 1), model=path,
                   n_train=len(y_tr), n_val=len(y_va), in_shape=list(X_tr.shape[1:]), **extra), rec_path)
    run_log(f"C2 {name}: epoch {cb.best_epoch}, F1 val {cb.best_f1:.4f}, {t_train:.0f}s")


if __name__ == "__main__":
    main()
