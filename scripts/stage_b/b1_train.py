"""
Tahap B1: latih model utama, seed 42 saja. 12 epoch tetap, tanpa
EarlyStopping; bobot epoch dengan F1-macro VALIDASI terbaik disimpan sebagai
artifacts/stage_b/lstm_main_seed42.h5. Data uji TIDAK disentuh.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

import json
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.experimental.enable_op_determinism()

B = "artifacts/stage_b"
SEED = 42
EPOCHS = 12
BATCH = 64
N_CLS = 9


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


def main():
    X_tr, y_tr = np.load(f"{B}/X_train.npy"), np.load(f"{B}/y_train.npy").astype(int)
    X_va, y_va = np.load(f"{B}/X_val.npy"), np.load(f"{B}/y_val.npy").astype(int)
    assert X_tr.shape[1:] == (10, 22)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLS), y=y_tr)
    cw = {i: float(w) for i, w in enumerate(cw)}

    # seed dikunci tepat sebelum model dibangun
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 22)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(N_CLS, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy")
    cb = BestValF1(X_va, y_va)
    t0 = time.time()
    model.fit(X_tr, tf.keras.utils.to_categorical(y_tr, N_CLS), epochs=EPOCHS, batch_size=BATCH,
              class_weight=cw, callbacks=[cb], verbose=0, shuffle=True)
    t_train = time.time() - t0
    model.set_weights(cb.best_w)
    model.save(f"{B}/lstm_main_seed42.h5")
    json.dump(dict(seed=SEED, epochs=EPOCHS, batch=BATCH, class_weight=cw, curve=cb.curve,
                   epoch_terpilih=cb.best_epoch, val_f1_macro_terpilih=float(cb.best_f1),
                   waktu_latih_detik=round(t_train, 1), tf=tf.__version__,
                   threads=dict(intra=4, inter=4, omp=4), deterministic_ops=True),
              open("results/stage_b/train_log.json", "w"), indent=1)
    print(f"epoch terpilih {cb.best_epoch}, val F1 {cb.best_f1:.4f}, waktu {t_train:.0f}s")


if __name__ == "__main__":
    main()
