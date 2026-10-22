"""
Driver Tahap C. Setiap langkah dijalankan sebagai subproses terpisah dan
dilewati bila keluarannya sudah ada (bisa dilanjutkan). Berhenti pada galat
pertama (tidak ada langkah pengganti).

Pemakaian: python scripts/stage_c/run_stage_c.py [c0|c1|c2|c3|c4|all]
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common_c import B2, C_RES, MODELS, RUNS, run_log, save_json  # noqa: E402

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(script, *args):
    cmd = [PY, os.path.join(HERE, script), *args]
    run_log(f"MULAI {script} {' '.join(a[:120] for a in args)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        run_log(f"GAGAL {script} (kode {r.returncode}) -- BERHENTI")
        sys.exit(r.returncode)


def keras(spec):
    if not os.path.exists(f"{RUNS}/{spec['name']}.json"):
        sh("c2_train_one.py", json.dumps(spec))


def tree(spec):
    if not os.path.exists(f"{RUNS}/{spec['name']}.json"):
        sh("c2_tree_one.py", json.dumps(spec))


def search_configs(model):
    """20 konfigurasi Tabel 7, rng default_rng(42) terpisah per model."""
    rng = np.random.default_rng(42)
    out = []
    for _ in range(20):
        if model == "rf":
            c = dict(n_estimators=int(rng.integers(50, 501)), max_depth=int(rng.integers(5, 51)),
                     min_samples_leaf=int(rng.integers(1, 21)))
        elif model == "xgb":
            c = dict(n_estimators=int(rng.integers(50, 501)), max_depth=int(rng.integers(3, 16)),
                     learning_rate=float(rng.uniform(0.01, 0.3)), subsample=float(rng.uniform(0.5, 1.0)))
        elif model == "mlp":
            c = dict(layers=int(rng.integers(1, 4)), units=int(rng.integers(32, 257)),
                     dropout=float(rng.uniform(0.1, 0.5)), lr=float(10 ** rng.uniform(-4, -2)))
        else:
            c = dict(units=int(rng.integers(32, 129)), dropout=float(rng.uniform(0.1, 0.5)),
                     lr=float(10 ** rng.uniform(-4, -2)))
        out.append(c)
    return out


def c2():
    # model utama seed 42 = model Tahap B2 (tidak dilatih ulang)
    dst = f"{MODELS}/main_s42.h5"
    if not os.path.exists(f"{RUNS}/main_s42.json"):
        shutil.copy(f"{B2}/lstm_main_seed42.h5", dst)
        tl = json.load(open("results/diag_log/train_log.json"))
        save_json(dict(spec=dict(name="main_s42", arch="lstm2", inputs="main", w=10, seed=42,
                                 sumber="salinan artifacts/stage_b2/lstm_main_seed42.h5 (tidak dilatih ulang)"),
                       curve=tl["curve"], epoch_terpilih=tl["epoch_terpilih"], val_f1_macro=tl["val_f1_macro_terpilih"],
                       waktu_latih_detik=tl["waktu_latih_detik"], model=dst), f"{RUNS}/main_s42.json")
    for s in (43, 44):
        keras(dict(name=f"main_s{s}", arch="lstm2", inputs="main", w=10, seed=s))
    for s in (45, 46):
        keras(dict(name=f"t6_urut_s{s}", arch="lstm2", inputs="main", w=10, seed=s))
    for s in (42, 43, 44, 45, 46):
        keras(dict(name=f"t6_acak_s{s}", arch="lstm2", inputs="main", w=10, seed=s, shuffle=True))
    for s in (42, 43, 44):
        keras(dict(name=f"control_s{s}", arch="lstm2", inputs="control", w=10, seed=s))
    for w in (1, 5, 20):
        for s in (42, 43, 44):
            keras(dict(name=f"abl_w{w}_s{s}", arch="lstm2", inputs="main", w=w, seed=s, ablation_subset=True))
    for model in ("rf", "xgb", "mlp", "lstm1"):
        cfgs = search_configs(model)
        save_json(cfgs, f"{C_RES}/c2_search_{model}_configs.json")
        for i, c in enumerate(cfgs):
            spec = dict(name=f"search_{model}_{i:02d}", model=model, seed=42, params=c, save=False)
            if model in ("rf", "xgb"):
                tree(spec)
            else:
                keras(dict(spec, arch=model, inputs="main", w=10))
        res = [json.load(open(f"{RUNS}/search_{model}_{i:02d}.json"))["val_f1_macro"] for i in range(20)]
        best = int(np.argmax(res))
        for s in (42, 43, 44):
            spec = dict(name=f"{model}_s{s}", model=model, seed=s, params=cfgs[best], search_index=best)
            if model in ("rf", "xgb"):
                tree(spec)
            else:
                keras(dict(spec, arch=model, inputs="main", w=10))
    sh("c2_summary.py")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("c0", "all"):
        sh("c0_prep.py")
    if step in ("c1", "all"):
        sh("c1_tests.py")
    if step in ("c2", "all"):
        c2()
    if step in ("c3", "all"):
        sh("c3_eval.py")
    if step in ("c4", "all"):
        for part in ("primer", "m4", "m3_proc", "m3_pert_0.01", "m3_pert_0.005", "m3_pert_0.02", "sens_bg", "deletion"):
            sh("c4_xai.py", part)
        sh("c4_shortcut.py")
        sh("c4_report.py")
    run_log(f"driver selesai: {step}")


if __name__ == "__main__":
    main()
