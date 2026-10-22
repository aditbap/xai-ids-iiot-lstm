"""
Driver Tahap D (berurutan, satu proses anak per langkah; berhenti pada galat;
langkah yang keluarannya sudah ada dilewati oleh skrip masing-masing).
Pemakaian: python scripts/stage_d/run_stage_d.py [d0|d1|pred|d12|d3|d4|all]
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common_d import D_MODELS, D_RES, run_log, save_json, sha256  # noqa: E402

HERE = os.path.dirname(__file__)


def sh(script, *args):
    run_log(f"MULAI {script} {' '.join(args)}")
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args])
    if r.returncode != 0:
        run_log(f"GAGAL {script} {' '.join(args)} (kode {r.returncode}) -- BERHENTI")
        sys.exit(r.returncode)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("d0", "all"):
        sh("d0_prep.py")
    if step in ("d1", "all"):
        for s in ("43", "44"):
            sh("d1_train_old.py", s)
        hp = f"{D_RES}/d1_models_hash.json"
        if not os.path.exists(hp):
            save_json({f"{D_MODELS}/old_s{s}.h5": sha256(f"{D_MODELS}/old_s{s}.h5") for s in (43, 44)}, hp)
    if step in ("pred", "all"):
        sh("d_predict.py")
    if step in ("d12", "all"):
        sh("d12_report.py", "all")
    if step in ("d3", "all"):
        sh("d3_select.py")
        sh("d3_c3.py")
        for part in ("primer", "m4", "m3_proc", "m3_pert_0.01", "m3_pert_0.005", "m3_pert_0.02", "sens_bg", "deletion"):
            sh("d3_c4_xai.py", part)
        sh("d3_c4_shortcut.py")
        sh("d3_c4_report.py")
    if step in ("d4", "all"):
        for s in ("42", "43", "44"):
            sh("d4_seed.py", s)
        sh("d4_seed.py", "report")
    run_log(f"driver selesai: {step}")


if __name__ == "__main__":
    main()
