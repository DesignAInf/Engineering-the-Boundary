#!/usr/bin/env bash
# Reproduce all simulations and figures of the paper.
# Usage:  ./run_all.sh            (writes to results/ and paper/figs/)
#         ETB_RESDIR=/tmp/r ETB_FIGDIR=/tmp/f ./run_all.sh   (writes elsewhere)
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # avoid BLAS oversubscription
cd "$(dirname "$0")/code"
echo "[1/5] Gaussian fields";        python3 sim1_gaussian_fields.py
echo "[2/5] Emergence";              python3 sim2_emergence.py
echo "[3/5] DMBD-style inversion";   python3 sim3_inversion.py && python3 sim3_figure.py && python3 sim3_flat_prior.py
echo "[4/5] Foraging";               python3 sim4_run.py && python3 sim4_figures.py && python3 sim4_calibration.py
echo "[5/5] Pavilion";               python3 sim5_run.py && python3 sim5_run.py --fixed-horizon && python3 sim5_figures.py
echo "done"
