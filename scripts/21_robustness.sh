#!/usr/bin/env bash
# Robustness runs on top of the full audit (CellProfiler only, powered endpoints only, same 3 repeats / 1000 resamples):
#   aggregation rules `all` and `allpodcc`, and chemical-cluster grouped CV (+ its own permutation null).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; NJ=${1:-8}; L=results/logs
stamp() { echo "=== $1 $(date)"; }
for agg in all allpodcc; do
  stamp "audit cellprofiler $agg"; $PY scripts/03_run_audit.py --feature-set cellprofiler --agg $agg --powered-only --n-repeats 3 --n-boot 1000 --n-jobs $NJ > $L/rob_cellprofiler_$agg.log 2>&1
  $PY scripts/15_calibrate_verdicts.py --config cellprofiler_$agg --null-from cellprofiler_allpod > $L/rob_cal_$agg.log 2>&1
done
stamp "audit cellprofiler allpod, chemical-cluster groups"
$PY scripts/03_run_audit.py --feature-set cellprofiler --agg allpod --powered-only --group-by chem_cluster --n-repeats 3 --n-boot 1000 --n-jobs $NJ --out-dir results/cellprofiler_allpod_chem > $L/rob_chem.log 2>&1
stamp "null, chemical-cluster groups"
$PY scripts/08_null_control.py --feature-set cellprofiler --agg allpod --group-by chem_cluster --n-endpoints 50 --n-perms 2 --n-repeats 3 --n-boot 1000 --seed 3 --n-jobs $NJ --out-name null_control_chem.csv > $L/rob_chem_null.log 2>&1
cp results/cellprofiler_allpod/null_control_chem.csv results/cellprofiler_allpod_chem/null_control.csv
$PY scripts/15_calibrate_verdicts.py --config cellprofiler_allpod_chem > $L/rob_chem_cal.log 2>&1
$PY scripts/04_run_enrichment.py --config cellprofiler_allpod_chem > $L/rob_chem_enr.log 2>&1
stamp "ROBUSTNESS DONE"
