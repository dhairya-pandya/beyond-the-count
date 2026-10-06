#!/usr/bin/env bash
# Full three-representation audit with matched-settings permutation nulls, then every downstream table. CPU only; ~6-10 h on a 10-core laptop.
# Stage A: audits of ALL 404 endpoints (3 CV repeats, 1000 bootstrap resamples) for CellProfiler, CP-CNN, DINOv2 ('allpod').
# Stage B: label-permutation nulls with the SAME settings (3 repeats, 1000 resamples) per representation.
# Stage C: null-calibrated verdicts, enrichment, paper comparison, compound classes, filter-stratified check, headline numbers, sensitivity, figures.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NJ=${1:-8}
L=results/logs
mkdir -p $L
stamp() { echo "=== $1 $(date)"; }
for fs in cellprofiler cpcnn dino; do
  stamp "audit $fs"; $PY scripts/03_run_audit.py --feature-set $fs --agg allpod --n-repeats 3 --n-boot 1000 --n-jobs $NJ > $L/full_$fs.log 2>&1
done
declare -A RUNS=( [cellprofiler]=100 [cpcnn]=75 [dino]=30 )   # endpoints x 2 permutations x (seed 0, seed 1 for the first two) -> 400 / 300 / 60 runs
for fs in cellprofiler cpcnn dino; do
  stamp "null $fs"; $PY scripts/08_null_control.py --feature-set $fs --agg allpod --n-endpoints ${RUNS[$fs]} --n-perms 2 --n-repeats 3 --n-boot 1000 --seed 0 --n-jobs $NJ > $L/fullnull_$fs.log 2>&1
  if [ "$fs" != "dino" ]; then
    $PY scripts/08_null_control.py --feature-set $fs --agg allpod --n-endpoints ${RUNS[$fs]} --n-perms 2 --n-repeats 3 --n-boot 1000 --seed 1 --append --n-jobs $NJ >> $L/fullnull_$fs.log 2>&1
  fi
done
for fs in cellprofiler cpcnn dino; do
  c=${fs}_allpod; stamp "downstream $c"
  $PY scripts/15_calibrate_verdicts.py --config $c > $L/cal_$c.log 2>&1
  $PY scripts/04_run_enrichment.py --config $c > $L/enr_$c.log 2>&1
  $PY scripts/06_compare_to_paper.py --config $c > $L/paper_$c.log 2>&1
  $PY scripts/18_compound_classes.py --config $c > $L/cc_$c.log 2>&1
  $PY scripts/19_filter_stratified.py --config $c > $L/fs_$c.log 2>&1
  $PY scripts/12_headline_numbers.py --config $c > $L/hn_$c.log 2>&1
  $PY scripts/13_sensitivity_postprocess.py --config $c > $L/sens_$c.log 2>&1
done
$PY scripts/09_technical_probe.py --config cellprofiler_allpod --n-jobs $NJ > $L/probe.log 2>&1
$PY scripts/10_feature_families.py --config cellprofiler_allpod --n-jobs $NJ > $L/families.log 2>&1
$PY scripts/07_finalize_results.py --primary cellprofiler_allpod > $L/finalize.log 2>&1
stamp "ALL DONE"
