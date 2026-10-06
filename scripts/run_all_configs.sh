#!/usr/bin/env bash
# Run every (feature set, aggregation) configuration used for the sensitivity analysis. CPU only.
# The two primary runs (cellprofiler_allpod, cpcnn_allpod) were done on ALL endpoints; secondary runs use --powered-only.
# Usage: bash scripts/run_all_configs.sh [N_JOBS]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NJ=${1:-9}
for cfg in "cellprofiler allpod" "cpcnn allpod" "dino allpod" "cpcnn all" "cpcnn allpodcc" "cellprofiler all" "cellprofiler allpodcc" "dino all" "dino allpodcc"; do
  set -- $cfg
  if [ -f "results/$1_$2/audit_table.csv" ]; then echo "skip $1 $2 (done)"; continue; fi
  EXTRA="--powered-only"; REP=3
  [ "$1" = "dino" ] && REP=2   # 4,436 features: 2 repeats keeps it laptop-sized
  echo "=== $1 $2 $(date)"
  $PY scripts/03_run_audit.py --feature-set "$1" --agg "$2" --n-repeats "$REP" --n-boot 1000 --n-jobs "$NJ" $EXTRA > "results/logs/$1_$2.log" 2>&1
  $PY scripts/04_run_enrichment.py --config "$1_$2" > "results/logs/$1_$2.enrichment.log" 2>&1
  $PY scripts/06_compare_to_paper.py --config "$1_$2" > "results/logs/$1_$2.paper.log" 2>&1 || true
done
echo "all configs done $(date)"
