#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Example:
#   PRETRAIN=/path/to/ckpt.pth SEEDS="3407 3408 3409" \
#   bash tools/analysis_tools/run_b_soft_lambda_seed_suite.sh
PRETRAIN="${PRETRAIN:-}"
SEEDS="${SEEDS:-3407 3408 3409}"

CFG_SOFT05="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_nus-3d.py"
CFG_SOFT03="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_lam03_nus-3d.py"
CFG_SOFT02="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_lam02_nus-3d.py"

BASE_CFG_OPTS=(
  "val_evaluator.version=v1.0-mini"
  "test_evaluator.version=v1.0-mini"
)
if [[ -n "$PRETRAIN" ]]; then
  BASE_CFG_OPTS+=("load_from=${PRETRAIN}")
fi

for seed in $SEEDS; do
  WD05="work_dirs/abmini_cross_b_geomask_soft_s${seed}_lam05"
  WD03="work_dirs/abmini_cross_b_geomask_soft_s${seed}_lam03"
  WD02="work_dirs/abmini_cross_b_geomask_soft_s${seed}_lam02"

  echo "[seed=${seed}] [1/4] Train soft lambda=0.5"
  python tools/train.py "$CFG_SOFT05" --work-dir "$WD05" \
    --cfg-options "${BASE_CFG_OPTS[@]}" "randomness.seed=${seed}"

  echo "[seed=${seed}] [2/4] Train soft lambda=0.3"
  python tools/train.py "$CFG_SOFT03" --work-dir "$WD03" \
    --cfg-options "${BASE_CFG_OPTS[@]}" "randomness.seed=${seed}"

  echo "[seed=${seed}] [3/4] Train soft lambda=0.2"
  python tools/train.py "$CFG_SOFT02" --work-dir "$WD02" \
    --cfg-options "${BASE_CFG_OPTS[@]}" "randomness.seed=${seed}"

  echo "[seed=${seed}] [4/4] Compare soft lambdas (0.5 vs 0.3 vs 0.2)"
  python tools/analysis_tools/compare_ab_matched_ious.py \
    --exp-a "$WD05" \
    --exp-b "$WD03" \
    --exp-c "$WD02" \
    --max-iter 500
done
