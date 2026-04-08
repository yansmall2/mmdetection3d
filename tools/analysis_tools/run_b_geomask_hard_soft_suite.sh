#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN="${PRETRAIN:-}"
CFG_OPTS=(
  "val_evaluator.version=v1.0-mini"
  "test_evaluator.version=v1.0-mini"
)
if [[ -n "$PRETRAIN" ]]; then
  CFG_OPTS+=("load_from=${PRETRAIN}")
fi

CFG_B="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_nus-3d.py"
CFG_BM_HARD="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_on_nus-3d.py"
CFG_BM_SOFT="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_nus-3d.py"

WD_B="work_dirs/abmini_cross_b"
WD_BM_HARD="work_dirs/abmini_cross_b_geomask"
WD_BM_SOFT="work_dirs/abmini_cross_b_geomask_soft"

echo "[1/4] Train B baseline"
python tools/train.py "$CFG_B" --work-dir "$WD_B" --cfg-options "${CFG_OPTS[@]}"

echo "[2/4] Train B+M hard mask"
python tools/train.py "$CFG_BM_HARD" --work-dir "$WD_BM_HARD" --cfg-options "${CFG_OPTS[@]}"

echo "[3/4] Train B+M soft mask"
python tools/train.py "$CFG_BM_SOFT" --work-dir "$WD_BM_SOFT" --cfg-options "${CFG_OPTS[@]}"

echo "[4/4] Compare B vs hard vs soft"
python tools/analysis_tools/compare_ab_matched_ious.py \
  --exp-a "$WD_B" \
  --exp-b "$WD_BM_HARD" \
  --exp-c "$WD_BM_SOFT" \
  --max-iter 500
