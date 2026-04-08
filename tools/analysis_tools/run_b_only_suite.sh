#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Optional override for local pretrained checkpoint to avoid network failure.
# Example:
#   PRETRAIN=/path/to/bevfusion_lidar_voxel...pth bash tools/analysis_tools/run_b_only_suite.sh
PRETRAIN="${PRETRAIN:-}"
CFG_OPTS=(
  "val_evaluator.version=v1.0-mini"
  "test_evaluator.version=v1.0-mini"
)
if [[ -n "$PRETRAIN" ]]; then
  CFG_OPTS+=("load_from=${PRETRAIN}")
fi

CFG_B="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_nus-3d.py"
CFG_B05="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_fusionlr05_nus-3d.py"
CFG_B02="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_fusionlr02_nus-3d.py"

WD_B="work_dirs/abmini_cross_b"
WD_B05="work_dirs/abmini_cross_b_lr05"
WD_B02="work_dirs/abmini_cross_b_lr02"

echo "[1/4] Train B (default)"
python tools/train.py "$CFG_B" --work-dir "$WD_B" --cfg-options "${CFG_OPTS[@]}"

echo "[2/4] Train B (fusion lr x0.5)"
python tools/train.py "$CFG_B05" --work-dir "$WD_B05" --cfg-options "${CFG_OPTS[@]}"

echo "[3/4] Train B (fusion lr x0.2)"
python tools/train.py "$CFG_B02" --work-dir "$WD_B02" --cfg-options "${CFG_OPTS[@]}"

echo "[4/4] Compare B variants (first/last/full windows)"
python tools/analysis_tools/compare_ab_matched_ious.py \
  --exp-a "$WD_B" \
  --exp-b "$WD_B05" \
  --exp-c "$WD_B02" \
  --max-iter 500
