#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Optional override for local pretrained checkpoint to avoid network failure.
# Example:
#   PRETRAIN=/path/to/bevfusion_lidar_voxel...pth bash tools/analysis_tools/run_b_geomask_suite.sh
PRETRAIN="${PRETRAIN:-}"
CFG_OPTS=(
  "val_evaluator.version=v1.0-mini"
  "test_evaluator.version=v1.0-mini"
)
if [[ -n "$PRETRAIN" ]]; then
  CFG_OPTS+=("load_from=${PRETRAIN}")
fi

CFG_B="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_nus-3d.py"
CFG_BM="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_on_nus-3d.py"

WD_B="work_dirs/abmini_cross_b"
WD_BM="work_dirs/abmini_cross_b_geomask"

echo "[1/3] Train B (CrossAttentionFuser baseline)"
python tools/train.py "$CFG_B" --work-dir "$WD_B" --cfg-options "${CFG_OPTS[@]}"

echo "[2/3] Train B+M (Geometry-mask + CrossAttentionFuser)"
python tools/train.py "$CFG_BM" --work-dir "$WD_BM" --cfg-options "${CFG_OPTS[@]}"

echo "[3/3] Compare B vs B+M (first/last/full windows)"
python tools/analysis_tools/compare_ab_matched_ious.py \
  --exp-a "$WD_B" \
  --exp-b "$WD_BM" \
  --max-iter 500
