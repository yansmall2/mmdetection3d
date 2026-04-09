#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CFG_OPTS=(
  "val_dataloader.dataset.metainfo.version=v1.0-mini"
  "test_dataloader.dataset.metainfo.version=v1.0-mini"
)

CONV_CFG="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_conv_nus-3d.py"
CROSS_CFG="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_nus-3d.py"
CONV_WD="work_dirs/abmini_conv"
CROSS_WD="work_dirs/abmini_cross"

echo "[1/3] Run A (ConvFuser)"
python tools/train.py "$CONV_CFG" --work-dir "$CONV_WD" --cfg-options "${CFG_OPTS[@]}"

echo "[2/3] Run B (CrossAttentionFuser)"
python tools/train.py "$CROSS_CFG" --work-dir "$CROSS_WD" --cfg-options "${CFG_OPTS[@]}"

echo "[3/3] Compare first 500 iters and val metrics"
python tools/analysis_tools/compare_ab_matched_ious.py \
  --exp-a "$CONV_WD" \
  --exp-b "$CROSS_WD" \
  --max-iter 500
