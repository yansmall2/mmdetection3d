#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN="${PRETRAIN:-}"
SEED="${SEED:-3409}"
PILOT_TAG="${PILOT_TAG:-ow_pilot_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-work_dirs/${PILOT_TAG}}"
HOLDOUT_CLASSES_CFG="${HOLDOUT_CLASSES_CFG:-['construction_vehicle']}"
UNKNOWN_OBJ_THRESH="${UNKNOWN_OBJ_THRESH:-0.5}"
UNKNOWN_CLS_THRESH="${UNKNOWN_CLS_THRESH:-0.3}"
UNKNOWN_LABEL_ID="${UNKNOWN_LABEL_ID:-10}"
UNKNOWN_IOU_THR="${UNKNOWN_IOU_THR:-0.25}"
UNKNOWN_SCORE_THR="${UNKNOWN_SCORE_THR:-0.1}"

if [[ -z "$PRETRAIN" ]]; then
  echo "[ERROR] PRETRAIN is empty. Please set PRETRAIN=/abs/path/to/bevfusion_ckpt.pth"
  exit 2
fi

mkdir -p "$RUN_ROOT"
MANIFEST="$RUN_ROOT/manifest.tsv"
{
  echo -e "exp\tconfig\twork_dir\tcheckpoint\ttrain_status\ttrain_sec\tknown_status\tknown_sec\topen_status\topen_sec"
} > "$MANIFEST"

COMMON_CFG_OPTS=(
  "val_dataloader.dataset.metainfo.version=v1.0-mini"
  "test_dataloader.dataset.metainfo.version=v1.0-mini"
  "load_from=${PRETRAIN}"
  "randomness.seed=${SEED}"
  "train_dataloader.dataset.dataset.holdout_classes=${HOLDOUT_CLASSES_CFG}"
  "val_evaluator.holdout_classes=${HOLDOUT_CLASSES_CFG}"
  "test_evaluator.holdout_classes=${HOLDOUT_CLASSES_CFG}"
  "val_evaluator.unknown_iou_thr=${UNKNOWN_IOU_THR}"
  "test_evaluator.unknown_iou_thr=${UNKNOWN_IOU_THR}"
  "val_evaluator.unknown_score_thr=${UNKNOWN_SCORE_THR}"
  "test_evaluator.unknown_score_thr=${UNKNOWN_SCORE_THR}"
  "val_evaluator.unknown_label_id=${UNKNOWN_LABEL_ID}"
  "test_evaluator.unknown_label_id=${UNKNOWN_LABEL_ID}"
)

CFG_B="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_nus-3d.py"
CFG_BM_SOFT02="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_lam02_nus-3d.py"
CFG_BM_SOFT05="projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_nus-3d.py"

WD_B="$RUN_ROOT/B_baseline"
WD_BM_SOFT02="$RUN_ROOT/BM_soft_lam02"
WD_BM_SOFT05="$RUN_ROOT/BM_soft_lam05"

exp_names=("B" "BM_soft_lam02" "BM_soft_lam05")
exp_cfgs=("$CFG_B" "$CFG_BM_SOFT02" "$CFG_BM_SOFT05")
exp_wds=("$WD_B" "$WD_BM_SOFT02" "$WD_BM_SOFT05")

find_checkpoint() {
  local work_dir="$1"
  if [[ -f "$work_dir/latest.pth" ]]; then
    echo "$work_dir/latest.pth"
    return 0
  fi
  local ckpt
  ckpt="$(ls -1 "$work_dir"/epoch_*.pth 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "$ckpt" ]]; then
    echo "$ckpt"
  fi
}

run_eval_mode() {
  local cfg="$1"
  local ckpt="$2"
  local eval_dir="$3"
  local mode="$4"
  local export_dir="$eval_dir/nusc_exports"

  mkdir -p "$eval_dir"
  mkdir -p "$export_dir"
  python tools/test.py "$cfg" "$ckpt" \
    --work-dir "$eval_dir" \
    --cfg-options \
      "${COMMON_CFG_OPTS[@]}" \
      "model.bbox_head.test_cfg.open_world_mode=${mode}" \
      "model.bbox_head.test_cfg.unknown_obj_thresh=${UNKNOWN_OBJ_THRESH}" \
      "model.bbox_head.test_cfg.unknown_cls_thresh=${UNKNOWN_CLS_THRESH}" \
      "model.bbox_head.test_cfg.unknown_label_id=${UNKNOWN_LABEL_ID}" \
      "test_evaluator.jsonfile_prefix=${export_dir}"
}

for i in "${!exp_names[@]}"; do
  name="${exp_names[$i]}"
  cfg="${exp_cfgs[$i]}"
  wd="${exp_wds[$i]}"

  echo "[RUN] ${name} train"
  train_status="ok"
  known_status="skipped"
  open_status="skipped"

  t0="$(date +%s)"
  if ! python tools/train.py "$cfg" --work-dir "$wd" --cfg-options "${COMMON_CFG_OPTS[@]}"; then
    train_status="failed"
  fi
  t1="$(date +%s)"
  train_sec=$((t1 - t0))

  ckpt="$(find_checkpoint "$wd")"
  if [[ -z "${ckpt:-}" ]]; then
    echo "[WARN] ${name}: checkpoint not found, skip evaluation"
    known_sec=0
    open_sec=0
    echo -e "${name}\t${cfg}\t${wd}\t\t${train_status}\t${train_sec}\t${known_status}\t${known_sec}\t${open_status}\t${open_sec}" >> "$MANIFEST"
    continue
  fi

  vis_json="$(find "$wd" -type f -path "*/vis_data/*.json" | head -n 1 || true)"
  if [[ -z "$vis_json" ]]; then
    echo "[WARN] ${name}: vis_data json missing; train comparison may be unavailable"
  fi

  echo "[RUN] ${name} eval known-only"
  k0="$(date +%s)"
  if run_eval_mode "$cfg" "$ckpt" "$wd/eval_known_only" "known_only"; then
    known_status="ok"
  else
    known_status="failed"
  fi
  k1="$(date +%s)"
  known_sec=$((k1 - k0))

  echo "[RUN] ${name} eval open-world"
  o0="$(date +%s)"
  if run_eval_mode "$cfg" "$ckpt" "$wd/eval_open_world" "open_world"; then
    open_status="ok"
  else
    open_status="failed"
  fi
  o1="$(date +%s)"
  open_sec=$((o1 - o0))

  echo -e "${name}\t${cfg}\t${wd}\t${ckpt}\t${train_status}\t${train_sec}\t${known_status}\t${known_sec}\t${open_status}\t${open_sec}" >> "$MANIFEST"
done

if [[ -d "$WD_B" && -d "$WD_BM_SOFT02" && -d "$WD_BM_SOFT05" ]]; then
  HAVE_A="$(find "$WD_B" -type f -path "*/vis_data/*.json" | head -n 1 || true)"
  HAVE_B="$(find "$WD_BM_SOFT02" -type f -path "*/vis_data/*.json" | head -n 1 || true)"
  HAVE_C="$(find "$WD_BM_SOFT05" -type f -path "*/vis_data/*.json" | head -n 1 || true)"
  if [[ -n "$HAVE_A" && -n "$HAVE_B" && -n "$HAVE_C" ]]; then
    echo "[RUN] Compare first/last windows for B vs BM-soft(0.2/0.5)"
    python tools/analysis_tools/compare_ab_matched_ious.py \
      --exp-a "$WD_B" \
      --exp-b "$WD_BM_SOFT02" \
      --exp-c "$WD_BM_SOFT05" \
      --max-iter 500 | tee "$RUN_ROOT/compare_train_windows.txt"
  else
    echo "[WARN] Skip compare_ab_matched_ious.py because vis_data json is missing in one or more runs"
  fi
fi

python tools/analysis_tools/summarize_openworld_pilot.py \
  --manifest "$MANIFEST" \
  --output-csv "$RUN_ROOT/pilot_summary.csv" \
  --output-md "$RUN_ROOT/pilot_summary.md"

echo "[DONE] Pilot suite finished."
echo "manifest: $MANIFEST"
echo "summary : $RUN_ROOT/pilot_summary.csv"
echo "report  : $RUN_ROOT/pilot_summary.md"
