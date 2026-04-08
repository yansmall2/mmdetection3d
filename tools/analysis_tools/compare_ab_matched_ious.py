#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Optional


METRIC_KEYS = [
    "matched_ious",
    "matched_iou_max",
    "layer_-1_loss_bbox",
    "layer_-1_loss_cls",
    "assign_num_pos",
    "dbg_query_heat_mean",
    "dbg_query_obj_mean",
    "dbg_geo_mask_ratio",
    "dbg_geo_mask_mean",
    "dbg_geo_mask_nonzero",
]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_scalars(work_dir: Path):
    cands = sorted(work_dir.glob("**/vis_data/scalars.json"))
    if cands:
        return cands[-1]
    cands = sorted(work_dir.glob("**/vis_data/*.json"))
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"no vis_data json found under: {work_dir}")


def summarize(rows):
    summary = {}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            summary[key] = {
                "mean": mean(vals),
                "min": min(vals),
                "max": max(vals),
                "last": vals[-1],
            }
    return summary, len(rows)


def _scan_metrics_from_rows(rows: Iterable[dict]) -> Dict[str, float]:
    out = {}
    map_candidates = []
    nds_candidates = []
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            key_l = key.lower()
            if (
                ("map" in key_l or "mean_ap" in key_l)
                and "heatmap" not in key_l
            ):
                map_candidates.append(float(value))
            if "nds" in key_l or "nd_score" in key_l:
                nds_candidates.append(float(value))
    if map_candidates:
        out["bbox_mAP"] = map_candidates[-1]
    if nds_candidates:
        out["NDS"] = nds_candidates[-1]
    return out


def _scan_metrics_from_text(text: str) -> Dict[str, float]:
    out = {}
    patterns = {
        "bbox_mAP": [
            r"(?:bbox_mAP|mean_ap|(?:^|[/_])mAP)\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ],
        "NDS": [
            r"(?:\bNDS\b|nd_score)\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ],
    }
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                out[metric] = float(matches[-1])
                break
    return out


def _read_metrics_summary_json(work_dir: Path) -> Optional[Dict[str, float]]:
    cands = sorted(work_dir.glob("**/metrics_summary.json"))
    if not cands:
        return None
    try:
        data = json.loads(cands[-1].read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    out = {}
    if "mean_ap" in data:
        out["bbox_mAP"] = float(data["mean_ap"])
    if "nd_score" in data:
        out["NDS"] = float(data["nd_score"])
    return out or None


def find_last_val_metrics(work_dir: Path):
    # Priority 1: official NuScenes metrics summary json (most reliable).
    metrics = _read_metrics_summary_json(work_dir)
    if metrics:
        return metrics

    # Priority 2: vis_data json rows.
    vis_jsons = sorted(work_dir.glob("**/vis_data/*.json"))
    for path in reversed(vis_jsons):
        rows = load_jsonl(path)
        metrics = _scan_metrics_from_rows(rows)
        if metrics:
            return metrics

    # Priority 3: plain-text log parsing.
    log_files = sorted(work_dir.glob("**/*.log"))
    if not log_files:
        return {}
    text = log_files[-1].read_text(encoding="utf-8", errors="ignore")
    return _scan_metrics_from_text(text)


def print_summary(name, section, summary, n):
    print(f"\n[{section}]")
    print(f"rows_used: {n}")
    for key in METRIC_KEYS:
        if key not in summary:
            continue
        s = summary[key]
        print(
            f"{key:20s} mean={s['mean']:.6f} min={s['min']:.6f} "
            f"max={s['max']:.6f} last={s['last']:.6f}"
        )


def print_exp(name, section_summaries, val_metrics):
    print(f"\n=== {name} ===")
    for sec_name, (summary, n) in section_summaries.items():
        print_summary(name, sec_name, summary, n)
    if val_metrics:
        print("val_metrics:", ", ".join(f"{k}={v:.6f}" for k, v in val_metrics.items()))
    else:
        print("val_metrics: not found")


def split_sections(rows, max_iter):
    all_rows = [r for r in rows if "iter" in r]
    all_rows.sort(key=lambda x: int(x["iter"]))
    head = [r for r in all_rows if int(r["iter"]) <= max_iter]
    if all_rows:
        max_seen_iter = max(int(r["iter"]) for r in all_rows)
        tail_start = max(1, max_seen_iter - max_iter + 1)
        tail = [r for r in all_rows if int(r["iter"]) >= tail_start]
    else:
        tail = []
    return {
        "first_window": summarize(head),
        "last_window": summarize(tail),
        "full_run": summarize(all_rows),
    }


def matched_iou_mean(section_tuple):
    summary, _ = section_tuple
    if "matched_ious" not in summary:
        return None
    return summary["matched_ious"]["mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-a", required=True, help="work_dir for A")
    parser.add_argument("--exp-b", required=True, help="work_dir for B")
    parser.add_argument("--exp-c", default=None, help="optional work_dir for C")
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    exp_inputs = [("A", Path(args.exp_a)), ("B", Path(args.exp_b))]
    if args.exp_c:
        exp_inputs.append(("C", Path(args.exp_c)))

    exp_data = {}
    for name, exp_dir in exp_inputs:
        rows = load_jsonl(find_scalars(exp_dir))
        sections = split_sections(rows, args.max_iter)
        val = find_last_val_metrics(exp_dir)
        exp_data[name] = dict(sections=sections, val=val)
        print_exp(name, sections, val)

    # Delta report focused on mean matched_ious.
    print("\n=== Delta (mean matched_ious) ===")
    section_names = ["first_window", "last_window", "full_run"]
    pairs = [("A", "B")]
    if "C" in exp_data:
        pairs.extend([("A", "C"), ("B", "C")])
    for s in section_names:
        print(f"[{s}]")
        for x, y in pairs:
            xv = matched_iou_mean(exp_data[x]["sections"][s])
            yv = matched_iou_mean(exp_data[y]["sections"][s])
            if xv is None or yv is None:
                print(f"  {x}-{y}: n/a")
                continue
            print(f"  {x}-{y}: {xv - yv:+.6f}")


if __name__ == "__main__":
    main()
