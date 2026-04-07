#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from statistics import mean


METRIC_KEYS = [
    "matched_ious",
    "matched_iou_max",
    "layer_-1_loss_bbox",
    "layer_-1_loss_cls",
    "assign_num_pos",
    "dbg_query_heat_mean",
    "dbg_query_obj_mean",
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


def summarize(rows, max_iter):
    rows = [r for r in rows if int(r.get("iter", 10**9)) <= max_iter]
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


def find_last_val_metrics(work_dir: Path):
    log_files = sorted(work_dir.glob("**/*.log"))
    if not log_files:
        return {}
    text = log_files[-1].read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "bbox_mAP": r"\bbbox_mAP\b[^0-9\-]*([0-9]*\.?[0-9]+)",
        "NDS": r"\bNDS\b[^0-9\-]*([0-9]*\.?[0-9]+)",
    }
    out = {}
    for k, p in patterns.items():
        matches = re.findall(p, text)
        if matches:
            out[k] = float(matches[-1])
    return out


def print_summary(name, summary, n, val_metrics):
    print(f"\n=== {name} ===")
    print(f"rows_used: {n}")
    for key in METRIC_KEYS:
        if key not in summary:
            continue
        s = summary[key]
        print(
            f"{key:20s} mean={s['mean']:.6f} min={s['min']:.6f} "
            f"max={s['max']:.6f} last={s['last']:.6f}"
        )
    if val_metrics:
        print("val_metrics:", ", ".join(f"{k}={v:.6f}" for k, v in val_metrics.items()))
    else:
        print("val_metrics: not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-a", required=True, help="work_dir for A")
    parser.add_argument("--exp-b", required=True, help="work_dir for B")
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    a_dir = Path(args.exp_a)
    b_dir = Path(args.exp_b)

    a_rows = load_jsonl(find_scalars(a_dir))
    b_rows = load_jsonl(find_scalars(b_dir))

    a_summary, a_n = summarize(a_rows, args.max_iter)
    b_summary, b_n = summarize(b_rows, args.max_iter)
    a_val = find_last_val_metrics(a_dir)
    b_val = find_last_val_metrics(b_dir)

    print_summary("A", a_summary, a_n, a_val)
    print_summary("B", b_summary, b_n, b_val)

    if "matched_ious" in a_summary and "matched_ious" in b_summary:
        delta = a_summary["matched_ious"]["mean"] - b_summary["matched_ious"]["mean"]
        print(f"\nDelta(mean matched_ious, A-B): {delta:.6f}")


if __name__ == "__main__":
    main()
