#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_manifest(manifest: Path):
    rows = []
    with manifest.open('r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            row = {k: (parts[i] if i < len(parts) else '') for i, k in enumerate(header)}
            rows.append(row)
    return rows


def find_latest_json(root: Path, name: str) -> Optional[Path]:
    cands = sorted(root.glob(f'**/{name}'))
    return cands[-1] if cands else None


def read_metrics_summary(eval_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    path = find_latest_json(eval_dir, 'metrics_summary.json')
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return None, None
    m_ap = data.get('mean_ap')
    nds = data.get('nd_score')
    return (float(m_ap) if m_ap is not None else None,
            float(nds) if nds is not None else None)


def read_unknown_counts(eval_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    path = find_latest_json(eval_dir, 'results_nusc.json')
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return None, None
    results = data.get('results', {})
    total = 0
    unknown = 0
    for _, boxes in results.items():
        if not isinstance(boxes, list):
            continue
        total += len(boxes)
        for box in boxes:
            if str(box.get('detection_name', '')).lower() == 'unknown':
                unknown += 1
    return unknown, total


def fmt_float(x: Optional[float]) -> str:
    return '' if x is None else f'{x:.6f}'


def fmt_int(x: Optional[int]) -> str:
    return '' if x is None else str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = parse_manifest(manifest)

    out_rows = []
    for row in rows:
        wd = Path(row['work_dir'])
        known_dir = wd / 'eval_known_only'
        open_dir = wd / 'eval_open_world'

        known_map, known_nds = read_metrics_summary(known_dir)
        open_map, open_nds = read_metrics_summary(open_dir)
        unk_count, total_count = read_unknown_counts(open_dir)

        out_rows.append({
            'exp': row.get('exp', ''),
            'config': row.get('config', ''),
            'work_dir': row.get('work_dir', ''),
            'checkpoint': row.get('checkpoint', ''),
            'train_status': row.get('train_status', ''),
            'train_sec': row.get('train_sec', ''),
            'known_status': row.get('known_status', ''),
            'known_sec': row.get('known_sec', ''),
            'known_bbox_mAP': fmt_float(known_map),
            'known_NDS': fmt_float(known_nds),
            'open_status': row.get('open_status', ''),
            'open_sec': row.get('open_sec', ''),
            'open_bbox_mAP': fmt_float(open_map),
            'open_NDS': fmt_float(open_nds),
            'open_unknown_count': fmt_int(unk_count),
            'open_total_boxes': fmt_int(total_count),
        })

    fieldnames = [
        'exp', 'config', 'work_dir', 'checkpoint',
        'train_status', 'train_sec',
        'known_status', 'known_sec', 'known_bbox_mAP', 'known_NDS',
        'open_status', 'open_sec', 'open_bbox_mAP', 'open_NDS',
        'open_unknown_count', 'open_total_boxes'
    ]

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    out_md = Path(args.output_md)
    with out_md.open('w', encoding='utf-8') as f:
        f.write('# Open-World Pilot Summary\n\n')
        f.write(f'- manifest: `{manifest}`\n')
        f.write(f'- total experiments: **{len(out_rows)}**\n\n')
        f.write('| exp | train | known | open | known mAP | known NDS | open mAP | open NDS | unknown/open total |\n')
        f.write('|---|---|---|---|---:|---:|---:|---:|---:|\n')
        for r in out_rows:
            unknown_ratio = ''
            if r['open_unknown_count'] and r['open_total_boxes']:
                unknown_ratio = f"{r['open_unknown_count']}/{r['open_total_boxes']}"
            f.write(
                f"| {r['exp']} | {r['train_status']} ({r['train_sec']}s) | "
                f"{r['known_status']} ({r['known_sec']}s) | "
                f"{r['open_status']} ({r['open_sec']}s) | "
                f"{r['known_bbox_mAP']} | {r['known_NDS']} | "
                f"{r['open_bbox_mAP']} | {r['open_NDS']} | {unknown_ratio} |\n"
            )

    print(f'[OK] wrote {out_csv}')
    print(f'[OK] wrote {out_md}')


if __name__ == '__main__':
    main()
