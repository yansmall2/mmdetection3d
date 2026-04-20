#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_MAP_KEYS = {'mean_ap', 'meanap', 'bbox_map', 'map'}
_NDS_KEYS = {'nd_score', 'ndscore', 'nds'}


def parse_manifest(manifest: Path):
    rows: List[Dict[str, str]] = []
    with manifest.open('r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        if not header or all(not col for col in header):
            return rows
        required = {'exp', 'work_dir'}
        missing = required.difference(header)
        if missing:
            raise ValueError(
                f'manifest is missing required columns: {sorted(missing)}')
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            row = {k: (parts[i] if i < len(parts) else '') for i, k in enumerate(header)}
            rows.append(row)
    return rows


def find_latest_json(root: Path, name: str) -> Optional[Path]:
    if not root.exists():
        return None
    cands = [p for p in root.glob(f'**/{name}') if p.is_file()]
    if not cands:
        return None
    return max(cands, key=_safe_mtime)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _walk_numbers(obj, max_depth: int = 6):
    """Yield (key, value) pairs for all numeric leaves in a nested mapping."""
    if max_depth < 0:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                yield str(k), float(v)
            else:
                yield from _walk_numbers(v, max_depth=max_depth - 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_numbers(v, max_depth=max_depth - 1)


def _is_valid_metric(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def _extract_metrics_from_json_dict(data: dict) -> Tuple[Optional[float], Optional[float]]:
    # Many runners wrap metrics in nested dicts (e.g. {"metric": {...}}).
    # Prefer exact key matches, then fall back to composite names like
    # "NuScenes/bbox_mAP" or "val/nd_score".
    exact_map = None
    exact_nds = None
    fuzzy_map = None
    fuzzy_nds = None
    for k, v in _walk_numbers(data):
        k_low = k.strip().lower()
        if k_low in _MAP_KEYS and _is_valid_metric(v):
            exact_map = v
        elif k_low in _NDS_KEYS and _is_valid_metric(v):
            exact_nds = v
        else:
            if _is_valid_metric(v) and ('map' in k_low or 'mean_ap' in k_low) and 'heatmap' not in k_low:
                fuzzy_map = v
            if _is_valid_metric(v) and ('nds' in k_low or 'nd_score' in k_low):
                fuzzy_nds = v
    return (
        exact_map if exact_map is not None else fuzzy_map,
        exact_nds if exact_nds is not None else fuzzy_nds,
    )


def _read_json_metrics(path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return None, None
    if isinstance(data, dict):
        return _extract_metrics_from_json_dict(data)
    return None, None


def _scan_metrics_from_logs(eval_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    # Fallback: parse plain-text logs.
    if not eval_dir.exists():
        return None, None
    log_files = sorted(eval_dir.glob('**/*.log'), key=_safe_mtime)
    if not log_files:
        log_files = sorted(eval_dir.glob('**/*.txt'), key=_safe_mtime)
    if not log_files:
        return None, None
    patterns = {
        'mAP': [
            r'(?:bbox_mAP|bbox_map|mean_ap|\bmAP\b)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        ],
        'NDS': [
            r'(?:\bNDS\b|nd_score)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        ],
    }
    for log_path in reversed(log_files):
        text = log_path.read_text(encoding='utf-8', errors='ignore')
        m_ap = None
        nds = None
        for pat in patterns['mAP']:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            if matches:
                m_ap = float(matches[-1])
                break
        for pat in patterns['NDS']:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            if matches:
                nds = float(matches[-1])
                break
        if m_ap is not None or nds is not None:
            return m_ap, nds
    return None, None


def read_metrics_summary(eval_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    if not eval_dir.exists():
        return None, None

    # 1) Exact expected filename.
    path = find_latest_json(eval_dir, 'metrics_summary.json')
    if path is not None:
        m_ap, nds = _read_json_metrics(path)
        if m_ap is not None or nds is not None:
            return m_ap, nds

    # 2) Any json that contains mean_ap/nd_score (handles different output layouts).
    candidates = []
    for p in eval_dir.glob('**/*.json'):
        try:
            # Skip huge prediction dumps.
            if p.stat().st_size > 50 * 1024 * 1024:
                continue
        except OSError:
            continue
        m_ap, nds = _read_json_metrics(p)
        if m_ap is not None or nds is not None:
            candidates.append((_safe_mtime(p), m_ap, nds))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, m_ap, nds = candidates[-1]
        return m_ap, nds

    # 3) Log parsing fallback.
    return _scan_metrics_from_logs(eval_dir)


def read_unknown_counts(eval_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    if not eval_dir.exists():
        return None, None

    # Common names used by mmdet3d test scripts / NuScenes format exporters.
    path = (
        find_latest_json(eval_dir, 'results_nusc_with_unknown.json')
        or find_latest_json(eval_dir, 'analysis_results_nusc.json')
        or find_latest_json(eval_dir, 'results_nusc.json')
        or find_latest_json(eval_dir, 'results.json')
        or find_latest_json(eval_dir, 'pred_instances_3d.json')
    )
    if path is None:
        # Last resort: search a reasonable set of json files for NuScenes-format outputs.
        for p in sorted(eval_dir.glob('**/*.json'), key=_safe_mtime, reverse=True):
            if 'result' not in p.name.lower():
                continue
            try:
                if p.stat().st_size > 200 * 1024 * 1024:
                    continue
            except OSError:
                continue
            path = p
            break
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None

    results = data.get('results', {})
    if not isinstance(results, dict):
        return None, None

    total = 0
    unknown = 0
    for _, boxes in results.items():
        if not isinstance(boxes, list):
            continue
        total += len(boxes)
        for box in boxes:
            if not isinstance(box, dict):
                continue
            label = str(
                box.get('detection_name')
                or box.get('name')
                or box.get('label')
                or ''
            ).lower()
            if label == 'unknown':
                unknown += 1
    return unknown, total


def read_open_world_metrics(
        eval_dir: Path) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    if not eval_dir.exists():
        return None, None, None

    path = find_latest_json(eval_dir, 'open_world_metrics.json')
    if path is None:
        return None, None, None
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None

    recall = parse_float(str(data.get('unknown_recall', '')))
    gt_count = parse_int(str(data.get('unknown_gt_count', '')))
    pred_count = parse_int(str(data.get('unknown_pred_count', '')))
    return recall, gt_count, pred_count


def fmt_float(x: Optional[float]) -> str:
    return '' if x is None else f'{x:.6f}'


def fmt_int(x: Optional[int]) -> str:
    return '' if x is None else str(x)


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if value == '':
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: str) -> Optional[int]:
    value = value.strip()
    if value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt_ratio(numer: Optional[int], denom: Optional[int]) -> str:
    if numer is None or denom in (None, 0):
        return ''
    return f'{numer}/{denom} ({numer / denom:.2%})'


def unknown_summary(row) -> str:
    return fmt_ratio(
        parse_int(row.get('open_unknown_count', '')),
        parse_int(row.get('open_total_boxes', '')),
    )


def _status_count(rows, key: str, target: str) -> int:
    return sum(1 for row in rows if row.get(key) == target)


def _best_row(rows, key: str):
    scored_rows = []
    for row in rows:
        score = parse_float(row.get(key, ''))
        if score is not None:
            scored_rows.append((score, row))
    if not scored_rows:
        return None
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    return scored_rows[0][1]


def fmt_status(status: str, seconds: str) -> str:
    status = status or ''
    seconds = seconds.strip()
    if not status:
        return ''
    if not seconds:
        return status
    return f'{status} ({seconds}s)'


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
        work_dir_value = row.get('work_dir', '').strip()
        known_map = None
        known_nds = None
        open_map = None
        open_nds = None
        unk_count = None
        total_count = None
        unk_recall = None
        unk_gt_count = None
        unk_pred_count = None
        if work_dir_value:
            wd = Path(work_dir_value)
            known_dir = wd / 'eval_known_only'
            open_dir = wd / 'eval_open_world'
            known_map, known_nds = read_metrics_summary(known_dir)
            open_map, open_nds = read_metrics_summary(open_dir)
            unk_count, total_count = read_unknown_counts(open_dir)
            unk_recall, unk_gt_count, unk_pred_count = read_open_world_metrics(
                open_dir)

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
            'open_unknown_recall': fmt_float(unk_recall),
            'open_unknown_gt_count': fmt_int(unk_gt_count),
            'open_unknown_pred_count': fmt_int(unk_pred_count),
        })

    fieldnames = [
        'exp', 'config', 'work_dir', 'checkpoint',
        'train_status', 'train_sec',
        'known_status', 'known_sec', 'known_bbox_mAP', 'known_NDS',
        'open_status', 'open_sec', 'open_bbox_mAP', 'open_NDS',
        'open_unknown_count', 'open_total_boxes',
        'open_unknown_recall', 'open_unknown_gt_count', 'open_unknown_pred_count'
    ]

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open('w', encoding='utf-8') as f:
        f.write('# Open-World Pilot Summary\n\n')
        f.write(f'- manifest: `{manifest}`\n')
        f.write(f'- total experiments: **{len(out_rows)}**\n\n')

        f.write('## Status Overview\n\n')
        f.write(f'- train ok: **{_status_count(out_rows, "train_status", "ok")}** / {len(out_rows)}\n')
        f.write(f'- known eval ok: **{_status_count(out_rows, "known_status", "ok")}** / {len(out_rows)}\n')
        f.write(f'- open eval ok: **{_status_count(out_rows, "open_status", "ok")}** / {len(out_rows)}\n\n')

        best_known_map = _best_row(out_rows, 'known_bbox_mAP')
        best_known_nds = _best_row(out_rows, 'known_NDS')
        best_open_map = _best_row(out_rows, 'open_bbox_mAP')
        best_open_nds = _best_row(out_rows, 'open_NDS')
        best_unknown = _best_row(out_rows, 'open_unknown_count')
        best_unknown_recall = _best_row(out_rows, 'open_unknown_recall')

        f.write('## Best Results\n\n')
        if best_known_map is not None:
            f.write(
                f'- best known mAP: **{best_known_map["exp"]}** '
                f'({best_known_map["known_bbox_mAP"]})\n'
            )
        if best_known_nds is not None:
            f.write(
                f'- best known NDS: **{best_known_nds["exp"]}** '
                f'({best_known_nds["known_NDS"]})\n'
            )
        if best_open_map is not None:
            f.write(
                f'- best open mAP: **{best_open_map["exp"]}** '
                f'({best_open_map["open_bbox_mAP"]})\n'
            )
        if best_open_nds is not None:
            f.write(
                f'- best open NDS: **{best_open_nds["exp"]}** '
                f'({best_open_nds["open_NDS"]})\n'
            )
        if best_unknown_recall is not None:
            recall_text = best_unknown_recall['open_unknown_recall']
            gt_text = best_unknown_recall.get('open_unknown_gt_count', '')
            pred_text = best_unknown_recall.get('open_unknown_pred_count', '')
            suffix = ''
            if gt_text or pred_text:
                suffix = f' [gt={gt_text or ""}, pred={pred_text or ""}]'
            f.write(
                f'- best unknown recall: **{best_unknown_recall["exp"]}** '
                f'({recall_text}){suffix}\n'
            )
        best_unknown_text = '' if best_unknown is None else unknown_summary(best_unknown)
        if best_unknown is not None and best_unknown_text:
            f.write(
                f'- most unknown predictions: **{best_unknown["exp"]}** '
                f'({best_unknown_text})\n'
            )
        f.write('\n')

        f.write('## Per-Experiment Table\n\n')
        f.write('| exp | train | known | open | known mAP | known NDS | open mAP | open NDS | unknown recall | unknown/open total |\n')
        f.write('|---|---|---|---|---:|---:|---:|---:|---:|---:|\n')
        for r in out_rows:
            unknown_ratio = unknown_summary(r)
            unknown_recall = r.get('open_unknown_recall', '')
            f.write(
                f"| {r['exp']} | {fmt_status(r['train_status'], r['train_sec'])} | "
                f"{fmt_status(r['known_status'], r['known_sec'])} | "
                f"{fmt_status(r['open_status'], r['open_sec'])} | "
                f"{r['known_bbox_mAP']} | {r['known_NDS']} | "
                f"{r['open_bbox_mAP']} | {r['open_NDS']} | {unknown_recall} | {unknown_ratio} |\n"
            )

    print(f'[OK] wrote {out_csv}')
    print(f'[OK] wrote {out_md}')


if __name__ == '__main__':
    main()
