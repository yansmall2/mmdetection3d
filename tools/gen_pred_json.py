"""Generate prediction JSON using proper test pipeline (no augmentations).
Usage: source activate mm3d_5060 && python tools/gen_pred_json.py [--split val|train]

Output: JSON file in nuscenes submission format with 'unknown' class support.
"""

import argparse, os, sys
os.chdir('/home/yan/mmdetection3d')
# Keep original argv for argparse
_real_argv = sys.argv.copy()
sys.argv = ['test.py']  # Satisfy mmengine entry-point detection

import json, copy, torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader

import projects.BEVFusion.bevfusion  # register modules

CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def main():
    parser = argparse.ArgumentParser()
    # Parse from original argv (skip script name and 'test.py' override)
    parser.add_argument('--config', type=str,
                        default='projects/BEVFusion/configs/bevfusion_lidar-cam_abmini_cross_geomask_soft_lam03_nus-3d.py')
    parser.add_argument('--ckpt', type=str,
                        default='work_dirs/lam03_20260508_212031/epoch_4.pth')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=0,
                        help='Max samples (0=all)')
    args = parser.parse_args(_real_argv[1:])

    if args.out is None:
        args.out = f'work_dirs/lam03_20260508_212031/{args.split}_preds.json'

    cfg = Config.fromfile(args.config)
    cfg.work_dir = '/tmp/gen_json_final'
    cfg.model.train_cfg = None

    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, args.ckpt, map_location='cuda')
    model = runner.model.cuda().eval()

    # For val split: use existing val_dataloader dataset (already has test pipeline)
    # For train split: build train dataset with val pipeline
    if args.split == 'val':
        ds = runner.val_dataloader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset
    else:
        train_inner_cfg = copy.deepcopy(cfg.train_dataloader.dataset.dataset)
        val_pipeline = copy.deepcopy(cfg.val_dataloader.dataset.pipeline)
        train_inner_cfg.test_mode = True
        train_inner_cfg.pipeline = val_pipeline
        ds = DATASETS.build(train_inner_cfg)

    print(f'Split: {args.split}, dataset size: {len(ds)}')

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=pseudo_collate)

    results = {}
    total_unk = 0
    total_known = 0
    n_total = min(args.n_samples, len(loader)) if args.n_samples > 0 else len(loader)

    for idx, data in enumerate(loader):
        if args.n_samples > 0 and idx >= args.n_samples:
            break
        if idx % 50 == 0:
            print(f'  {idx}/{n_total}...')

        ds_sample = (data['data_samples'][0] if isinstance(data.get('data_samples'), list)
                     else data['data_samples'])
        sample_idx = ds_sample.metainfo.get('sample_idx', idx) if hasattr(ds_sample, 'metainfo') and ds_sample.metainfo else idx
        token = str(ds.get_data_info(sample_idx)['token'])

        data = model.data_preprocessor(data, False)
        for k, v in data['inputs'].items():
            if isinstance(v, torch.Tensor):
                data['inputs'][k] = v.cuda()
            elif isinstance(v, list):
                data['inputs'][k] = [x.cuda() if isinstance(x, torch.Tensor) else x for x in v]

        with torch.no_grad():
            outputs = model.predict(data['inputs'], data['data_samples'])

        pred = outputs[0].pred_instances_3d
        pboxes = pred.bboxes_3d.cpu().numpy()
        plabels = pred.labels_3d.cpu().numpy()
        pscores = pred.scores_3d.cpu().numpy()

        sample_preds = []
        for box, label, score in zip(pboxes, plabels, pscores):
            cx, cy, cz = float(box[0]), float(box[1]), float(box[2])
            w, l, h = float(box[3]), float(box[4]), float(box[5])
            yaw = float(box[6])
            qw, qx, qy, qz = np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)

            if label < 10:
                det_name = CLASS_NAMES[int(label)]
            else:
                det_name = 'unknown'

            sample_preds.append({
                'sample_token': token,
                'translation': [cx, cy, cz],
                'size': [w, l, h],
                'rotation': [qw, qx, qy, qz],
                'velocity': [0.0, 0.0],
                'detection_name': det_name,
                'detection_score': float(score),
                'attribute_name': '',
            })

        results[token] = sample_preds
        n_unk = int((plabels == 10).sum())
        n_known = int((plabels != 10).sum())
        total_unk += n_unk
        total_known += n_known

    out = {'meta': {'use_camera': True, 'use_lidar': True}, 'results': results}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)

    print(f'\nSaved {len(results)} samples, {total_known+total_unk} preds '
          f'(known={total_known} unk={total_unk}) -> {args.out}')


if __name__ == '__main__':
    main()
