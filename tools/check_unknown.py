import os
import torch
from mmengine.config import Config
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from torch.utils.data import DataLoader
from mmengine.dataset import DefaultSampler
import numpy as np

base_dir = '/home/yan/mmdetection3d'
config_file = os.path.join(base_dir, 'projects/BEVFusion/configs/bevfusion_lidar-cam_baseline_nus-3d.py')
checkpoint_file = os.path.join(base_dir, 'work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_1.pth')

os.chdir(base_dir)

print("Loading config...")
cfg = Config.fromfile(config_file)
if 'bbox_head' in cfg.model:
    if 'test_cfg' not in cfg.model.bbox_head:
        cfg.model.bbox_head.test_cfg = {}
    cfg.model.bbox_head.test_cfg['open_world_mode'] = 'open_world'
    cfg.model.bbox_head.test_cfg['unknown_obj_thresh'] = 0.5
    cfg.model.bbox_head.test_cfg['unknown_cls_thresh'] = 0.3
    cfg.model.bbox_head.test_cfg['dataset'] = 'nuScenes'
    cfg.model.bbox_head.test_cfg['nms_type'] = 'circle'
    
print("Building model...")
try:
    model = init_model(cfg, checkpoint_file, device='cuda:0')
except Exception as e:
    model = init_model(cfg, None, device='cpu')
    print("Warning: Failed to load checkpoint. Using untrained weights. Error:", e)

print("Building dataset...")
val_dataset = DATASETS.build(cfg.val_dataloader['dataset'])
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    sampler=DefaultSampler(val_dataset, shuffle=False),
    collate_fn=cfg.val_dataloader.get('collate_fn', lambda x: x),
    num_workers=0
)

print("\nRunning inference on a few samples to check for unknown objects...")
model.eval()
unknown_count_total = 0
num_samples_to_test = 20
num_classes = len(val_dataset.METAINFO['classes'])
unknown_id = num_classes

for i, data in enumerate(val_dataloader):
    if i >= num_samples_to_test: break
    
    with torch.no_grad():
        try:
            # Need to preprocess data:
            data = model.data_preprocessor(data, False)
            outputs = model(**data, mode='predict')
        except Exception as e:
            print(f"Error during val_step on sample {i}:", e)
            continue
            
        pred_instances = outputs[0].pred_instances_3d
        labels = pred_instances.labels_3d
        scores = pred_instances.scores_3d
        
        unknown_mask = labels == unknown_id
        num_unknown = unknown_mask.sum().item()
        unknown_count_total += num_unknown
        
        # print(f"\nSample {i}: found {len(labels)} total bounding boxes.")
        if num_unknown > 0:
            unknown_scores = scores[unknown_mask]
            print(f"Sample {i}: {num_unknown} boxes classified as 'unknown' (label_id={unknown_id}).")
            print(f"  => Unknown scores: {unknown_scores.cpu().numpy()}")

print(f"\nTotal unknown boxes found across {min(num_samples_to_test, len(val_dataloader))} samples: {unknown_count_total}")
