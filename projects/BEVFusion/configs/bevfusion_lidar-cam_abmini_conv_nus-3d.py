_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# A/B debug settings (mini-scale quick diagnosis)
randomness = dict(seed=3407)
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)

# Keep initialization strategy identical between A/B runs.
load_from = (
    'https://download.openmmlab.com/mmdetection3d/v1.1.0_models/'
    'bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_'
    'nus-3d-2628f933.pth')

# Ensure baseline fuser.
model = dict(fusion_layer=dict(type='ConvFuser', in_channels=[80, 256], out_channels=256))

# Force mini split version through dataset metainfo (compatible with this repo's
# NuScenesMetric, which reads dataset_meta['version'] rather than metric args).
val_dataloader = dict(dataset=dict(metainfo=dict(version='v1.0-mini')))
test_dataloader = dict(dataset=dict(metainfo=dict(version='v1.0-mini')))
