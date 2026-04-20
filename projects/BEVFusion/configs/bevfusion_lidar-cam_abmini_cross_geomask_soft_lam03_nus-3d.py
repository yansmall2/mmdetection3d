_base_ = ['./bevfusion_lidar-cam_abmini_cross_geomask_soft_nus-3d.py']

# B+M-soft with milder geometry boost.
model = dict(fusion_layer=dict(geo_mask_lambda=0.30))

test_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    metric='bbox',
    holdout_classes=['truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'],
    unknown_label_id=10
)
