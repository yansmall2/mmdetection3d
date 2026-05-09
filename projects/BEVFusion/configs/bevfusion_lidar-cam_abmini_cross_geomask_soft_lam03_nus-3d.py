_base_ = ['./bevfusion_lidar-cam_abmini_cross_geomask_soft_nus-3d.py']

# B+M-soft with milder geometry boost.
# Objectness branch follows the report's target direction:
# proposal-level BCE as the main supervision.
model = dict(
    view_transform=dict(
        use_depth_valid_mask=True,
        depth_valid_mask_weight=0.3,
        depth_dilate_kernel=3,
    ),

    fusion_layer=dict(
        geo_mask_lambda=0.30
    ),

    bbox_head=dict(
        # objectness: query-level BCE as primary supervision,
        # with light dense supervision to prevent dense head degradation.
        objectness_dense_supervision=True,
        objectness_dense_loss_weight=0.1,
        objectness_query_loss_weight=1.0,
        objectness_bce_pos_weight=1.0,
        objectness_use_query_seed_score=True,
        num_unknown_proposals=25,

        test_cfg=dict(
            open_world_mode='dual_track',
            unknown_obj_thresh=0.20,
            unknown_cls_thresh=0.30,
            unknown_label_id=10,
        )
    )
)

test_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    metric='bbox',
    holdout_classes=['truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'],
    unknown_label_id=10
)

val_evaluator = test_evaluator

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
