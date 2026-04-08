_base_ = ['./bevfusion_lidar-cam_abmini_cross_nus-3d.py']

# Stage-3 geometry-mask branch (default OFF for strict control baseline).
model = dict(
    fusion_layer=dict(
        type='CrossAttentionFuser',
        in_channels=[80, 256],
        out_channels=256,
        use_geometry_mask=False,
        geo_mask_tau=0.10,
        geo_mask_lambda=0.50,
        geo_mask_pool_kernel=3,
        geo_mask_type='hard',
        geo_mask_temp=0.10,
        use_residual_img=False))
