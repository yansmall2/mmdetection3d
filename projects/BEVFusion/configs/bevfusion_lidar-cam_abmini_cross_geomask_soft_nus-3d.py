_base_ = ['./bevfusion_lidar-cam_abmini_cross_geomask_nus-3d.py']

# B+M-soft: soft geometry mask to reduce hard-threshold sensitivity on sparse points.
model = dict(
    fusion_layer=dict(
        use_geometry_mask=True,
        geo_mask_type='soft',
        geo_mask_temp=0.10))
