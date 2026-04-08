_base_ = ['./bevfusion_lidar-cam_abmini_cross_geomask_soft_nus-3d.py']

# B+M-soft with milder geometry boost.
model = dict(fusion_layer=dict(geo_mask_lambda=0.30))
