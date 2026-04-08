_base_ = ['./bevfusion_lidar-cam_abmini_cross_geomask_nus-3d.py']

# B+M: enable geometry existence mask gating before cross-attention.
model = dict(fusion_layer=dict(use_geometry_mask=True))
