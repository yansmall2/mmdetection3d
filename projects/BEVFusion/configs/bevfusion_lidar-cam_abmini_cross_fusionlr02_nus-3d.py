_base_ = ['./bevfusion_lidar-cam_abmini_cross_nus-3d.py']

# B-focused stabilization:
# reduce learning rate of fusion/cross-attention block to ease early training.
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'fusion_layer': dict(lr_mult=0.2)}))
