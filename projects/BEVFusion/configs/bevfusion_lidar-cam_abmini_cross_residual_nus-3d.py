_base_ = ['./bevfusion_lidar-cam_abmini_cross_nus-3d.py']

# Optional follow-up config for step-4 validation:
# residual safeguard on image branch in CrossAttentionFuser.
model = dict(
    fusion_layer=dict(
        type='CrossAttentionFuser',
        in_channels=[80, 256],
        out_channels=256,
        use_residual_img=True,
        residual_alpha_init=0.1))
