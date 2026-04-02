from mmengine.config import Config
from mmdet3d.registry import MODELS
import torch

try:
    # 尝试加载配置文件
    cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
    
    # 尝试构建模型
    model = MODELS.build(cfg.model)
    print("模型构建成功！框架和新加入的 GeometryGuidedConvFuser 测试通过。")
    
    print("\n模型结构:")
    print(model)
except Exception as e:
    print(f"模型构建失败，遇到错误: {e}")
    import traceback
    traceback.print_exc()
