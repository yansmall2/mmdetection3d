import os
import numpy as np
import matplotlib.pyplot as plt

from mmdet3d.apis import init_model, inference_detector

# === 配置 ===
config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
pcd_file = 'demo/data/kitti/000008.bin'

# === 初始化模型 ===
model = init_model(config_file, checkpoint_file, device='cuda:0')

# === 推理 ===
result, data = inference_detector(model, pcd_file)

# === 读取点云 ===
points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)

# === 提取检测框 ===
bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
scores = result.pred_instances_3d.scores_3d.cpu().numpy()

# === 过滤低分框 ===
mask = scores > 0.3
bboxes = bboxes[mask]

# === 创建输出目录 ===
os.makedirs('outputs', exist_ok=True)

# === BEV 可视化 ===
plt.figure(figsize=(8, 8))

# 画点云（x, y）
plt.scatter(points[:, 0], points[:, 1], s=0.5)

# 画3D框（只画BEV矩形）
for box in bboxes:
    x, y, z, dx, dy, dz, yaw = box

    # 计算矩形四个角（BEV）
    corners = np.array([
        [dx/2, dy/2],
        [dx/2, -dy/2],
        [-dx/2, -dy/2],
        [-dx/2, dy/2]
    ])

    # 旋转
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    corners = corners @ rot.T

    # 平移
    corners += np.array([x, y])

    # 闭合矩形
    corners = np.vstack([corners, corners[0]])

    plt.plot(corners[:, 0], corners[:, 1])

plt.title("BEV 3D Detection Result (PointPillars)")
plt.xlabel("X")
plt.ylabel("Y")

# 保存
plt.savefig("outputs/result.png")
print("✅ Saved to outputs/result.png")