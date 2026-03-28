import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np

DATAROOT = 'data/nuscenes'
VERSION = 'v1.0-mini'

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

target_class = 'vehicle.construction'

found_samples = []

# ======================
# 1. 遍历所有 sample，找出包含 construction_vehicle 的样本
# ======================
print("正在搜索包含 construction_vehicle 的样本...")

for sample in nusc.sample:
    has_target = False
    target_annotations = []  # 存储目标标注信息
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        if ann['category_name'] == target_class:
            has_target = True
            target_annotations.append(ann)
    
    if has_target:
        found_samples.append({
            'sample': sample,
            'annotations': target_annotations
        })
        print(f"✅ 找到 sample token: {sample['token']} (包含 {len(target_annotations)} 个工程车辆)")

print(f"\n总共找到 {len(found_samples)} 个包含 construction_vehicle 的样本")

# ======================
# 2. 输出第一个找到的样本的详细信息（包括前视图）
# ======================
if len(found_samples) == 0:
    print("❌ 在 mini 数据集中没有找到 construction_vehicle 样本")
    print("提示：v1.0-mini 可能不包含 construction_vehicle，建议使用完整版 v1.0-trainval")
else:
    first_item = found_samples[0]
    sample = first_item['sample']
    target_anns = first_item['annotations']
    
    print(f"\n{'='*60}")
    print(f"第一个样本详细信息:")
    print(f"{'='*60}")
    print(f"Sample token: {sample['token']}")
    print(f"场景时间戳: {sample['timestamp']}")
    
    # ======================
    # 2.1 输出点云信息
    # ======================
    print(f"\n--- 点云数据 ---")
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = os.path.join(DATAROOT, lidar_data['filename'])
    print(f"点云路径: {lidar_path}")
    print(f"点云文件是否存在: {os.path.exists(lidar_path)}")
    print(f"点云时间戳: {lidar_data['timestamp']}")
    
    # ======================
    # 2.2 输出前视图（相机）信息
    # ======================
    print(f"\n--- 前视图（相机）数据 ---")
    
    # nuScenes 中前视相机的通道名称通常是 'CAM_FRONT'
    front_cam_token = sample['data'].get('CAM_FRONT')
    
    if front_cam_token:
        front_cam_data = nusc.get('sample_data', front_cam_token)
        front_cam_path = os.path.join(DATAROOT, front_cam_data['filename'])
        
        print(f"前视图路径: {front_cam_path}")
        print(f"前视图文件是否存在: {os.path.exists(front_cam_path)}")
        print(f"前视图时间戳: {front_cam_data['timestamp']}")
        print(f"图像尺寸: {front_cam_data['width']} x {front_cam_data['height']}")
        
        # 可选：获取相机的内参和外参
        if 'calibrated_sensor_token' in front_cam_data:
            calib_data = nusc.get('calibrated_sensor', front_cam_data['calibrated_sensor_token'])
            print(f"\n相机内参: {calib_data.get('camera_intrinsic', 'N/A')}")
            
            # 获取相机外参（相对于车辆坐标系）
            ego_pose = nusc.get('ego_pose', front_cam_data['ego_pose_token'])
            print(f"相机位置 (x, y, z): {calib_data['translation']}")
            print(f"相机旋转四元数: {calib_data['rotation']}")
    else:
        print("⚠️ 未找到前视图数据 (CAM_FRONT)")
        print("可用的相机通道:")
        for cam_key in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
            if cam_key in sample['data']:
                print(f"  - {cam_key}: 可用")
    
    # ======================
    # 2.3 输出工程车辆的具体信息（3D框）
    # ======================
    print(f"\n--- 工程车辆 (construction_vehicle) 详细信息 ---")
    for i, ann in enumerate(target_anns, 1):
        print(f"\n工程车辆 {i}:")
        print(f"  标注 token: {ann['token']}")
        print(f"  类别: {ann['category_name']}")
        print(f"  3D中心点 (x, y, z): {ann['translation']}")
        print(f"  3D尺寸 (w, l, h): {ann['size']}")
        print(f"  朝向旋转: {ann['rotation']}")
        
        # 计算车辆在鸟瞰图中的位置
        print(f"  鸟瞰图位置 (x, y): ({ann['translation'][0]:.2f}, {ann['translation'][1]:.2f})")
        
        # 可选：获取该标注的可见性和属性
        if 'visibility_token' in ann:
            visibility = nusc.get('visibility', ann['visibility_token'])
            print(f"  可见性: {visibility['description']}")
        
        if 'attribute_tokens' in ann and ann['attribute_tokens']:
            attributes = []
            for attr_token in ann['attribute_tokens']:
                attr = nusc.get('attribute', attr_token)
                attributes.append(attr['description'])
            print(f"  属性: {', '.join(attributes)}")
    
    # ======================
    # 2.4 可选：显示所有传感器数据
    # ======================
    print(f"\n--- 该样本可用的所有传感器数据 ---")
    for sensor_name, sensor_token in sample['data'].items():
        sensor_data = nusc.get('sample_data', sensor_token)
        print(f"  {sensor_name}: {sensor_data['filename']}")

# ======================
# 3. 辅助函数：在图像上绘制3D框（可选）
# ======================
def project_3d_box_to_image(nusc, sample, ann_token, camera_name='CAM_FRONT'):
    """
    将3D标注框投影到相机图像上（返回2D坐标）
    """
    from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
    from pyquaternion import Quaternion
    
    # 获取相机数据
    cam_token = sample['data'][camera_name]
    cam_data = nusc.get('sample_data', cam_token)
    
    # 获取标注框
    ann = nusc.get('sample_annotation', ann_token)
    
    # 创建3D框
    box = nusc.get_box(ann_token)
    
    # 将框从全局坐标系转换到相机坐标系
    box.rotate(Quaternion(cam_data['rotation']).inverse)
    box.translate(-np.array(cam_data['translation']))
    
    # 获取相机内参
    calib_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    intrinsic = np.array(calib_data['camera_intrinsic'])
    
    # 投影到图像平面
    corners_3d = box.corners().T  # 8个角点
    corners_2d = []
    
    for corner in corners_3d:
        # 透视投影
        x = corner[0] / corner[2]
        y = corner[1] / corner[2]
        
        # 应用内参
        u = intrinsic[0, 0] * x + intrinsic[0, 2]
        v = intrinsic[1, 1] * y + intrinsic[1, 2]
        
        corners_2d.append([u, v])
    
    return np.array(corners_2d)

print(f"\n{'='*60}")
print("搜索完成！")