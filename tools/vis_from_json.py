"""Visualize predictions from JSON results file on camera images.
Usage: python tools/vis_from_json.py [--json PATH] [--out_dir DIR] [--split val] [--n_samples N]
"""

import argparse
import os, sys
os.chdir('/home/yan/mmdetection3d')

import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nuscenes.nuscenes import NuScenes

# Category mapping: nuscenes name -> 10-class index
NUSC_TO_IDX = {
    'vehicle.car': 0,
    'vehicle.truck': 1,
    'vehicle.construction': 2,
    'vehicle.bus.rigid': 3,
    'vehicle.bus.bendy': 3,
    'vehicle.trailer': 4,
    'vehicle.emergency.ambulance': 4,
    'vehicle.emergency.police': 4,
    'movable_object.barrier': 5,
    'vehicle.motorcycle': 6,
    'vehicle.bicycle': 7,
    'human.pedestrian.adult': 8,
    'human.pedestrian.child': 8,
    'human.pedestrian.wheelchair': 8,
    'human.pedestrian.stroller': 8,
    'human.pedestrian.personal_mobility': 8,
    'human.pedestrian.police_officer': 8,
    'human.pedestrian.construction_worker': 8,
    'movable_object.trafficcone': 9,
}
CLASS_NAMES = ['car', 'truck', 'cv', 'bus', 'trailer', 'barrier',
               'moto', 'bike', 'ped', 'cone']
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# Color palette
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
UNKNOWN_COLOR = np.array([1.0, 0.0, 0.0])
GT_COLOR = np.array([0.0, 1.0, 0.0])


def get_corners(box):
    cx, cy, cz = float(box[0]), float(box[1]), float(box[2])
    w, l, h = float(box[3]), float(box[4]), float(box[5])
    rot_z = float(box[6]) if len(box) >= 7 else 0.0
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2,  h/2], [l/2, -w/2,  h/2], [l/2, w/2,  h/2], [-l/2, w/2,  h/2],
    ])
    cos_r, sin_r = np.cos(rot_z), np.sin(rot_z)
    R = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    return corners @ R.T + np.array([cx, cy, cz])


def project_box(box, l2c, K, img_W, img_H):
    corners_lidar = get_corners(box)
    corners_cam = (l2c[:3, :3] @ corners_lidar.T + l2c[:3, 3:4]).T
    if (corners_cam[:, 2] < 0.1).all():
        return None
    if K.shape == (4, 4):
        pts = (K @ np.hstack([corners_cam, np.ones((8, 1))]).T).T
    else:
        pts = (K @ corners_cam.T).T
    pts2d = pts[:, :2] / pts[:, 2:3].clip(1e-6)
    center_2d = pts2d.mean(axis=0)
    if center_2d[0] < 0 or center_2d[0] > img_W or center_2d[1] < 0 or center_2d[1] > img_H:
        return None
    return pts2d


def draw_box(ax, pts, color, lw=1.5):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e1, e2 in edges:
        ax.plot([pts[e1,0], pts[e2,0]], [pts[e1,1], pts[e2,1]],
                color=color, lw=lw, alpha=0.7)


def quaternion_yaw(q):
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_lidar2cam(nusc, cam_data):
    """Compute lidar-to-camera transform for a given camera sample_data."""
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])

    # Camera calibration: sensor → ego vehicle
    cam_trans = np.array(cs_record['translation'])
    cam_rot = np.array(cs_record['rotation'])  # [w, x, y, z]

    # Ego pose: ego → global
    ego_trans = np.array(pose_record['translation'])
    ego_rot = np.array(pose_record['rotation'])

    # Camera intrinsics
    K = np.array(cs_record['camera_intrinsic'])

    # Build 4x4 transform: lidar → ego → global → ego → camera
    # For nuscenes, lidar and camera are both on ego vehicle
    # Use lidar calibration directly from the same sample

    # Find lidar sample_data for this sample
    sample = nusc.get('sample', cam_data['sample_token'])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_trans = np.array(lidar_cs['translation'])
    lidar_rot = np.array(lidar_cs['rotation'])

    # Build transforms
    def rot_matrix(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ])

    R_lidar = rot_matrix(lidar_rot)
    T_lidar = np.eye(4)
    T_lidar[:3, :3] = R_lidar
    T_lidar[:3, 3] = lidar_trans

    R_cam = rot_matrix(cam_rot)
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_cam
    T_cam[:3, 3] = cam_trans

    # lidar2cam = inv(T_cam) @ T_lidar
    l2c = np.linalg.inv(T_cam) @ T_lidar
    return l2c, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str,
                        default='work_dirs/test_results/pred_instances_3d/results_nusc_with_unknown.json')
    parser.add_argument('--out_dir', type=str, default='vis_from_json')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'mini'])
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--data_root', type=str, default='data/nuscenes')
    args = parser.parse_args()

    version = 'v1.0-trainval'
    if args.split == 'mini':
        version = 'v1.0-mini'

    os.makedirs(args.out_dir, exist_ok=True)

    # Load predictions
    with open(args.json) as f:
        data = json.load(f)
    results = data['results']
    meta = data.get('meta', {})

    print(f'Loaded {len(results)} samples from {args.json}')
    print(f'Meta: {meta}')

    # Initialize nuscenes
    nusc = NuScenes(version=version, dataroot=args.data_root, verbose=False)

    # Build token -> sample lookup
    token_to_sample = {}
    for s in nusc.sample:
        token_to_sample[s['token']] = s

    # Collect stats
    total_preds = 0
    total_unk = 0
    total_gt = 0

    sample_tokens = list(results.keys())

    for vis_idx, sample_token in enumerate(sample_tokens):
        if vis_idx >= args.n_samples:
            break

        if sample_token not in token_to_sample:
            print(f'  [{vis_idx}] token {sample_token} not in {version}, skip')
            continue

        preds = results[sample_token]
        sample = token_to_sample[sample_token]

        # Parse predictions
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        n_unk = 0

        for p in preds:
            box = np.array(p['translation'] + p['size'] +
                          [quaternion_yaw(p['rotation'])])
            name = p['detection_name']
            score = p['detection_score']

            if name == 'unknown':
                pred_labels.append(10)
                n_unk += 1
            elif name in CLASS_NAMES:
                pred_labels.append(CLASS_NAMES.index(name))
            else:
                # Map from nuscenes full name
                idx = NUSC_TO_IDX.get(name)
                if idx is not None:
                    pred_labels.append(idx)
                else:
                    continue  # skip unrecognized classes

            pred_boxes.append(box)
            pred_scores.append(score)

        total_preds += len(pred_boxes)
        total_unk += n_unk

        # Parse GT from annotations
        gt_boxes = []
        gt_labels = []

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            cat = ann['category_name']
            idx = NUSC_TO_IDX.get(cat)
            if idx is None:
                continue

            box = np.array(list(ann['translation']) + list(ann['size']) +
                          [quaternion_yaw(ann['rotation'])])
            gt_boxes.append(box)
            gt_labels.append(idx)

        total_gt += len(gt_boxes)

        # Render 6 camera views
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
        axes = axes.flatten()

        for cam_idx, cam_name in enumerate(CAM_NAMES):
            ax = axes[cam_idx]

            if cam_name not in sample['data']:
                ax.text(0.5, 0.5, f'{cam_name}\nno data', ha='center', va='center')
                ax.axis('off')
                continue

            cam_data = nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(args.data_root, cam_data['filename'])

            if not os.path.exists(img_path):
                ax.text(0.5, 0.5, f'{cam_name}\nno image', ha='center', va='center')
                ax.axis('off')
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            ax.imshow(img)

            l2c, K = get_lidar2cam(nusc, cam_data)

            # Draw GT (green, thin)
            for box in gt_boxes:
                pts = project_box(box, l2c, K, W, H)
                if pts is not None:
                    draw_box(ax, pts, color='#00ff00', lw=0.8)

            # Draw known predictions (class color)
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label == 10:
                    continue  # draw unknowns last for visibility
                if score < 0.1:
                    continue
                pts = project_box(box, l2c, K, W, H)
                if pts is not None:
                    color = CLASS_COLORS[label % 10]
                    draw_box(ax, pts, color=color, lw=1.0)

            # Draw unknown predictions (red, bold)
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label != 10:
                    continue
                if score < 0.1:
                    continue
                pts = project_box(box, l2c, K, W, H)
                if pts is not None:
                    draw_box(ax, pts, color='#ff0000', lw=2.5)
                    cx = pts[:, 0].mean()
                    cy = max(pts[:, 1].min() - 5, 10)
                    ax.text(cx, cy, f'UNK:{score:.2f}', color='red',
                           fontsize=6, ha='center', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

            ax.set_title(cam_name, fontsize=10)
            ax.axis('off')

        # Legend
        handles = [mpatches.Patch(color='#00ff00', alpha=0.5,
                                   label=f'GT ({len(gt_boxes)})')]
        shown = set()
        for lbl in pred_labels:
            if lbl < 10 and lbl not in shown:
                shown.add(lbl)
                handles.append(mpatches.Patch(color=CLASS_COLORS[lbl], alpha=0.5,
                                              label=CLASS_NAMES[lbl]))
        if n_unk > 0:
            handles.append(mpatches.Patch(color='#ff0000', alpha=0.5,
                                          label=f'UNKNOWN ({n_unk})'))
        fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=7)
        fig.suptitle(f'Sample {vis_idx} | token={sample_token[:16]}... | '
                     f'preds={len(pred_boxes)} unk={n_unk}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])

        out_path = os.path.join(args.out_dir, f'sample_{vis_idx:03d}.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  [{vis_idx}] preds={len(pred_boxes)} unk={n_unk} gt={len(gt_boxes)} -> {out_path}')

    print(f'\nDone. {len(sample_tokens[:args.n_samples])} samples -> {args.out_dir}/')
    print(f'  Total preds: {total_preds}, unknowns: {total_unk}, GTs: {total_gt}')


if __name__ == '__main__':
    main()
