from FRSuctionAPI.algo_tf.common_interface import SuctionPipeline
import time
import cv2,sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── OBB 检测 + 预处理（同 infer_obb_middle.py） ──
from FRSuctionAPI.suction_utils.cv_tool import (
    generate_workspace_mask, get_scaled_obb_mask, preprocess_depth,
    get_mask_points_o3d)
from FRSuctionAPI.suction_utils.point_cloud_tool import (
    CameraInfo, create_point_cloud_from_depth_image)

from params import params_cfg as params
from cfg import seal_cfg

from FRSuctionAPI.tf_utils.gpu_utils import get_device

if __name__ == "__main__":
    
    # YOLO
    model = YOLO(params.get('obb_checkpoint'))
    model.to('cuda:0' if get_device().type == 'cuda' else 'cpu')

    # 测试图像
    # img_dir = Path(__file__).resolve().parent.parent / "local_mech_imgs"
    # pngs = sorted(img_dir.glob("*_rgb.png"))
    # if not pngs:
    #     print("No test images"); sys.exit(1)
    # stem = pngs[0].stem.replace("_rgb", "")

    local_rgb_path = './local_mech_imgs/1779781121.914089_rgb.png' 
    local_depth_path =  './local_mech_imgs/1779781121.914089_depth.png'
    color_img = cv2.imread(local_rgb_path)
    depth_img = cv2.imread(local_depth_path,cv2.IMREAD_UNCHANGED)

    if len(color_img.shape) == 2:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
    if len(depth_img.shape) == 3:
        depth_img = np.squeeze(depth_img)
    if seal_cfg.use_preprocess_depth:
        depth_img = preprocess_depth(depth_img, radius=seal_cfg.pre_depth_radius)
    h, w = color_img.shape[:2]

    # 场景点云
    K = params['CameraMatrix']
    fd = params.get('factor_depth', 1000.0)
    cam = CameraInfo(w, h, float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]), fd)
    scene = create_point_cloud_from_depth_image(depth_img, cam, organized=True)
    ws = (generate_workspace_mask(w, h, params.get('top_left', (0,0)),
                                   params.get('bottom_right', (w,h))) > 0) & (depth_img > 0)
    scene_pts = scene[ws]
    color_msk = color_img[ws]
    print(f"Scene: {scene_pts.shape[0]} pts")

    # YOLO detect
    res = model(color_img, conf=params.get('detect_conf', 0.25),
                save=False, show=False, show_labels=False, show_boxes=False, line_width=1)
    if len(res[0]) == 0:
        print("No detections"); sys.exit(1)
    xywhrs = res[0].obb.xywhr.cpu().data.numpy()
    print(f"Detected: {len(xywhrs)} OBBs")

    rm = generate_workspace_mask(w, h, params.get('region_left', (0,0)),
                                  params.get('region_right', (w,h)), use_scale_size=0)

    regions = []
    for i in range(min(len(xywhrs), seal_cfg.skip_detect_num)):
        sm = get_scaled_obb_mask(xywhrs[i], color_img, 1.0, 1.0)
        if seal_cfg.use_region_mask:
            sm = np.minimum(sm, rm.astype(np.uint8))
            if sm.max() == 0: continue
        fm = get_scaled_obb_mask(xywhrs[i], color_img, seal_cfg.ratio_major, seal_cfg.ratio_minor)

        reg = get_mask_points_o3d(scene, depth_img, sm, seal_cfg.num_neighbors, 1.0, seal_cfg.voxel_size)
        mid = get_mask_points_o3d(scene, depth_img, fm, seal_cfg.num_neighbors, 1.0, seal_cfg.voxel_size)
        if len(reg.points) < seal_cfg.min_region_points_num or len(mid.points) < seal_cfg.min_region_points_num:
            continue
        print(f"  mask {i}: region={len(reg.points)}, mid={len(mid.points)}")
        regions.append((np.asarray(reg.points, dtype=np.float32), mid))

    # 管道
    t1 = time.time()
    pipeline = SuctionPipeline()
    results = pipeline.process(regions, scene_pts)
    t2 = time.time()
    print(f"time: {t2-t1:.2f}s, candidates: {len(results)}")
    if results:
        print("Top scores:", [r['score'] for r in results[:10]])
        pipeline.visualize(scene_pts, results, disc_radius=seal_cfg.disc_radius,
                           tip_length=seal_cfg.tip_length,
                           vis_multi_suc_num=seal_cfg.vis_multi_suc_num,
                           color_masked=color_msk,
                           cylinder_ratio=seal_cfg.cylinder_ratio)