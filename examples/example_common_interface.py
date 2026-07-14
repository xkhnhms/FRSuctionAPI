from FRSuctionAPI.algo_tf.common_interface import SuctionPipeline
import time
import cv2,sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from FRSuctionAPI.suction_utils.cv_tool import (
    generate_workspace_mask, get_scaled_obb_mask, preprocess_depth,
    get_mask_points_o3d)
from FRSuctionAPI.suction_utils.point_cloud_tool import (
    CameraInfo, create_point_cloud_from_depth_image)

from params import params_cfg as params
from cfg import seal_cfg

from FRSuctionAPI.tf_utils.gpu_utils import get_device

if __name__ == "__main__":
    
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
