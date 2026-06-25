import time
import numpy as np
import open3d as o3d
from FRSuctionAPI.suction_utils.suction_collision import SuctionCupCollisionChecker, CupCollisionConfig
from FRSuctionAPI.suction_utils.suction_score import DepthPtpScorer
from FRSuctionAPI.sim_utils.frame_utils import compute_darboux_frame as _compute_darboux_frame
from FRSuctionAPI.algo2.infer_obb_middle import Solve_Suciton_Pose
from FRSuctionAPI.algo2.common_interface import SuctionPipeline

from cfg import seal_cfg


def test_ply(ply_path: str = "./test_ply_crossbeam/mask_000.ply"):
    """PLY 文件测试：加载 → 滤波 → pipeline.process_region() → 可视化"""
    # ── 加载 + 滤波（同 sim_infer.py） ──
    if isinstance(ply_path, str):
        pcd = o3d.io.read_point_cloud(ply_path)
    elif isinstance(ply_path, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ply_path.astype(np.float64))
    elif isinstance(ply_path, o3d.geometry.PointCloud):
        pcd = ply_path
    else:
        raise TypeError(f"unsupported type: {type(ply_path)}")
    pcd, _ = pcd.remove_statistical_outlier(100, 1.0)
    pts = np.asarray(pcd.points)
    print(f"cloud points: {len(pts)}")

    # ── 初始化管线 ──
    collision_checker = SuctionCupCollisionChecker(
        CupCollisionConfig(disc_radius=seal_cfg.disc_radius,
                           cup_height=seal_cfg.cup_height,
                           near_margin=seal_cfg.near_margin))
    suction_cfg = dict(
        collision_checker=collision_checker,
        depth_ptp_scorer=DepthPtpScorer(unit="m"),
        disc_radius=seal_cfg.disc_radius,
        sample_tol=max(seal_cfg.disc_radius * seal_cfg.disk_sample_tol_ratio,
                       seal_cfg.rim_tol_min_m, seal_cfg.disc_radius * 0.6),
        depth_band_m=seal_cfg.depth_band_for(seal_cfg.disc_radius),
        annular_inner_radius_ratio=seal_cfg.annular_inner_radius_ratio,
        annular_outer_radius_ratio=seal_cfg.annular_outer_radius_ratio,
        min_sectors_ok_ratio=seal_cfg.min_sectors_ok_ratio,
        disk_patch_frac_ok=seal_cfg.disk_patch_frac_ok,
        flatness_slack=seal_cfg.flatness_slack,
        multi_n_radial_samples=seal_cfg.multi_n_radial_samples,
        multi_n_sectors=seal_cfg.multi_n_sectors,
        multi_n_angles=seal_cfg.multi_n_angles,
        multi_n_radii=seal_cfg.multi_n_radii,
        annular_n_radial_samples=seal_cfg.annular_n_radial_samples,
        annular_n_sectors=seal_cfg.annular_n_sectors,
        patch_n_angles=seal_cfg.patch_n_angles,
        patch_n_radii=seal_cfg.patch_n_radii,
        min_support_points=30, support_coverage_ratio=0.35,
    )
    pipe = SuctionPipeline(collision_checker, suction_cfg)

    # region_pts 和 middle_o3d 都使用同一个点云（单 PLY 无中部区域概念）
    middle_o3d = o3d.geometry.PointCloud()
    middle_o3d.points = o3d.utility.Vector3dVector(pts.astype(np.float32))

    t1 = time.time()
    results = pipe.process_region(
        region_pts=pts,
        middle_o3d=middle_o3d,
        scene_points=None,  # 仅自碰撞
    )
    dt = time.time() - t1
    print(f"process_region: {len(results)} candidates in {dt:.2f}s")

    # ── 输出 + 可视化 ──
    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n总候选: {len(results)}")
    scores = [r["score"] for r in results[:10]]
    print("top 10:", "  ".join("%.3f" % s for s in scores))
    pipe.visualize_results(results, pts)

if __name__ == "__main__":
    test_ply(ply_path = "../test_ply_crossbeam/mask_000.ply")
   