import numpy as np
from dataclasses import dataclass


@dataclass
class SealCheckConfig:

    """通用参数配置"""
    GRAVITY_VEC = np.array([0.0, 0.0, -9.8], dtype=np.float64)
    NORMAL_ORIENT_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    point_sampler_nums: int = 40   # 单区域点云采样点云数  200  100   80   50  30


    """吸盘参数配置"""
    disc_radius: float  = 0.005     # 吸盘半径 R；密封圆盘、边缘扇区、碰撞圆柱半径  0.005  0.004
    disc_offset: float  = 0.015     # Offset distance along local x for suction center
    polygon: int = 64                    # 吸盘边缘离散为 64 边形
    tip_length: float  = 0.18      # 吸盘 tip 长度 0.0038  0.018
    cylinder_ratio: float = 0.25   # 可视化圆柱体的原始半径比例,默认 0.25


    """密封几何检测参数（对应 check_point_cloud_seal）。"""
    min_support_points: int = 30            # 最小近邻个数
    support_coverage_ratio: float  = 0.35   # 覆盖率系数
    rim_scale: float  = 1.2 # 
    min_sectors_ok_ratio: float  = 0.90 # 

    depth_ratio: float = 0.4            # 密封深度带 = ratio×R，用于平整度与深度判定  0.5
    depth_band_m: float = 0.0015        # 0.0
    n_rim_sectors: int = 16             # 吸盘圆周扇区数，用于边缘支撑检测
    rim_frac_ok: float =  0.875         # 全半径圆周扇区至少该比例有支撑 0.75
    rim_tol_ratio: float = 0.45         # 扇区探针邻域半径 = ratio×R（须贴近探针位置）
    rim_tol_min_m: float = 0.0012       # 扇区探针邻域下限 1.2mm
    rim_depth_ratio: float = 0.5        # 边缘扇区深度带 = max(depth_band, ratio×R) 0.5t
    

    """multi annular patch coverage 参数配置"""
    multi_n_radial_samples: int = 3     # 检测环形区域，径向采样点数
    multi_n_sectors: int = 8            # 检测环形区域，角度扇区数（采样数）
    multi_n_angles: int = 8             # 在吸盘圆盘内均匀采样，角度方向数量 
    multi_n_radii: int = 2              # 在吸盘圆盘内均匀采样，半径层数


    """annular参数配置"""
    annular_inner_radius_ratio: float = 0.5     # 内径比例（如 1.0 表示吸盘边缘）
    annular_outer_radius_ratio: float = 1.2     # 外径比例（如 1.5 表示外侧1.5倍半径）


    """annular disk patch coverage 参数配置"""
    annular_n_radial_samples: int = 5           # 检测环形区域，径向采样点数  5
    annular_n_sectors: int = 16                 # 检测环形区域，角度扇区数（采样数） 16
    patch_n_angles: int = 12                    # 在吸盘圆盘内均匀采样，角度方向数量  12
    patch_n_radii: int = 3                      # 在吸盘圆盘内均匀采样，半径层数 3

    disk_patch_frac_ok: float = 0.8             # 圆盘网格采样点被表面支撑的比例
    disk_sample_tol_ratio: float = 0.40         # 采样点 NN 容差 = ratio×R (default:0.42)
    flatness_slack: float = 1.1                 # 平整度：depth_ptp <= depth_band×slack
    min_support_points: int = 0
    support_coverage_ratio: float = 0.35        # 自动估计圆盘最少支撑点时的有效覆盖比例
    density_ref_spacing_m: float = 0.001        # 密封密度分归一化用的参考点间距 0.8mm



    """抗扭参数配置"""
    use_wrench_score = False             # 是否採用抗扭功能
    wrench_k = 30.0                     # 抗扭分缩放 k，越大对扭矩越不敏感

    """碰撞检测参数配置"""
    cup_height: float = 0.15            # 吸盘高度：碰撞腔体沿外法向高度 15mm
    near_margin: float = 0.005          # 近端留空距离，避免接触面附近的点被误判为碰撞

    """多吸盤参数配置"""
    use_multi_suctions: bool = True     # 是否使用多吸盤
    suction_spacing: float = 0.01       # 吸盘间距 15mm  0.008  0.01  0.012
    num_side_suctions: float = 1        # 每侧吸盘数量  2  
    side_min_score: float = 0.75         # 任一侧吸盘密封/碰撞检测 最小得分

    """ 旋转遍历参数 """
    angle_start: float = 0                      # 起始角度（度）
    angle_end: float = 60                       # 结束角度（度）
    angle_step: float = 3                       # 步长（度） 5  2  3
    side_tolerance: float =0.003                # 侧吸盘位置是否在物体表面附近  
    skip_min_suction_nums: int = 50             # 12 饱和检测：收集至少 10 个方向后，
    skip_min_suc_score_ratio: float = 0.005     # 0.01 如果最新角度与历史最优之差 < 0.01（1%），说明分数已经稳定，不再继续。


    """ 线程参数 """
    thread_workers: int = 8             # 线程池工作线程数


    """可视化参数"""
    save_detect_res: bool = False       # 是否保存检测结果
    vis_multi_suc_num: int = 3         # 可视化多吸盘的个数, 单吸盘全部保留  10
    vis_region_suction: bool = False    # 可视化单区域点云的吸取效果
    vis_mask: bool = False              # 可视化局部mask
    vis_region_points: bool = False     # 可视化局部区域points
    vis_fps: bool = False               # 可视化fps
    vis_wrench: bool = False            # 可视化wrench
    print_wrench_log: bool = False      # 可视化wrench
    vis_scaled_obb: bool = False        # 可视化obb


    save_region_ply: bool = False       # 是否保存单区域点云
    use_region_mask: bool = True        # 是否将局部mask限定在安全工作区域内

    """ 调优加速参数 """
    use_preprocess_depth: bool =False   # 是否预处理深度图
    pre_depth_radius: int = 3           # 核半径

    skip_detect_num: int = 7            # skip 检测个数   10  8  7
    ratio_major: float = 0.75           # 0.75
    ratio_minor: float = 0.45            # 1.0  0.45 


    num_neighbors:int = 30         # 生成局部点云下采样的最近邻个数
    voxel_size: float = 0.002      # 生成局部点云下采样的size  对于 4mm 吸盘可改为 0.0025~0.003，点云减少 ~30%，密封精度几乎无影响。
    upper_ratio: float = 0.80      # 过滤底层点云比例
    max_angle_deg: float = 70.0    # 保留的最大角度值
    

    min_region_points_num: int = 50         # 最少的用于计算的区域点云数
    min_single_suction_score: float = 0.5   # 过滤掉计算的单吸盘最小分数
    min_acquired_suction_nums: int = 45     # 当前区域最少的合适吸取点个数,超过的则不再遍历,用于加速  20



    depth_ptp_bins = [
        (0.0015, 0.90, 1.00),
        (0.0020, 0.85, 0.90),
        (0.0025, 0.80, 0.85),
        (0.0035, 0.70, 0.80),
        (0.0055, 0.50, 0.70),
        (0.01, 0.00, 0.50),
    ]

    def depth_band_for(self, disc_radius: float) -> float:
        if self.depth_band_m > 0:
            return self.depth_band_m
        return max(0.001, disc_radius * self.depth_ratio)
    

seal_cfg = SealCheckConfig()