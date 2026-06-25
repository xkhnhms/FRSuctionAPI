import time
from FRSuctionAPI.algo.infer_seg_middle import Solve_Suciton_Pose


if __name__ == "__main__":

    # test_ply = '../test_models/002.ply'
    # test_ply = '../test_ply_pipe/imgs/plys/01.ply'

    # local_rgb_path = './local_mech_imgs/1779781121.914089_rgb.png' 
    # local_depth_path =  './local_mech_imgs/1779781121.914089_depth.png'

    local_rgb_path = '../local_mech_imgs/1779781616.5058525_rgb.png' 
    local_depth_path =  '../local_mech_imgs/1779781616.5058525_depth.png'


    suction_pose_solver = Solve_Suciton_Pose()
    # suction_pose_solver.load_point_cloud_from_ply(test_ply)

    t1 =time.time()
    suction_pose_solver.solve(rgb_path=local_rgb_path,depth_path=local_depth_path,is_vis=True)
    t2 =time.time()
    print(f'time costs: {t2-t1}')