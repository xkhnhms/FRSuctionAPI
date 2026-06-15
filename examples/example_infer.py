import time
from FRSuctionAPI.algo.infer import Solve_Suciton_Pose

if __name__ == "__main__":

    # test_ply = '../test_models/002.ply'
    # test_ply = '../test_ply_pipe/imgs/plys/01.ply'
    test_ply = './test_ply_crossbeam/mask_000.ply'

    suction_pose_solver = Solve_Suciton_Pose()
    # suction_pose_solver.load_point_cloud_from_ply(test_ply)

    t1 =time.time()
    suction_pose_solver.solve(test_ply,print_log=False,is_vis=True)
    t2 =time.time()
    print(f'time costs: {t2-t1}')