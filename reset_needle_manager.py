import numpy as np
from scipy.spatial.transform import Rotation as R

from utils_transformation import pose_to_T
from step_utils_reward import RewardUtils
from Kinematics.dvrkKinematics import dvrkKinematics

class NeedleManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.num_gp = self.config_manager.num_gp
        self.num_ga = self.config_manager.num_ga
        self.Tw_rb1 = self.config_manager.Tw_rb1
        self.Tw_rb2 = self.config_manager.Tw_rb2
        self.Tw_cam = self.config_manager.Tw_cam

        self.needle_radius = config_manager.config['needle_radius']

        self.gp_lst = config_manager.gp_lst
        self.ga_lst = config_manager.ga_lst
        self.gp_theta_lst = config_manager.gp_theta_lst

        self.ho_ori_lst = config_manager.ho_ori_lst
        self.dvrkkin = dvrkKinematics()

    def generate_needle_pose(self):
        '''
        :return: numpy.ndarray: randome needle pose [x, y, z, qx, qy, qz, qw]
        '''
        # Make sure there exists at least one state that can be reached without joint limit violation.
        pose = np.array([10, 10, 10, 0, 0, 0, 1])

        while not self._is_feasible(pose):
            # print(pose)
            # lower_lim = np.array([-0.05, 0.03, 0.0])
            # upper_lim = np.array([0.05, 0.08, 0.0])
            lower_lim = np.array([-0.1, 0.0, 0.0])
            upper_lim = np.array([0.1, 0.15, 0.0])
            pos = np.random.uniform(lower_lim, upper_lim)
            # Orientation
            # orientation은 z축이 무조건 1 or -1로 고정.
            z_axis = np.random.choice([1, -1])
            z_axis = np.array([0, 0, z_axis])
            rand_vec = np.random.randn(3)
            proj = np.dot(rand_vec, z_axis) * z_axis
            x_axis = rand_vec - proj
            x_axis = x_axis / np.linalg.norm(x_axis)

            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)

            ori_random = np.column_stack((x_axis, y_axis, z_axis))
            TT = np.identity(4)
            TT[:3, :3] = ori_random
            TT[:3, -1] = pos
            TT = np.linalg.inv(self.Tw_cam) @ TT
            pos = TT[:3, -1]

            # ori_random = R.from_matrix(ori_random).as_quat()
            ori_random = R.from_matrix(TT[:3, :3]).as_quat()
            pose = np.concatenate((pos, ori_random))

        print("Tw_no_init:\n", pose_to_T(pose))

        # Visualize
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # utils_transformation.plot_needle(ax, utils_transformation.pose_to_T(pose))
        # plt.legend()
        # plt.axis('equal')
        # ax.grid(True)
        # plt.show()

        return pose



    def Tno_ntarg(self, gp_idx, ga_idx):
        """바늘 원점에서 타겟까지의 변환 행렬 계산"""
        Tno_ntarg = self.gp_lst[gp_idx] @ self.ga_lst[ga_idx]
        theta = self.gp_theta_lst[gp_idx]
        Tno_ntarg[:3, -1] = self.needle_radius * np.array([np.cos(theta), np.sin(theta), 0]).T
        return Tno_ntarg


    def _is_feasible(self, needle_pose):
        '''
        Args:
            needle_pose: numpy.ndarray: randome needle pose [x, y, z, qx, qy, qz, qw]
        Returns: True asap if at least there exists one feasible state
        '''
        Tw_no = pose_to_T(needle_pose)
        for p in range(self.num_gp):
            for a in range(self.num_ga):
                Tw_targ = Tw_no @ self.Tno_ntarg(p, a)
                Trb1_targ = np.linalg.inv(self.Tw_rb1) @ Tw_targ
                targ_joint = self.dvrkkin.ik(Trb1_targ)
                if not RewardUtils.joint_isin_limit(targ_joint):
                    return True
                Trb2_targ = np.linalg.inv(self.Tw_rb2) @ Tw_targ
                targ_joint = self.dvrkkin.ik(Trb2_targ)
                if not RewardUtils.joint_isin_limit(targ_joint):
                    return True
        return False


if __name__ == "__main__":
    from _config import env_config
    from reset_config_manager import ConfigurationManager
    config_manager = ConfigurationManager(env_config)
    needle_manager = NeedleManager(config_manager)

    print(needle_manager.generate_needle_pose())
