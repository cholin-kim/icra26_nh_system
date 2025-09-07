import numpy as np
from scipy.spatial.transform import Rotation as R




class ConfigurationManager:
    def __init__(self, env_config):
        self.config = env_config

        self.Tw_rb1 = env_config['Tw_rb1']
        self.Tw_rb2 = env_config['Tw_rb2']

        # Grasping parameters
        self.num_gp = env_config['num_gp']
        self.num_ga = env_config['num_ga']
        self.num_gh = env_config['num_gh']


        # Handover orientation parameters
        self.num_ho_ori_plane = env_config['num_ho_ori_plane']
        self.num_ho_plane = env_config['num_ho_plane']
        self.num_ho_ori = env_config['num_ho_ori']
        assert self.num_ho_ori == self.num_ho_ori_plane * self.num_ho_plane

        # State space size
        self.state_dim = env_config['state_dim']

        # Handover position
        self.ho_pos = env_config['ho_pos']

        self.initial_joint_pos_1 = env_config['initial_joint_pos_1']
        self.initial_joint_pos_2 = env_config['initial_joint_pos_2']

        self.needle_radius = env_config['needle_radius']
        self.goal_state = env_config['goal_state']
        self.init_needle_pos = env_config['init_needle_pos']

        # Initialize predefined lists
        self._initialize_predefined_lists() # define self.ho_ori_lst

        self.num_action = self.num_gp * self.num_ga * self.num_gh * self.num_ho_ori


    def _initialize_predefined_lists(self):
        """Initialize all predefined lists from configuration"""
        # Grasping point list
        self.gp_theta_lst = [(np.pi / (self.num_gp + 1)) * (i + 1) for i in range(self.num_gp)]
        self.gp_lst = np.zeros((self.num_gp, 4, 4))
        self.gp_lst[:, :3, :3] = R.from_euler("Z", self.gp_theta_lst, degrees=False).as_matrix()
        self.gp_lst[:, 3, 3] = 1

        # Grasping angle list
        self.ga_lst = np.zeros((self.num_ga, 4, 4))
        assert int(self.num_ga % 2) == 0
        num_ga_half = int(self.num_ga / 2)

        angle = (2 * np.pi) / num_ga_half
        ga_lst_R = []
        for p in range(num_ga_half):
            ga_lst_R.append(R.from_euler("Y", angle * p, degrees=False).as_matrix())
        for p in range(num_ga_half):
            ga_lst_R.append(ga_lst_R[p] @ R.from_euler("Z", np.pi, degrees=False).as_matrix())

        self.ga_lst[:, :3, :3] = ga_lst_R
        self.ga_lst[:, 3, 3] = 1

        # Handover orientation list
        ## needs to be changed if num_ho_plane >= 2

        # if self.num_ho_plane == 1:
        #     self.ho_ori_lst = np.zeros((self.num_ho_ori, 4, 4))
        #     assert self.num_ho_ori % 2 == 0
        #     num_ho_ori_half = int(self.num_ho_ori / 2)
        #
        #     angle2 = (2 * np.pi) / num_ho_ori_half
        #     ho_ori_lst_R = []
        #     for q in range(num_ho_ori_half):
        #         ho_ori_lst_R.append(R.from_euler("Z", angle2 * q, degrees=False).as_matrix())
        #     for q in range(num_ho_ori_half):
        #         ho_ori_lst_R.append(ho_ori_lst_R[q] @ R.from_euler("Y", np.pi, degrees=False).as_matrix())
        #     ## tmp
        #     # for r in range(len(ho_ori_lst_R)):
        #     #     ho_ori_lst_R[r] = R.from_euler("Y", np.pi/2, degrees=False).as_matrix() @ ho_ori_lst_R[r]
        #
        # elif self.num_ho_plane == 2:
        #     self.ho_ori_lst = np.zeros((self.num_ho_ori, 4, 4))
        #     assert self.num_ho_ori % 4 == 0
        #     num_ho_ori_block = int(self.num_ho_ori / 4)
        #
        #     angle2 = (2 * np.pi) / num_ho_ori_block
        #     ho_ori_lst_R = []
        #     for q in range(num_ho_ori_block):
        #         ho_ori_lst_R.append(R.from_euler("Z", angle2 * q, degrees=False).as_matrix())
        #     for q in range(num_ho_ori_block):
        #         ho_ori_lst_R.append(ho_ori_lst_R[q] @ R.from_euler("Y", np.pi, degrees=False).as_matrix())
        #     for q in range(len(ho_ori_lst_R)):
        #         ho_ori_lst_R.append(R.from_euler("Y", np.pi/2, degrees=False).as_matrix() @ ho_ori_lst_R[q])
        #
        # elif self.num_ho_plane == 3:
        #     self.ho_ori_lst = np.zeros((self.num_ho_ori, 4, 4))
        #     assert self.num_ho_ori % 4 == 0
        #     num_ho_ori_block = int(self.num_ho_ori / 4)
        #
        #     angle2 = (2 * np.pi) / num_ho_ori_block
        #     ho_ori_lst_R = []
        #     for q in range(num_ho_ori_block):
        #         ho_ori_lst_R.append(R.from_euler("Z", angle2 * q, degrees=False).as_matrix())
        #     for q in range(num_ho_ori_block):
        #         ho_ori_lst_R.append(ho_ori_lst_R[q] @ R.from_euler("Y", np.pi, degrees=False).as_matrix())
        #     tmp = len(ho_ori_lst_R)
        #     for q in range(tmp):
        #         ho_ori_lst_R.append(R.from_euler("Y", np.pi/2, degrees=False).as_matrix() @ ho_ori_lst_R[q])
        #     for q in range(tmp):
        #         ho_ori_lst_R.append(R.from_euler("X", np.pi/2, degrees=False).as_matrix() @ ho_ori_lst_R[q])

        if self.num_ho_plane == 1:
            self.ho_ori_lst = np.zeros((self.num_ho_ori, 4, 4))
            assert self.num_ho_ori % 2 == 0
            num_ho_ori_half = int(self.num_ho_ori / 2)

            angle2 = (2 * np.pi) / num_ho_ori_half
            ho_ori_lst_R = []
            for q in range(num_ho_ori_half):
                ho_ori_lst_R.append(R.from_euler("Z", angle2 * q, degrees=False).as_matrix())
            for q in range(num_ho_ori_half):
                ho_ori_lst_R.append(ho_ori_lst_R[q] @ R.from_euler("Y", np.pi, degrees=False).as_matrix())


        elif self.num_ho_plane == 2:
            self.ho_ori_lst = np.zeros((self.num_ho_ori, 4, 4))
            assert self.num_ho_ori % 4 == 0
            num_ho_ori_block = int(self.num_ho_ori / 4)

            angle2 = (2 * np.pi) / num_ho_ori_block
            ho_ori_lst_R_tmp = []
            ho_ori_lst_R = []
            for q in range(num_ho_ori_block):
                ho_ori_lst_R_tmp.append(R.from_euler("Z", angle2 * q, degrees=False).as_matrix())
            for q in range(num_ho_ori_block):
                ho_ori_lst_R_tmp.append(ho_ori_lst_R_tmp[q] @ R.from_euler("Y", np.pi, degrees=False).as_matrix())

            for q in range(len(ho_ori_lst_R_tmp)):
                    ho_ori_lst_R.append(R.from_euler("Y", np.pi/6, degrees=False).as_matrix() @ ho_ori_lst_R_tmp[q])
            for q in range(len(ho_ori_lst_R_tmp)):
                ho_ori_lst_R.append(R.from_euler("Y", -np.pi/6, degrees=False).as_matrix() @ ho_ori_lst_R_tmp[q])

        self.ho_ori_lst[:, :3, :3] = ho_ori_lst_R
        self.ho_ori_lst[:, 3, 3] = 1

if __name__ == "__main__":
    from _config import env_config
    config_manager = ConfigurationManager(env_config=env_config)
    from utils_transformation import *
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for Tw_needle in config_manager.ho_ori_lst:

        plot_needle(ax, Tw_needle)
        # plt.legend()
        plt.axis('equal')
        ax.grid(True)
        # ax.set_zlim(bottom=0)
    plt.show()
    # plt.close(fig)
