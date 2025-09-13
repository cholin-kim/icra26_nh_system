import numpy as np
from scipy.spatial.transform import Rotation as R



class NeedleManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.num_gp = self.config_manager.num_gp
        self.num_ga = self.config_manager.num_ga
        self.Tw_rb1 = self.config_manager.Tw_rb1
        self.Tw_rb2 = self.config_manager.Tw_rb2

        self.needle_radius = config_manager.config['needle_radius']

        self.gp_lst = config_manager.gp_lst
        self.ga_lst = config_manager.ga_lst
        self.gp_theta_lst = config_manager.gp_theta_lst

        self.ho_ori_lst = config_manager.ho_ori_lst



    def Tno_ntarg(self, gp_idx, ga_idx):
        """바늘 원점에서 타겟까지의 변환 행렬 계산"""
        Tno_ntarg = self.gp_lst[gp_idx] @ self.ga_lst[ga_idx]
        theta = self.gp_theta_lst[gp_idx]
        Tno_ntarg[:3, -1] = self.needle_radius * np.array([np.cos(theta), np.sin(theta), 0]).T
        return Tno_ntarg




if __name__ == "__main__":
    from _config import env_config
    from reset_config_manager import ConfigurationManager
    config_manager = ConfigurationManager(env_config)
    needle_manager = NeedleManager(config_manager)

    # print(needle_manager.generate_needle_pose())
    print(needle_manager.Tno_ntarg(2, 3).tolist())
