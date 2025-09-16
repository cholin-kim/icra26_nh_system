from DQN_cam.reset_config_manager import ConfigurationManager
from DQN_cam.reset_needle_manager import NeedleManager
from DQN_cam.utils_transformation import pose_to_T
import numpy as np

'''
All variables, parameters should be called from configuration_manger. configuration manger initializes the list when called.
'''

class ResetOrchestrator:
    def __init__(self, env_config):
        # Initialize
        self.config_manager = ConfigurationManager(env_config)
        self.needle_manager = NeedleManager(self.config_manager)
        self.goal_state= self.config_manager.goal_state

        needle_pose = self.needle_manager.generate_needle_pose()
        self.Tw_needle = pose_to_T(needle_pose)
        # self.Tc_needle = pose_to_T(needle_pose)
        self.ho_pos = self.config_manager.ho_pos

        # set joint position
        self.joint_pos_1 = self.config_manager.initial_joint_pos_1
        self.joint_pos_2 = self.config_manager.initial_joint_pos_2

        # update needle pose after the pickup
        # reset orchestrator를 불러오면, robot joint pos reset, needle pose 는 set in random 상태.
        # self.state = np.array([0, 0, 0])
        self.state = np.array([-10, -10, -10])
        self.state = np.append(self.state, needle_pose)
        self.state_dim = self.config_manager.state_dim



if __name__ == "__main__":
    from DQN_cam._config import env_config
    reset = ResetOrchestrator(env_config=env_config)
    print('s0:', reset.state)