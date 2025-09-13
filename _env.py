import gymnasium as gym
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

from _reset_orchestrator import ResetOrchestrator
from _step_orchestrator import StepOrchestrator
from typing import Dict, Tuple


class NeedleHandoverEnv(gym.Env):
    def __init__(self, env_config: dict, step_config: dict):
        super().__init__()
        self.config = env_config
        self.step_config = step_config

        # 모듈 초기화
        self.reset_orchestrator = ResetOrchestrator(env_config=self.config)
        self.step_orchestrator = StepOrchestrator(
            env_config=self.config,
            step_config=step_config,
            reset=self.reset_orchestrator
        )

        # 런타임 변수
        self.state : np.ndarray = self.reset_orchestrator.state
        self.goal_state: np.ndarray = self.reset_orchestrator.goal_state
        self.Tw_no = self.reset_orchestrator.Tw_needle
        self.num_step = 0
        self.prev_action_lst = []
        self.joint_pos_1 = self.reset_orchestrator.joint_pos_1
        self.joint_pos_2 = self.reset_orchestrator.joint_pos_2

        # 공간 정의
        self.num_action = self.reset_orchestrator.config_manager.num_action
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            # shape=(10 if self.include_needle_pose else 3,),
            shape = (10,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_action)
        self.state_dim = self.reset_orchestrator.config_manager.config.get('state_dim')


    def reset(self, **kwargs):
        self.reset_orchestrator = ResetOrchestrator(env_config=self.config)
        self.prev_action_lst = []
        self.num_step = 0
        self.Tw_no = self.reset_orchestrator.Tw_needle
        self.joint_pos_1 = self.reset_orchestrator.joint_pos_1
        self.joint_pos_2 = self.reset_orchestrator.joint_pos_2
        self.goal_state = self.reset_orchestrator.goal_state
        self.state = self.reset_orchestrator.state
        self.obs = copy.deepcopy(self.state)
        return self.obs


    def step(self, action: int):
        self.num_step += 1

        new_state, reward, done, new_Tw_no, joint_pos_1, joint_pos_2 = self.step_orchestrator.step(
            action=int(action),
            state=self.state,
            Tw_no=self.Tw_no,   # 코드 바뀌면서 필요 없을 것 같음. 확인 필요.
            num_step=self.num_step,
            prev_action_lst=self.prev_action_lst,
            goal_state=self.goal_state
        )


        self.joint_pos_1 = joint_pos_1
        self.joint_pos_2 = joint_pos_2

        # 환경 상태 업데이트
        self.prev_action_lst.append(action)
        self.Tw_no = new_Tw_no

        # 관측값 생성
        self.state = new_state
        self.obs = copy.deepcopy(self.state)

        return self.obs, float(reward), bool(done)


if __name__ == "__main__":
    from _config import env_config, step_config

    # include_needle_pose = True 테스트
    step_config = step_config.copy()
    env_needle_pose = NeedleHandoverEnv(env_config=env_config, step_config=step_config)
    print("Pose env observation shape:", env_needle_pose.observation_space.shape)
    env_needle_pose.reset()



