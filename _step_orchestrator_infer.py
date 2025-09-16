from typing import Tuple, List, Dict, Callable

from DQN_cam.step_reward_calculator import RewardCalculator
from DQN_cam.utils_transformation import *
from Kinematics.dvrkKinematics import dvrkKinematics


class StepOrchestrator:
    """
    DQN_cam 환경의 step 실행을 총괄하는 오케스트레이터
    RewardCalculator를 중심으로 모든 step 로직을 통합 관리
    """

    def __init__(self, step_config: Dict, env_config: Dict, reset:Callable):

        reward_map = step_config.get('reward_map')
        self.reset = reset
        self.reward_calculator = RewardCalculator(step_config)

        # parameters, defined in reset
        self.num_gp = self.reset.config_manager.num_gp
        self.num_ga = self.reset.config_manager.num_ga
        self.num_gh = self.reset.config_manager.num_gh
        self.num_ho_ori = self.reset.config_manager.num_ho_ori


        self.ho_ori_lst = self.reset.config_manager.ho_ori_lst

        self.gp_lst = self.reset.config_manager.gp_lst
        self.gp_theta_lst = self.reset.config_manager.gp_theta_lst
        self.ga_lst = self.reset.config_manager.ga_lst

        # self.needle_radius = self.reset.config_manager.needle_radius
        self.Tw_rb1 = self.reset.config_manager.Tw_rb1
        self.Tw_rb2 = self.reset.config_manager.Tw_rb2
        self.Tw_cam = self.reset.config_manager.Tw_cam
        # self.joint_pos_1 = self.reset.joint_pos_1
        # self.joint_pos_2 = self.reset.joint_pos_2
        # self.state = self.reset.state   # state 0 = [gp, ga, gh, needle_pose(7,)]

        self.dvrkkin = dvrkKinematics()
        self.env_config = env_config


    def step(self,
             action: int,
             state: np.ndarray,
             Tw_no: np.ndarray,
             num_step: int,
             prev_action_lst: List,
             goal_state: np.ndarray,
             ) -> Tuple[np.ndarray, float, bool, np.ndarray, np.ndarray]:
        """
        Args:
            action: action idx
            state: current state[gp, ga, gh, needle_pose(7,)]
            joint_pos_1: robot_l current joint position
            joint_pos_2: robot_r current joint position
            num_step
            goal_state

        Returns:
        """

        # Step1: Action decoding, Update state
        action_unpacked = self._unpack_action(action)
        next_gp, next_ga, next_gh, ho_ori_idx = action_unpacked

        ### invalid grasping hand를 rule based로 정리하기 위함
        if num_step == 1:
            pass
        else:
            if next_gh == int(state[2]):
                next_gh = abs(1-int(state[2]))
            else:
                pass
        ###
        action_unpacked = (next_gp, next_ga, next_gh, ho_ori_idx)
        if num_step == 1:
            print("action_unpacked:", next_gp, next_ga, next_gh, ho_ori_idx)
        else:
            print("\naction_unpacked:", next_gp, next_ga, next_gh, ho_ori_idx)


        new_state = np.array([next_gp, next_ga, next_gh])
        Tw_no_new = self._get_Tw_no(ho_ori_idx)
        # print("Tw_no_new:\n", Tw_no_new)

        Tno_targ = self.reset.needle_manager.Tno_ntarg(next_gp, next_ga)


        # Step2: Early termination check & reward calculation
        context = {'state': state,            # InvalidGP/GHStrategy가 사용
                   'action': action_unpacked, # InvalidGP/GHStrategy가 사용    !!! unpacked  !!!
                   'goal_state': goal_state,  # ReachedGoalStrategy가 사용
                   'num_step': num_step,      # StepLimitStrategy가 사용
                   'prev_action_lst': prev_action_lst,
                   # 'needle_pose' : T_to_pose(Tw_no_new) # needle_origin_w after the action
        }
        ##
        left_to_right = False
        if state[2] != next_gh:
            if state[2] == 0 and next_gh == 1:
                left_to_right = True


        if num_step == 1:   # Tw_pu, Tw_pu_after
            # rb1, rb2가 아니라, pickup, after the pickup(ho_pos 이동) 으로 Tw_ntarg1, Tw_ntarg2, joint_pos_1, joint_pos_2 결정
            ## after the pickup, use needle orientation(action_unpacked[3:] to get ready for the first handover
            Tw_targ1 = Tw_no @ Tno_targ # cam 기준
            Tw_targ2 = Tw_no_new @ Tno_targ
            # print("Tw_pickup:\n", Tw_targ1)
            # print("Tw_pickup_after:\n", Tw_targ2)


            if action_unpacked[2] == 0: # left arm is picking up
                Trb_targ = np.linalg.inv(self.Tw_rb1) @ Tw_targ1
                joint_pos_1 = self.dvrkkin.ik(Trb_targ)
                Trb_targ2 = np.linalg.inv(self.Tw_rb1) @ Tw_targ2
                joint_pos_2 = self.dvrkkin.ik(Trb_targ2)
            elif action_unpacked[2] == 1: # right arm is picking up
                Trb_targ = np.linalg.inv(self.Tw_rb2) @ Tw_targ1
                joint_pos_1 = self.dvrkkin.ik(Trb_targ)
                Trb_targ2 = np.linalg.inv(self.Tw_rb2) @ Tw_targ2
                joint_pos_2 = self.dvrkkin.ik(Trb_targ2)

            context['joint_pos_1'] = joint_pos_1
            context['joint_pos_2'] = joint_pos_2
            # print("targ_joint_pos_l:", joint_pos_1)
            # print("targ_joint_pos_r:", joint_pos_2)

            last_target = Tw_targ2





        else:
            # Tw_giver = Tw_no_new @ self.Tno_targ_prime
            Tw_giver = Tw_no_new @ self.reset.needle_manager.Tno_ntarg(int(state[0]), int(state[1]))
            Tw_receiver = Tw_no_new @ Tno_targ
            if left_to_right:
                Tw_targ1, Tw_targ2 = Tw_giver, Tw_receiver
            else:
                Tw_targ1, Tw_targ2 = Tw_receiver, Tw_giver

            # print("Tw_targ_l:\n", Tw_targ1)
            # print("Tw_targ_r:\n", Tw_targ2)

            Trb1_ntarg1 = np.linalg.inv(self.Tw_rb1) @ Tw_targ1
            Trb2_ntarg2 = np.linalg.inv(self.Tw_rb2) @ Tw_targ2


            joint_pos_1 = self.dvrkkin.ik(Trb1_ntarg1)
            joint_pos_2 = self.dvrkkin.ik(Trb2_ntarg2)
            # print("targ_joint_pos_l:", joint_pos_1)
            # print("targ_joint_pos_r:", joint_pos_2)

            context['joint_pos_1'] = joint_pos_1   # after the action
            context['joint_pos_2'] = joint_pos_2

            last_target = Tw_targ2 if left_to_right else Tw_targ1




        # ## add disturbance
        # Tnoise = self._noise(deg=30)
        # Ttarg_prime_no = Tnoise @ np.linalg.inv(Tno_targ)
        # Tw_no_new_noise = last_target @ Ttarg_prime_no
        # self.Tno_targ_prime = np.linalg.inv(Ttarg_prime_no)  # applied after the action


        terminated, truncated, reward_type, reward = self.reward_calculator.evaluate(context)
        # print("reward_type:", reward_type)

        # print("Tw_no_new_noise:\n", Tw_no_new_noise)
        # new_state = np.append(new_state, T_to_pose(Tw_no_new_noise))
        new_state = np.append(new_state, T_to_pose(Tw_no_new))
        # Visualization
        # import matplotlib.pyplot as plt
        # ax = visualize_poses([T_to_pose(Tw_targ1), T_to_pose(Tw_targ2)])
        # plot_needle(ax, Tw_no)
        # plot_needle(ax, Tw_no_new)
        # plot_needle(ax, Tw_no_new_noise)
        # plt.legend()
        # plt.axis('equal')
        # ax.grid(True)
        # plt.show()


        # 7단계: 결과 반환
        if terminated or truncated:
            return new_state, reward, True, Tw_no_new, Tw_targ1, Tw_targ2, reward_type
        else:
            return new_state, reward, False, Tw_no_new, Tw_targ1, Tw_targ2, reward_type

    '''
    Step Utils
    '''
    # def _targ_noise(self, std_deg=5.0):
    #     '''
    #     Returns: Ttarg_targ'(noise with xy frame)
    #     '''
    #     x_noise = np.deg2rad(np.random.uniform(-std_deg, std_deg))
    #     y_noise = np.deg2rad(np.random.uniform(-std_deg, std_deg))
    #     R_noise = R.from_euler("XY", [x_noise, y_noise])
    #     Ttarg_targ_prime = np.identity(4)
    #     Ttarg_targ_prime[:3, :3] = R_noise
    #     return Ttarg_targ_prime

    def _unpack_action(self, action: int) -> Tuple[int, int, int, int]:
        """
        액션 인덱스를 개별 컴포넌트로 디코딩

        Returns:
            (next_gp, next_ga, next_gh, ho_ori_idx)
        """
        action_shape = (self.num_gp, self.num_ga, self.num_gh, self.num_ho_ori)

        indices = np.unravel_index(action, action_shape)
        next_gp, next_ga, next_gh, ho_ori_idx = indices
        return int(next_gp), int(next_ga), int(next_gh), int(ho_ori_idx)

    def _get_Tw_no(self, ho_ori_idx):
        Tw_no = self._ho_ori_HO_ORI(ho_ori_idx)
        Tw_no[:3, -1] = self.reset.ho_pos
        Tw_no = np.linalg.inv(self.Tw_cam) @ Tw_no
        return Tw_no



    def _ho_ori_HO_ORI(self, ho_ori_idx):
        assert ho_ori_idx < self.num_ho_ori, "ho_ori_idx must be less than num_of_ho_ori"
        HO_ORI = self.ho_ori_lst[ho_ori_idx]
        return HO_ORI


    def _noise(self, deg=5):
        theta_x = np.deg2rad(np.random.uniform(-deg, deg))
        theta_y = np.deg2rad(np.random.uniform(-deg, deg))
        T = np.identity(4)
        T[:3, :3] = R.from_euler('XY', [theta_x, theta_y]).as_matrix()
        return T
