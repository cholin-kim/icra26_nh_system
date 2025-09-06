from typing import Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np


from rl_sth.step_utils_reward import RewardUtils
from rl_sth.Kinematics.dvrkKinematics import dvrkKinematics as dvrkkin
from rl_sth.Kinematics import dvrkVar



class RewardStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        """
        :param context = {
                   'state': state,            # InvalidGP/GHStrategy가 사용
                   'action': action,          # InvalidGP/GHStrategy가 사용    !!! unpacked  !!!
                   'goal_state': goal_state,  # ReachedGoalStrategy가 사용
                   'num_step': num_step,      # StepLimitStrategy가 사용
                   'prev_action_lst': prev_actions  # LoopStrategy가 사용
                   'needle_pose' : T_to_pose(Tw_no_new) # needle_origin_w after the action
                   'joint_pos_1',
                   'joint_pos_2'
                    }
        :return: (bool, reward), reward w/o weight
        """
        pass

class JointLimitStrategy(RewardStrategy):
    """관절 한계 위반 검사 전략"""

    def __init__(self, joint_limit_checker: callable, reward: float):
        """
        Args:
            joint_limit_checker: q -> bool 검사 함수
            reward: 위반 시 reward(<0)
        """
        self.checker = joint_limit_checker
        self.reward = reward

    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        q1, q2 = context['joint_pos_1'], context['joint_pos_2']
        violated = self.checker(q1) or self.checker(q2)
        return violated, (self.reward if violated else 0.0)

class JointMarginStrategy(RewardStrategy):
    def __init__(self):
        pass

    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        q1, q2 = context['joint_pos_1'], context['joint_pos_2']
        if RewardUtils.joint_isin_limit(q1) or RewardUtils.joint_isin_limit(q2):
            return False, 0.0
        else:
            joint_range_mid = (dvrkVar.joint_range_upper_limit + dvrkVar.joint_range_lower_limit) / 2
            joint_margin1 = 4 * ((q1 - joint_range_mid) ** 2) / (
                    (dvrkVar.joint_range_upper_limit - dvrkVar.joint_range_lower_limit) ** 2)
            joint_margin2 = 4 * ((q2 - joint_range_mid) ** 2) / (
                    (dvrkVar.joint_range_upper_limit - dvrkVar.joint_range_lower_limit) ** 2)

            # return True, (sum(joint_margin1) / 6 + sum(joint_margin2) / 6)
            # return True, (sum(joint_margin1[4:]) / 2 + sum(joint_margin2[4:]) / 2)
            return True, 1- (sum(joint_margin1[4:]) / 2 + sum(joint_margin2[4:]) / 2)



class CollisionStrategy(RewardStrategy):
    """충돌 위반 검사 전략"""

    def __init__(self, collision_checker: callable, reward: float):
        """
        Args:
            collision_checker: (T1, T2) -> bool 검사 함수
            penalty: 위반 시 페널티
        """
        self.checker = collision_checker
        self.reward = reward


    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        prev_action_lst = context['prev_action_lst']
        if len(prev_action_lst) == 0:   # first step
            return False, 0.0
        else:
            T1 = dvrkkin.fk(context['joint_pos_1'])[0][-1]
            T2 = dvrkkin.fk(context['joint_pos_2'])[0][-1]
            violated = self.checker(T1, T2)
            return violated, (self.reward if violated else 0.0)

### Sihyeoung edited
class CollisionStrategy_ground(RewardStrategy):
    def __init__(self, collision_checker_ground: callable, reward: float):
        self.checker_ground = collision_checker_ground
        self.reward = reward

    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        ground_threshold = 0.008
        rb_z = 0.2
        Tbs1 = dvrkkin.fk(context['joint_pos_1'])[0]
        Tbs2 = dvrkkin.fk(context['joint_pos_2'])[0]

        violated = self.checker_ground(Tbs1, Tbs2, ground_threshold, rb_z)

        return violated, (self.reward if violated else 0.0)
###

class ReachedGoalStrategy(RewardStrategy):
    def __init__(self, reward: float):
        self.reward = reward

    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        goal_state = np.array(context['goal_state'][:3], dtype=int)
        # state = context['state']
        state = np.array(context['action'][:3], dtype=int)   # state0 = (0, 0, 0, ~~)로 바꾸면서, action 즉, next state를 고려해야 하여 수정되었다.

        if np.array_equal(goal_state, state):
            return True, self.reward
        else:
            return False, 0.0


class StepLimitStrategy(RewardStrategy):
    def __init__(self, reward: float, step_limit: int):
        self.reward = reward
        self.step_limit = step_limit

    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        num_step = context['num_step']
        violated = num_step >= self.step_limit
        return violated, (self.reward if violated else 0.0)


class InvalidGPStrategy(RewardStrategy):
    def __init__(self, reward: float):
        self.reward = reward

    def evaluate(self, context: Dict) -> Tuple[bool, float]:
        state = context['state']
        action = context['action']
        violated = (int(state[0]) == int(action[0]))
        return violated, (self.reward if violated else 0.0)




class RewardCalculator:
    def __init__(self, step_config):
        self.termination_strategies = []
        self.truncation_strategies = []
        self.else_strategies = []

        termination_reward_map = step_config.get('termination_reward_map')
        truncation_reward_map = step_config.get('truncation_reward_map')

        # self.truncation_strategy_names = []
        # self.termination_strategy_names = []

        self.transition_reward = step_config.get('transition')
        step_limit = step_config.get('max_steps')

        if 'step_limit' in truncation_reward_map:   # reward 를 중복해서 확인하지 않을 때는 step_limit을 첫번째로 확인해야 한다.
            self.add_truncation_strategy(
                StepLimitStrategy(reward=truncation_reward_map['step_limit'], step_limit=step_limit)
            )
            # self.truncation_strategy_names.append('StepLimitStrategy')

        if 'invalid_grasping_point' in truncation_reward_map:
            self.add_truncation_strategy(
                InvalidGPStrategy(reward=truncation_reward_map['invalid_grasping_point'])
            )
            # self.truncation_strategy_names.append('InvalidGPStrategy')

        if 'collision_ground' in truncation_reward_map:
            self.add_truncation_strategy(
                CollisionStrategy_ground(collision_checker_ground=RewardUtils.check_collision_ground,
                                        reward=truncation_reward_map['collision_ground'])
            )
            # self.truncation_strategy_names.append('CollisionStrategy_ground')


        if 'joint_limit' in truncation_reward_map:
            self.add_truncation_strategy(
                JointLimitStrategy(joint_limit_checker=RewardUtils.joint_isin_limit,
                                   reward=truncation_reward_map['joint_limit'])
            )
            # self.truncation_strategy_names.append('JointLimitStrategy')


        if 'reached_goal' in termination_reward_map:
            self.add_termination_strategy(
                ReachedGoalStrategy(reward=termination_reward_map['reached_goal'])
            )
            # self.termination_reward_map.append('ReachedGoalStrategy')


        if 'joint_margin' in step_config:
            self.add_strategy(
                JointMarginStrategy()
            )


    def add_strategy(self, strategy: RewardStrategy):
        """전략 추가 시 weight 유효성 검증"""
        if not isinstance(strategy, RewardStrategy):
            raise TypeError("Strategy must inherit from RewardStrategy")
        # self.strategies.append(strategy)
        self.else_strategies.append(strategy)

    def add_truncation_strategy(self, strategy: RewardStrategy):
        """전략 추가 시 weight 유효성 검증"""
        if not isinstance(strategy, RewardStrategy):
            raise TypeError("Strategy must inherit from RewardStrategy")
        self.truncation_strategies.append(strategy)

    def add_termination_strategy(self, strategy: RewardStrategy):
        """전략 추가 시 weight 유효성 검증"""
        if not isinstance(strategy, RewardStrategy):
            raise TypeError("Strategy must inherit from RewardStrategy")
        self.termination_strategies.append(strategy)



    def evaluate(self, context: Dict) -> Tuple[bool, str, float]:
        total_reward = 0.0
        truncated = False
        terminated = False
        reward_type = {}

        # 1. Check Truncation
        for strategy in self.truncation_strategies:
            is_violated, reward = strategy.evaluate(context)
            if is_violated:
                total_reward += reward
                reward_type[strategy.__class__.__name__] = reward
                truncated = True
                # add transition reward
                total_reward += self.transition_reward  # small penalty for each step
                reward_type['Transition'] = self.transition_reward
                return terminated, truncated, reward_type, total_reward
            else:
                continue

        # 2. Check Termination
        assert not truncated
        for strategy in self.termination_strategies:
            is_violated, reward = strategy.evaluate(context)
            if is_violated:
                reward_type[strategy.__class__.__name__] = reward
                total_reward += reward
                terminated = True
                # no transition reward added
                # check and add joint margin strategy
                for strategy in self.else_strategies:
                    is_violated, reward = strategy.evaluate(context)
                    if is_violated:
                        reward_type[strategy.__class__.__name__] = reward
                        total_reward += reward
                return terminated, truncated, reward_type, total_reward

        # 3. no truncation nor termination signal captured
        assert not truncated
        assert not terminated
        total_reward += self.transition_reward  # small penalty for each step
        reward_type['Transition'] = self.transition_reward
        for strategy in self.else_strategies:
            is_violated, reward = strategy.evaluate(context)
            if is_violated:
                reward_type[strategy.__class__.__name__] = reward
                total_reward += reward  # small penalty for each step
        return terminated, truncated, reward_type, total_reward