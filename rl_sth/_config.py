import numpy as np

# TW_RB1 = np.identity(4)
# TW_RB2 = np.identity(4)
#
# TW_RB1[:3, -1] = np.array([-0.12, 0, 0.14]).T
# TW_RB2[:3, -1] = np.array([0.12, 0, 0.14]).T

TW_RB1 = np.array(
[[ 9.99999982e-01, -1.45821419e-04, -1.20288867e-04, -1.17465392e-01],
 [-1.60143467e-06, -6.42851847e-01,  7.65990537e-01,  6.73936401e-02],
 [-1.89025748e-04, -7.65990523e-01, -6.42851836e-01,  1.82525761e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


TW_RB2 = np.array(
[[ 9.99999982e-01, -1.45821419e-04, -1.20288867e-04,  1.22534604e-01],
 [-1.60143467e-06, -6.42851847e-01,  7.65990537e-01,  6.73932558e-02],
 [-1.89025748e-04, -7.65990523e-01, -6.42851836e-01,  1.82480395e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


env_config = {
    # === Robot base transformation matrices ===
    'Tw_rb1': TW_RB1,
    'Tw_rb2': TW_RB2,

    # === Grasping parameters ===
    'num_gp': 5,  # grasping points
    'num_ga': 4,  # grasping angles,
    'num_gh': 2,  # grasping hands

    # === Handover orientation parameters ===
    'num_ho_ori_plane': 4,     # num_ho_ori in one plane
    'num_ho_plane': 2,
    'num_ho_ori': 8,   # num_ho_ori_plane * num_ho_plane

    # === Handover Position(fixed) ===
    'ho_pos': [0.0, 0.1, 0.05],


    # === State space size ===
    'state_dim': 10,

    # === Needle physical property ===
    'needle_radius': 0.024,  # (m), largest needle diameter 48mm
    # === Initial Needle Position ===
    'init_needle_pos': [0.0, 0.0, 0.0],



    # === Goal state ===
    # 'goal_state': [1, 0, 1, 0.0, 0.1, 0.05, 0.0, 0.0, 0.0, 1.0],  # 기본 목표 상태
    'goal_state': [1, 0, 1, 0.0, 0.1, 0.05, 0, 0.707, 0, 0.707],


    # === initial joint position ===
    'initial_joint_pos_1': [0.0100, 0.0100, 0.0700, 0.0100, 0.0100, 0.0100],
    'initial_joint_pos_2': [0.0100, 0.0100, 0.0700, 0.0100, 0.0100, 0.0100],


}

step_config = {
    'termination_reward_map' : {
        'reached_goal'          : +10,},

    'truncation_reward_map' : {
        'joint_limit'           : -5,
        'step_limit'            : -5,
        'invalid_grasping_point': -5,
        'collision_ground'      : -5,
        # 'invalid_grasping_hand' : -1,     # currently not used, gh automatically switches after the first step.
    },

    ## else
    'transition'            : -1,   # transition * max_steps < other truncation reward
    'joint_margin'          : None,

    # step_limit criteria
    'max_steps' : 5,

}

train_cfg = {
    "lr"           : 1e-4,
    "gamma"        : 0.98,
    "buffer_limit" : 100000,
    "batch_size"   : 64,
}