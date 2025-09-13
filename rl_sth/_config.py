import numpy as np
from scipy.spatial.transform import Rotation as R

tvec, rvec = [0.009620040250292524, 0.0001692052469176161, 0.15000193981221874], [2.00664984346439, 0.005120409365822809, 0.09097453687127555]
Tcam_w = np.identity(4)
Tcam_w[:3, -1] = tvec
Tcam_w[:3, :3] = R.from_euler('XYZ', rvec).as_matrix()

Tw_cam = np.linalg.inv(Tcam_w)



Tcam_rbBlue = np.array(
    [[0.9568239353642368, -0.18481676787500562, -0.22434508915135978, 0.16785639412189424],
     [-0.1528289575159219, 0.3366504016615289, -0.9291446694706524, -0.1834281322832605],
     [0.24724737907345332, 0.9233142852644314, 0.2938698081939357, 0.19208732533275585], [0.0, 0.0, 0.0, 1.0]]
)
Tcam_rbYellow = np.array(
    [[0.9953586220003775, 0.017297906606251145, -0.09466781943486136, -0.19689570767728143],
     [-0.09577812143769632, 0.2738245848369944, -0.9569987712597722, -0.14205479210213995],
     [0.00936830098662574, 0.9616240841232553, 0.2742104224327275, 0.12749130251648313], [0.0, 0.0, 0.0, 1.0]]
)

env_config = {
    # === Robot base transformation matrices ===
    'Tw_rb1': Tcam_rbBlue,
    'Tw_rb2': Tcam_rbYellow,
    'Tw_cam': Tw_cam,

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