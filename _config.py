import numpy as np
from scipy.spatial.transform import Rotation as R

Tcam_rbBlue = np.array(
    [[0.958227689670898, -0.21768249007965917, -0.18551018371154965, 0.17551387734934729],
     [-0.09430895426216324, 0.37185710016414775, -0.9234869345061079, -0.17092286002679188],
     [0.27001021442521606, 0.9024060231238654, 0.33579436197741447, 0.19878618409315504], [0.0, 0.0, 0.0, 1.0]]

)
Tcam_rbYellow = np.array(
    [[0.9977628695816074, -0.04424886682120791, -0.05011281142896951, -0.1929004230309927],
     [-0.03362939432434292, 0.3156440822984118, -0.9482815389679223, -0.16374099581128818],
     [0.057778195901692066, 0.9478453729881434, 0.31344988272978047, 0.14549715406600355], [0.0, 0.0, 0.0, 1.0]]
)

Tcam_w = np.identity(4)


rvec = [2.000704471390556, -0.004855342843570476, -2.9530378153928445]
Tcam_w[:3, -1] = [0.005541766210923305, -0.0103154460209131, 0.14533409657221247]
Tcam_w[:3, :3] = R.from_euler('XYZ', rvec).as_matrix()
Tw_cam = np.linalg.inv(Tcam_w)

# Tcam_no = np.array(
#     [[-0.0127571, - 0.74122115 ,- 0.67113968,  0.00899206],
#      [0.93273507,  0.23305442 ,- 0.27511986, - 0.0141315],
#     [0.36033672 ,- 0.62950524,
# 0.68838986,
# 0.11794758],
# [0.   ,       0.   ,       0.    ,      1.]]
# )



# Tw_rbBlue = Tw_cam @ Tcam_rbBlue
# Tw_rbYellow = Tw_cam @ Tcam_rbYellow
# print(Tw_rbBlue.tolist())
# print(Tw_rbYellow.tolist())
# TrbBlue_1 = np.array(
#     [[0.7159998682549201, 0.03545111850223421, 0.6971996893687471, -0.17691619987250565],
#      [0.005898802978514631, 0.9983668592607112, -0.056822693119244726, 0.019429265763257125],
#      [-0.698075492180006, 0.04479768439153683, 0.714621420537335, -0.18380646793838934], [0.0, 0.0, 0.0, 1.0]]
# )
# TrbYellow_2 = np.array(
#     [[0.6978269278962297, -0.3434517163585013, -0.6285527004423017, 0.11858644113485635],
#      [0.010119736768074369, 0.882180308755619, -0.47080303075870844, 0.08767494026844173],
#      [0.7161949243262307, 0.3221782447252452, 0.6190848156715724, -0.11583910207904086], [0.0, 0.0, 0.0, 1.0]]
#
# )
# Tcam_1 = Tcam_rbBlue @ TrbBlue_1
# Tcam_2 = Tcam_rbYellow @ TrbYellow_2
# print(Tcam_1)
# print(Tcam_2)
# [[ 0.81430695 -0.19166717  0.5478758   0.03585644]
#  [ 0.57933191  0.32653637 -0.74682564  0.0227297 ]
#  [-0.03575942  0.92554724  0.37693974  0.10682891]
#  [ 0.          0.          0.          1.        ]]
# [[ 0.65992747 -0.39786411 -0.63733813 -0.07265377]
#  [-0.69942769 -0.01551061 -0.71453505 -0.03020683]
#  [ 0.27440234  0.91731324 -0.28851305  0.1991414 ]
#  [ 0.          0.          0.          1.        ]]
# print(Tw_cam @ Tcam_1)
# print(Tw_cam @ Tcam_2)
# [[ 0.78836456 -0.12367043  0.60264994  0.05120967]
#  [-0.35007097  0.7153529   0.60474833 -0.05360517]
#  [-0.50589687 -0.6877324   0.52066545 -0.00209251]
#  [ 0.          0.          0.          1.        ]]
# [[ 0.70397264 -0.3181697  -0.63497288 -0.04726352]
#  [ 0.4820741   0.87060767  0.09821836  0.06184964]
#  [ 0.52156215 -0.37524702  0.76626535  0.00636637]
#  [ 0.          0.          0.          1.        ]]
# ho_pos_w(cam x)
# TrbYellow_ee = np.array(
# [[-0.9894671754545006, -0.1379823834803001, -0.043767231431609356, 0.1901136701944518], [0.13771416388482538, -0.8041051141615475, -0.5783163273198746, 0.002624582235816913], [0.04460401062232071, -0.5782523905945745, 0.8146377446497699, -0.1702376479073105], [0.0, 0.0, 0.0, 1.0]]
# )
# Tcam_ee = Tcam_rbYellow @ TrbYellow_ee
# print(Tcam_ee)
# print(Tw_cam @ Tcam_ee)
# quit()


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
    'num_ho_ori_plane': 4,  # num_ho_ori in one plane
    'num_ho_plane': 2,
    'num_ho_ori': 8,  # num_ho_ori_plane * num_ho_plane

    # === Handover Position(fixed) ===
    'ho_pos': [0.0, 0.02, 0.02],

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
    'termination_reward_map': {
        'reached_goal': +10, },

    'truncation_reward_map': {
        'joint_limit': -5,
        'step_limit': -5,
        'invalid_grasping_point': -5,
        'collision_ground': -5,
        # 'invalid_grasping_hand' : -1,     # currently not used, gh automatically switches after the first step.
    },

    ## else
    'transition': -1,  # transition * max_steps < other truncation reward
    'joint_margin': None,

    # step_limit criteria
    'max_steps': 5,

}

train_cfg = {
    "lr": 1e-4,
    "gamma": 0.98,
    "buffer_limit": 100000,
    "batch_size": 64,
}
