import numpy as np

# Kinematics variables



# dvrk variables
L1 = 0.4318  # Rcc (m)
L2 = 0.4162  # tool
L3 = 0.0091  # pitch ~ yaw (m)
L4 = 0.0095  # yaw ~ tip (m)


# modified DH
def dhparam(joints):
    # [a, alpha, d, theta]
    q1, q2, q3, q4, q5, q6 = np.array(joints).T
    return np.array([[0,            np.pi/2,     0,             np.pi/2 + q1],  # Arm Yaw
                     [0,            -np.pi/2,    0,             -np.pi/2 + q2], # Arm Pitch
                     [0,            np.pi/2,     q3-L1,      0],             # Arm Insertion
                     [0,            0,           L2,        q4],            # Tool Roll
                     [0,            -np.pi/2,    0,             -np.pi/2 + q5], # Tool Pitch
                     [L3,  -np.pi/2,    0,             -np.pi/2 + q6], # Tool Yaw
                     [0,            -np.pi/2,    L4,  0],            # End Effector
                     [0,            np.pi,       0,             np.pi]])        # Match to Base




# modified DH
def dhparam_sym_flipped():
    # flipped version to check professor's Pieper solution appraoch.
    import sympy as sp
    # [a, alpha, d, theta]
    q1, q2, q3, q4, q5, q6 = sp.symbols("q1, q2, q3, q4, q5, q6")

    return np.array([[0,            0,     L4,             0],
                     [0,            -sp.pi/2,    0,             -sp.pi/2 + q6],
                     [L3,            -sp.pi/2,     0,      -sp.pi/2 + q5],
                     [0,            -sp.pi/2,           q3-L1+L2,        0],
                     [0,            0,    0,             q4],
                     [0,  -sp.pi/2,    0,             q2+sp.pi/2],
                     [0,            sp.pi/2,    0,  q1-sp.pi/2],
                     [0,            -sp.pi/2,       0,             sp.pi]])



# Joint Limit

# based on https://research.intusurg.com/images/2/26/User_Guide_DVRK_Jan_2019.pdf
# 안맞는 것 같음.
# joint_range_upper_limit = np.array([1.5994, 0.94249, 0.24001, 3.0485, 3.0528, 3.0376, 3.0399])  # rad, m(q3)
# joint_range_lower_limit = np.array([-1.605, -0.93556, -0.002444, -3.0456, -3.0414, -3.0481, -3.0498])

# based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8205948
# 이것도 아닌 것 같음.
# joint_range_lower_limit = np.array([-np.deg2rad(60), -np.deg2rad(45), 0.05, -np.deg2rad(180), -np.deg2rad(90), -np.deg2rad(90)])  # rad, m(q3)
# joint_range_upper_limit = np.array([np.deg2rad(60),-np.deg2rad(45), 0.18, np.deg2rad(180), np.deg2rad(90), np.deg2rad(90)])  # rad, m(q3)

# based on SurRoL: https://github.com/hding2455/SurRoL-v2/blob/SurRoL-v2/surrol/robots/psm.py
joint_range_lower_limit = np.deg2rad([-91, -53, 0, -260, -80, -80])  # rad, m(q3)
joint_range_upper_limit = np.deg2rad([91, 53, 240, 260, 80, 80])
joint_range_upper_limit[2] = 0.24
# [-1.58824962, -0.9250245, 0.  , -3.53785606, -1.3962634, -1.3962634, -0.3490659]
# [ 1.58824962,  0.9250245, 0.24,  3.53785606,  1.3962634,  1.3962634,  1.3962634]


# vel_ratio = 0.1
# acc_ratio = 0.1
# v_max = np.array([np.pi, np.pi, 0.2, 3*2*np.pi, 3*2*np.pi, 3*2*np.pi])*vel_ratio       # max velocity (rad/s) or (m/s)
# a_max = np.array([np.pi, np.pi, 0.2, 2*2*np.pi, 2*2*np.pi, 2*2*np.pi])*acc_ratio       # max acceleration (rad/s^2) or (m/s^2)
#
#
# a5 = 0.0091
# g1 = 0.005     # gripper offset
# p_offset = 0.01759
#
# joint_min = np.array([-np.pi/2, -np.pi / 3, 0, -np.pi * 3/2, -np.pi * 2/3, -np  .pi *  2/3])
# joint_max = np.array([np.pi/2, np.pi /3, 0.50, np.pi * 3/2, np.pi * 2/3, np.pi * 2/3])



# Cartesian space limits


