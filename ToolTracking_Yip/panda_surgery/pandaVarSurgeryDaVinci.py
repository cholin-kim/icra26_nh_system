import numpy as np
from panda.pandaVar import *

# Kinematics variables
L6 = 0.13671 + 0.44861 + 0.00889    # (Length of driving part) + (shaft proximal ~ wrist body) + (wrist body ~ pitch axis)
L7 = 0.00899    # pitch axis ~ yaw axis
dj = 0.00965    # jaw length
params = np.array([L1, L2, L3, L4, L5, L6, L7, dj, offset])


# whole kinematics: from panda base to jaw (10 DoF)
def dhparam(joints):
    # [alpha, a, d, theta]
    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = np.array(joints).T
    return np.array([[0, 0, L1, q1],
                     [-np.pi/2, 0, 0, q2],
                     [np.pi/2, 0, L2, q3],
                     [np.pi/2, offset, 0, q4],
                     [-np.pi/2, -offset, L3, q5],
                     [np.pi/2, 0, 0, q6],
                     [np.pi/2, L4, 0, q7],
                     [0, 0, L5, 0],     # to flange
                     [0, 0, L6, q8],    # gripper base to dummy
                     [np.pi/2, 0, 0, np.pi/2+q9],  # dummy to frame 8
                     [-np.pi/2, L7, 0, -np.pi/2+q10],   # pitch ~ yaw
                     [-np.pi/2, 0, dj, 0]])


# RCM Kinematics: from RCM to jaw (6 DoF)
def dhparam_RCM(joints):
    # [alpha, a, d, theta]
    qy, qp, qt, q8, q9, q10 = np.array(joints).T
    return np.array([[np.pi/2, 0, 0, np.pi/2+qy],
                     [-np.pi/2, 0, 0, -np.pi/2+qp],
                     [np.pi/2, 0, qt, 0],
                     [0, 0, 0, q8],                     # roll
                     [-np.pi/2, 0, 0, -np.pi/2+q9],     # pitch
                     [-np.pi/2, L7, 0, -np.pi/2+q10],   # yaw
                     [-np.pi/2, 0, dj, 0]])