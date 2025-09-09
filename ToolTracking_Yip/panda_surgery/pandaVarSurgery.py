import numpy as np
from panda.pandaVar import *

# Kinematics variables
L6 = 0.5184     # Length of driving part+shaft
L7 = 0.01225    # distal disk ~ jaw joint
dj = 0.01558    # jaw length
t = 0.0011     # disk thickness
h = 0.0006     # slit thickness

params = np.array([L1, L2, L3, L4, L5, L6, L7, dj, offset, t, h])


def dhparam_surgery(joints, theta_offset=0.0):
    # [alpha, a, d, theta]
    q1, q2, q3, q4, q5, q6, q7 = np.array(joints).T
    return np.array([[0, 0, L1, q1],
                     [-np.pi/2, 0, 0, q2],
                     [np.pi/2, 0, L2, q3],
                     [np.pi/2, offset, 0, q4],
                     [-np.pi/2, -offset, L3, q5],
                     [np.pi/2, 0, 0, q6],
                     [np.pi/2, L4, 0, q7],
                     [0, 0, L5, 0],

                     # The last three coordinates are attached to connect the gripper to the flange.
                     [0, 0, 0, theta_offset],   # Flange to Gripper Base (GB)
                     [0, 0, L6, 0], # Gripper Base to dummy
                     [np.pi/2, 0, 0, np.pi/2]])

def dhparam_RCM(joints):
    # [alpha, a, d, theta]
    qy, qp, qt, qr = np.array(joints).T
    return np.array([[np.pi/2, 0, 0, np.pi/2+qy],
                     [-np.pi/2, 0, 0, -np.pi/2+qp],
                     [np.pi/2, 0, qt, 0],
                     [0, 0, 0, qr],
                     [0, 0, -L6, np.pi],
                     [0, 0, L6, 0],
                     [np.pi/2, 0, 0, np.pi/2]])
