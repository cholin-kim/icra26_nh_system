import numpy as np

# Kinematics variables
L1 = 0.3330  # base ~ shoulder (m)
L2 = 0.3160  # shoulder ~ elbow (m)
L3 = 0.3840  # elbow ~ wrist (m)
L4 = 0.0880  # wrist offset (m)
L5 = 0.1070 # flange offset (m)
Lh = 0.1034    # flange_franka_hand_tcp
offset = 0.0825 # elbow offset
L6 = 0.5184     # Length of driving part+shaft
L7 = 0.01225    # distal disk ~ jaw joint
# L7 = 0.0    # distal disk ~ jaw joint
dj = 0.01558    # jaw length
t = 0.0011     # disk thickness
h = 0.0006     # slit thickness

params = np.array([L1, L2, L3, L4, L5, L6, L7, dj, offset, t, h])

# def dhparam(joints):
#     """
#     joints = (N_conf x 7)
#     dh_params = (N_conf x 9 x 4)
#     """
#     params = np.array([[0, 0, L1, 0],
#                        [-np.pi/2, 0, 0, 0],
#                        [np.pi/2, 0, L2, 0],
#                        [np.pi/2, offset, 0, 0],
#                        [-np.pi/2, -offset, L3, 0],
#                        [np.pi/2, 0, 0, 0],
#                        [np.pi/2, L4, 0, 0],
#                        [0, 0, L5, 0],
#                        [0, 0, Lg+Lt, -np.pi/4]])
#     dh_params = np.tile(params, (len(joints), 1, 1))
#     dh_params[:, :7, 3] = joints
#     return dh_params

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
                     [0, 0, L6, theta_offset],
                     [-np.pi / 2, 0, 0, -np.pi / 2]])

def dhparam(joints):
    # [alpha, a, d, theta]
    q1, q2, q3, q4, q5, q6, q7 = np.array(joints).T
    return np.array([[0, 0, L1, q1],
                     [-np.pi/2, 0, 0, q2],
                     [np.pi/2, 0, L2, q3],
                     [np.pi/2, offset, 0, q4],
                     [-np.pi/2, -offset, L3, q5],
                     [np.pi/2, 0, 0, q6],
                     [np.pi/2, L4, 0, q7],
                     [0, 0, L5, 0]])

def dhparam_RCM(joints, theta_offset=0.0):
    # [alpha, a, d, theta]
    qy, qp, qt, qr = np.array(joints).T
    return np.array([[np.pi/2, 0, 0, np.pi/2+qy],
                     [-np.pi/2, 0, 0, -np.pi/2+qp],
                     [np.pi/2, 0, qt, 0],
                     [0, 0, 0, qr],
                     [0, 0, -L6, np.pi],
                     [0, 0, L6, theta_offset],
                     [-np.pi / 2, 0, 0, -np.pi / 2]])

# Cartesian space limits
v_max = 1.7000  # trans vel (m/s)
a_max = 13.0000 # trans acc (m/s^2)
j_max = 6500.0000   # trans jerk (m/s^3)
v_max_rot = 2.5000  # rot vel (rad/s)
a_max_rot = 25.0000 # rot acc (rad/s^2)
j_max_rot = 12500.0000  # rot jerk (rad/s^3)
v_max_elbow = 2.1750    # max. vel of elbow (rad/s)
a_max_elbow = 10.0000   # max. acc of elbow (rad/s^2)
j_max_elbow = 5000.0000 # max. jerk of elbow (rad/s^3)

# Joint space limits
q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])    # joint limit (rad)
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])    # joint limit (rad)
qv_ratio = 0.8   # safety margin
qa_ratio = 0.8
qj_ratio = 0.8
qv_max = np.array([ 2.1750, 2.1750,  2.1750,  2.1720,  2.6100,  2.6100,  2.6100])*qv_ratio    # max. ang. vel (rad/s)
qa_max = np.array([15.0000, 7.5000, 10.0000, 12.5000, 15.0000, 20.0000, 20.0000])*qa_ratio    # max. ang. acc (rad/s^2)
qj_max = np.array([   7500,   3750,    5000,    6250,    7500,   10000,   10000])*qj_ratio    # max. ang. jerk (rad/s^3)
tau_max = np.array([87, 87, 87, 87, 12, 12, 12])    # max torque (Nm)
ror_max = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000])  # max roratum (Nm/s)


import os
file_mass = os.path.dirname(__file__) + '/link_properties/mass.npy'
file_com = os.path.dirname(__file__) + '/link_properties/p_com.npy'
file_inertia = os.path.dirname(__file__) + '/link_properties/Is.npy'
if os.path.isfile(file_mass) & os.path.isfile(file_com) & os.path.isfile(file_inertia):
    mass = np.load(file_mass)
    p_com = np.load(file_com)
    Is = np.load(file_inertia)
# else:
#     import rospy
#     from gazebo_msgs.srv import GetLinkProperties
#     # Request service
#     name_service = '/gazebo/get_link_properties'
#     rospy.wait_for_service(name_service)
#     service_handle = rospy.ServiceProxy(name_service, GetLinkProperties)
#
#     # Parsing from the result
#     mass = []
#     p_com = []
#     Is = []
#     for i in range(8):
#         result = service_handle(link_name='panda_link'+str(i))
#         mass.append(result.mass)
#         p_com.append([result.com.position.x, result.com.position.y, result.com.position.z])
#         Is.append([[result.ixx, result.ixy, result.ixz],
#                    [result.ixy, result.iyy, result.iyz],
#                    [result.ixz, result.iyz, result.izz]])
#     mass = np.array(mass)
#     p_com = np.array(p_com)
#     Is = np.array(Is)
#
#     print("mass=", mass)
#     print("p_com=", p_com)
#     print("Is=", Is)
#     np.save('/home/minho/pycharmprojects/panda/link_properties/mass.npy', mass)
#     np.save('/home/minho/pycharmprojects/panda/link_properties/p_com.npy', p_com)
#     np.save('/home/minho/pycharmprojects/panda/link_properties/Is.npy', Is)