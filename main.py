import time

import numpy as np
import cv2
import copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from needleDetect.visualize import visualize_needle_3d_live, visualize_frame_projection
from needleDetect.ImgUtils import ImgUtils

### Main
from coppeliasim_ctrl import CoppeliaCmd
from detection import NeedleDetection
from RL import RL
####



def flatten_T(T):
    flattened = np.zeros(7)
    flattened[:3] = T[:3, -1]
    flattened[3:] = R.from_matrix(T[:3, :3]).as_quat()
    return flattened


def main():
    import sys, os
    sys.path.append("/home/surglab/icra26_nh_system/segment-anything-2-real-time")
    '''
    !! World = Cam !!
    '''

    # cmd = CoppeliaCmd()
    # n_det = NeedleDetection(cmd)
    n_det = NeedleDetection()
    rl = RL()
    vis = True

    cv2.namedWindow('img_data', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img_data', 1024, 512)


    prev_gp = None
    prev_go = None
    prev_gh = None


    # Tw_psm1 = cmd.Tw_psm1
    # Tw_psm2 = cmd.Tw_psm2
    # print(Tw_psm1)
    # print(Tw_psm2)
    '''
[[ 1.    0.    0.   -0.12]
 [ 0.    1.    0.    0.  ]
 [ 0.    0.    1.    0.14]
 [ 0.    0.    0.    1.  ]]
[[1.   0.   0.   0.12]
 [0.   1.   0.   0.  ]
 [0.   0.   1.   0.14]
 [0.   0.   0.   1.  ]]
    '''
    if vis:
        plt.ion()  # interactive mode 켜기
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    init_flag = True
    t = 0
    try:
        # initial pose setting
        # cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject('/World'))
        # cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), [0, 0, 0, 0, 0, 0, 1], cmd.sim.getObject('/World'))
        # cmd.update()
        # cmd.set_joint([0, 0, 0, 0, 0, 0, 0], which="PSM1")
        # cmd.set_joint([0, 0, 0, 0, 0, 0, 0], which="PSM2")
        # print("initialized")
        time.sleep(1)


        # while not cmd.break_flag:
        while True:
            ## Coppeliasim -- no needle detection, only see if segment-anything-2-real-time is working
            # needle_pose_w, _ = n_det.get_needle_frame()
            # needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = n_det.get_needle_frame()
            # needle_pose_w = cmd.get_needle_pose()
            # print("needle_pose\n", needle_pose_w)
            # if needle_pose_w is None:
            #     print("needle_pose_w is None")
            #     continue

            ## Physical --
            needle_detection_res = n_det.get_needle_frame()
            if needle_detection_res is None:
                print("needle_pose_w is None")
                continue

            # needle_pose_w, _ = needle_detection_res
            needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = needle_detection_res
            print("needle_pose\n", needle_pose_w)

            if vis:
                visualize_needle_3d_live(ax, needle_pose_w, start_3d, end_3d, points_3d, center_3d)
                overlay_L, overlay_R = visualize_frame_projection(n_det.image_L, n_det.image_R, needle_pose_w, n_det.P_L, n_det.P_R, points_3d, start_3d, center_3d, end_3d)
                cv2.imshow("Stereo Overlay", ImgUtils.stack_stereo_img(overlay_L, overlay_R, 0.5))

            # needle_pose_w = np.array(
            #     [[0.41470563,  0.90848837, - 0.05165388 , 0.00249347],
            #      [-0.34699006,  0.21035932 , 0.91397311 , 0.01262579],
            #      [0.84119982, - 0.36110642, 0.40247363 ,0.13303022],
            #      [0.  ,        0.       ,   0.         , 1.]])

            # needle_pose_w_flatten = flatten_T(T=needle_pose_w)  # [x, y, z, qx, qy, qz, qw]
            # if init_flag:
            #     state = np.array([-10, -10, -10])
            #     state = np.append(state, needle_pose_w_flatten)
            #     init_flag = False
            # else:
            #     # state = np.append(gp_state, needle_pose_w_flatten)    # <- 실제로는 detection한 needle pose 사용.
            #     # tmp
            #     state = np.append(gp_state, flatten_T(new_needle_pose_w))
            # print("state:", state)
            #
            # # inference
            # action = rl.infer(state)
            # new_state, reward, done, new_needle_pose_w, joint_pos_1, joint_pos_2 = rl.step(action, state, needle_pose_w)
            # print("reward:", reward)
            # if reward <= -1:
            #     print("Something's got wrong")  # exit signal
            #     break
            #
            # # cmd
            # if prev_gh is None:
            #     if int(new_state[2]) == 0:
            #         robot = 'PSM1'
            #     elif int(new_state[2]) == 1:
            #         robot = 'PSM2'
            #     else:
            #         print("no robot selected for picking up")
            #         break
            #
            #     # this robot will do joint_pos_1, joint_pos_2 in order
            #     # cmd.set_joint(q_targ=joint_pos_1, which=robot)
            #     # print("pickup")
            #     # time.sleep(1)
            #     # cmd.set_joint(q_targ=joint_pos_2, which=robot)
            #     # print("pickup done")
            #     # time.sleep(1)
            #
            # else:
            #     if gp_state[2] == 0 and new_state[2] == 1:
            #         left_to_right = True
            #         giver_robot = 'PSM1'
            #         receiver_robot = 'PSM2'
            #     elif gp_state[2] == 1 and new_state[2] == 0:
            #         left_to_right = False
            #         giver_robot = 'PSM2'
            #         receiver_robot = 'PSM1'
            #     else:
            #         print("Invalid GH occured")  # exit signal
            #
            #     # 'PSM1' will do joint_pos_1, 'PSM2' will do joint_pos_2.
            #     # if left_to_right: jp1 then jp2(jp1_approach, jp1, jp1_deproach?, jp2_approach, jp2, jp2_deproach?)
            #     # vice versa
            #     # this robot will do joint_pos_1, joint_pos_2 in order
            #     if left_to_right:
            #         pass
            #         # cmd.set_joint(q_targ=joint_pos_1, which='PSM1')
            #         # print("ready to handover")
            #         # time.sleep(1)
            #         # cmd.set_joint(q_targ=joint_pos_2, which='PSM2')
            #         # print("handover finished")
            #         # time.sleep(1)
            #     else:
            #         pass
            #         # cmd.set_joint(q_targ=joint_pos_2, which='PSM2')
            #         # print("ready to handover")
            #         # time.sleep(1)
            #         # cmd.set_joint(q_targ=joint_pos_1, which='PSM1')
            #         # print("handover finished")
            #         # time.sleep(1)
            #
            # prev_gh = state[2]
            # gp_state = new_state[:3]
            #
            # if done:
            #     print("Reached the goal")
            #     quit()

    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()