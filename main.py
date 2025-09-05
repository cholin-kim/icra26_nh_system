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
    flattened[:3] = T[:3]
    flattened[3:] = R.from_matrix(T[:3, :3]).as_quat()
    return flattened


def main():
    cmd = CoppeliaCmd()
    n_det = NeedleDetection(cmd)
    rl = RL()
    vis = False

    cv2.namedWindow('img_data', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img_data', 1024, 512)


    prev_gp = None
    prev_go = None
    prev_gh = None

    
    Tw_psm1 = cmd.Tw_psm1
    Tw_psm2 = cmd.Tw_psm2
    '''
    [[-0.29619813  0.93969262  0.17101007  0.10514312]
     [-0.81379768 -0.34202014  0.46984631  0.1       ]
     [ 0.5         0.          0.8660254   0.03      ]
     [ 0.          0.          0.          1.        ]]
    [[ 2.96200738e-01  9.39692621e-01  1.71005559e-01  1.06894448e-01]
     [-8.13799228e-01  3.42020143e-01 -4.69843632e-01 -1.00000000e-01]
     [-4.99995940e-01  3.83940743e-06  8.66027748e-01  3.00000000e-02]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    '''
    if vis:
        plt.ion()  # interactive mode 켜기
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    init_flag = True
    t = 0
    try:
        while not cmd.break_flag:
            ## Coppeliasim -- no needle detection, only see if segment_anything is working
            # needle_pose_w, _ = n_det.get_needle_frame()
            needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = n_det.get_needle_frame()
            print("needle_pose\n", needle_pose_w)
            if needle_pose_w is None:
                print("needle_pose_w is None")
                continue

            ## Physical --
            # needle_pose_w, _ = n_det.get_needle_frame()
            # needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = n_det.get_needle_frame()
            # print("needle_pose\n", needle_pose_w)
            # if needle_pose_w is None:
            #     print("needle_pose_w is None")
            #     continue


            if vis:
                visualize_needle_3d_live(ax, needle_pose_w, start_3d, end_3d, points_3d, center_3d)
                overlay_L, overlay_R = visualize_frame_projection(n_det.image_L, n_det.image_R, needle_pose_w, n_det.P_L, n_det.P_R, points_3d, start_3d, center_3d, end_3d)
                cv2.imshow("Stereo Overlay", ImgUtils.stack_stereo_img(overlay_L, overlay_R, 0.5))

            needle_pose_w_flatten = flatten_T(T=needle_pose_w)  # [x, y, z, qx, qy, qz, qw]
            if init_flag:
                state = np.array([-10, -10, -10])
                state = np.append(state, needle_pose_w_flatten)
                init_flag = False
            else:
                pass

            # inference
            action = rl.infer(state)
            new_state, reward, done, new_needle_pose_w, joint_pos_1, joint_pos_2 = rl.step(action, state, needle_pose_w)

            if reward <= -1:
                print("Something's got wrong")  # exit signal

            # cmd
            if prev_gh is None:
                if state[2] == 0:
                    robot = 'PSM1'
                elif state[2] == 1:
                    robot = 'PSM2'

                # this robot will do joint_pos_1, joint_pos_2 in order
            else:
                if state[2] == 0 and action[2] == 1:
                    left_to_right = True
                    giver_robot = 'PSM1'
                    receiver_robot = 'PSM2'
                elif state[2] == 1 and action[2] == 0:
                    left_to_right = False
                    giver_robot = 'PSM2'
                    receiver_robot = 'PSM1'
                else:
                    print("Invalid GH occured")  # exit signal

                # 'PSM1' will do joint_pos_1, 'PSM2' will do joint_pos_2.
                # if left_to_right: jp1 then jp2(jp1_approach, jp1, jp1_deproach?, jp2_approach, jp2, jp2_deproach?)
                # vice versa

            prev_gh = action[2]

    #         # # change this to relative pose motion in real robot
    #         # T_ee2ee = np.eye(4)
    #         # T_ee2ee[2, -1] -= 0.01
    #         # T_w2gpoffset = T_w2gp @ T_ee2ee
    #         # cmd.set_ee_pose(T_w2gpoffset, which=robot)
    #         # last_hand = copy.copy(robot)
    #         #
    #         # # grasp
    #         # cmd.set_ee_pose(T_w2gp, which=robot)
    #         #
    #         # # change this to relative pose motion in real robot
    #         # cmd.set_ee_pose(T_w2gpoffset, which=robot)
    #         # last_hand = copy.copy(robot)
    #         #
    #         # t += 1
    #         # print(t)
    #         # if t >= 100:
    #         #     cmd.break_flag = True
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()