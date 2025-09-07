import time

import numpy as np
import cv2
import copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt






from RL import RL




def flatten_T(T):
    flattened = np.zeros(7)
    flattened[:3] = T[:3, -1]
    flattened[3:] = R.from_matrix(T[:3, :3]).as_quat()
    return flattened

from rl_sth.Kinematics.dvrkKinematics import dvrkKinematics
def get_waypoint(joint_pos, distance=0.03):
    Trb_targ = dvrkKinematics.fk(joint_pos)[0][-1]
    Ttarg_targ = np.identity(4)
    Ttarg_targ[2, -1] = distance
    return dvrkKinematics.ik(Trb_targ @ Ttarg_targ)




def main_coppelia():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from coppeliasim_ctrl import CoppeliaCmd
    import sys, os
    sys.path.append("/home/surglab/icra26_nh_system/segment-anything-2-real-time")
    '''
    !! World = Cam !!
    '''

    cmd = CoppeliaCmd()
    rl = RL()
    vis = False

    cv2.namedWindow('img_data', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img_data', 1024, 512)


    prev_gp = None
    prev_go = None
    prev_gh = None


    Tw_psm1 = cmd.Tw_psm1
    Tw_psm2 = cmd.Tw_psm2
    # print(Tw_psm1)
    # print(Tw_psm2)
    # print(cmd.get_needle_pose())
    '''
[[ 9.99999982e-01 -1.45821419e-04 -1.20288867e-04 -1.17465392e-01]
 [-1.60143467e-06 -6.42851847e-01  7.65990537e-01  6.73936401e-02]
 [-1.89025748e-04 -7.65990523e-01 -6.42851836e-01  1.82525761e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 9.99999982e-01 -1.45821419e-04 -1.20288867e-04  1.22534604e-01]
 [-1.60143467e-06 -6.42851847e-01  7.65990537e-01  6.73932558e-02]
 [-1.89025748e-04 -7.65990523e-01 -6.42851836e-01  1.82480395e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 9.99999982e-01 -1.45821419e-04 -1.20288867e-04 -2.49431320e-02]
 [-1.60143467e-06 -6.42851847e-01  7.65990537e-01 -8.55640182e-03]
 [-1.89025748e-04 -7.65990523e-01 -6.42851836e-01  3.00922960e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    '''


    fake_grasp = True
    init_flag = True
    try:
        # initial pose setting
        cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject('/World'))
        random_pose = np.array([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])
        random_pose[:2] += np.array([np.random.uniform(-1, 1) * 0.1, np.random.uniform(-1, 1) * 0.05])
        cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), random_pose.tolist(), -1)
        cmd.update()
        cmd.set_joint([0, 0, 0.1, 0, 0, 0, 0], which="PSM1")
        cmd.set_joint([0, 0, 0.1, 0, 0, 0, 0], which="PSM2")
        print("initialized")
        time.sleep(1)

        while not cmd.break_flag:
            ### Coppeliasim -- no needle detection, only see if segment-anything-2-real-time is working
            needle_pose_w = cmd.get_needle_pose()
            print("needle_pose\n", needle_pose_w)
            if needle_pose_w is None:
                print("needle_pose_w is None")
                continue
            #####################################################

            needle_pose_w_flatten = flatten_T(T=needle_pose_w)  # [x, y, z, qx, qy, qz, qw]
            if init_flag:
                state = np.array([-10, -10, -10])
                state = np.append(state, needle_pose_w_flatten)
                init_flag = False
            else:
                state = np.append(gp_state, needle_pose_w_flatten)    # <- 실제로는 detection한 needle pose 사용.
            print("state:", state)

            # inference
            action = rl.infer(state)
            new_state, reward, done, new_needle_pose_w, joint_pos_1, joint_pos_2 = rl.step(action, state, needle_pose_w)
            print("reward:", reward)
            if reward <= -1:
                print("Reward less than -1")  # exit signal
                # 일단 계속.
            #     break

            # cmd
            if prev_gh is None:
                if int(new_state[2]) == 0:
                    robot = 'PSM1'
                elif int(new_state[2]) == 1:
                    robot = 'PSM2'
                else:
                    print("no robot selected for picking up")
                    break

                # cmd
                cmd.open_jaw(robot)
                time.sleep(0.5)
                joint_pos_1_prev = get_waypoint(joint_pos_1)
                cmd.set_joint_rel(q_targ=joint_pos_1_prev, which=robot, jaw='OPEN')
                print("pickup_prev")
                time.sleep(0.5)
                cmd.set_joint_rel(q_targ=joint_pos_1, which=robot, jaw='OPEN')
                time.sleep(0.5)
                cmd.close_jaw(robot)
                time.sleep(0.5)
                print("pickup")

                if fake_grasp:
                    parent = '/PSM1_ee' if robot == 'PSM1' else '/PSM2_ee'
                    cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject(parent), True)
                else:
                    cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(), cmd.sim.getObject(cmd.world))
                    cmd.update()
                cmd.set_joint_rel(q_targ=joint_pos_2, which=robot, jaw='STILL')
                print("pickup done")
                time.sleep(1)

            else:
                if gp_state[2] == 0 and new_state[2] == 1:
                    left_to_right = True
                    giver_robot = 'PSM1'
                    receiver_robot = 'PSM2'
                elif gp_state[2] == 1 and new_state[2] == 0:
                    left_to_right = False
                    giver_robot = 'PSM2'
                    receiver_robot = 'PSM1'
                else:
                    print("Invalid GH occured")  # exit signal
                    break

                # cmd
                joint_pos_first = joint_pos_1 if left_to_right else joint_pos_2
                joint_pos_second = joint_pos_2 if left_to_right else joint_pos_1
                joint_pos_second_prev = get_waypoint(joint_pos_second)
                joint_pos_first_after = get_waypoint(joint_pos_first)
                cmd.set_joint_rel(q_targ=joint_pos_first, which=giver_robot, jaw='STILL')
                print("ready to handover")
                time.sleep(1)
                if fake_grasp:
                    cmd.open_jaw(which=receiver_robot)
                    cmd.set_joint_rel(q_targ=joint_pos_second_prev, which=receiver_robot, jaw='OPEN')
                    time.sleep(0.5)
                    cmd.set_joint_rel(q_targ=joint_pos_second, which=receiver_robot, jaw='OPEN')
                    time.sleep(0.5)
                    cmd.close_jaw(which=receiver_robot)
                    time.sleep(0.5)
                    parent = '/PSM2_ee' if receiver_robot == 'PSM2' else '/PSM1_ee'
                    cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject(parent), True)
                else:
                    cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(), cmd.sim.getObject(cmd.world))
                    cmd.update()
                    cmd.set_joint_rel(q_targ=joint_pos_second_prev, which=receiver_robot)
                    time.sleep(0.5)
                    cmd.set_joint_rel(q_targ=joint_pos_second, which=receiver_robot)

                cmd.open_jaw(which=giver_robot)
                time.sleep(0.5)
                cmd.set_joint_rel(joint_pos_first_after, which=giver_robot, jaw='OPEN')
                print("handover finished")
                time.sleep(3)

            prev_gh = state[2]
            gp_state = new_state[:3]

            if done:
                print("Reached the goal")
                quit()

    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()

def main():
    from detection import NeedleDetection
    from needleDetect.visualize import visualize_needle_3d_live, visualize_frame_projection
    from needleDetect.ImgUtils import ImgUtils
    import sys, os
    sys.path.append("/home/surglab/icra26_nh_system/segment-anything-2-real-time")
    '''
    !! World = Cam !!
    '''

    # cmd = CoppeliaCmd()
    n_det = NeedleDetection()
    rl = RL()
    vis = False

    cv2.namedWindow('img_data', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img_data', 1024, 512)


    prev_gp = None
    prev_go = None
    prev_gh = None


    # Tw_psm1 = cmd.Tw_psm1
    # Tw_psm2 = cmd.Tw_psm2
    # print(Tw_psm1)
    # print(Tw_psm2)
    # print(cmd.get_needle_pose())
    '''
[[ 9.99999982e-01 -1.45821419e-04 -1.20288867e-04 -1.17465392e-01]
 [-1.60143467e-06 -6.42851847e-01  7.65990537e-01  6.73936401e-02]
 [-1.89025748e-04 -7.65990523e-01 -6.42851836e-01  1.82525761e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 9.99999982e-01 -1.45821419e-04 -1.20288867e-04  1.22534604e-01]
 [-1.60143467e-06 -6.42851847e-01  7.65990537e-01  6.73932558e-02]
 [-1.89025748e-04 -7.65990523e-01 -6.42851836e-01  1.82480395e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 9.99999982e-01 -1.45821419e-04 -1.20288867e-04 -2.49431320e-02]
 [-1.60143467e-06 -6.42851847e-01  7.65990537e-01 -8.55640182e-03]
 [-1.89025748e-04 -7.65990523e-01 -6.42851836e-01  3.00922960e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    '''

    if vis:
        plt.ion()  # interactive mode 켜기
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    init_flag = True

    try:
        # initial pose setting
        cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject('/World'))
        random_pose = np.array([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])
        random_pose[:2] += np.array([np.random.uniform(-1, 1) * 0.1, np.random.uniform(0, 1) * 0.15])
        cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), random_pose.tolist(), -1)
        cmd.update()
        cmd.set_joint([0, 0, 0.1, 0, 0, 0, 0], which="PSM1")
        cmd.set_joint([0, 0, 0.1, 0, 0, 0, 0], which="PSM2")
        print("initialized")
        time.sleep(1)

        while not cmd.break_flag:
        # while True:
            ### Coppeliasim -- no needle detection, only see if segment-anything-2-real-time is working
            needle_pose_w = cmd.get_needle_pose()
            print("needle_pose\n", needle_pose_w)
            if needle_pose_w is None:
                print("needle_pose_w is None")
                continue
            #####################################################


            ## Physical #########################################
            # needle_detection_res = n_det.get_needle_frame()
            # if needle_detection_res is None:
            #     print("needle_pose_w is None")
            #     continue

            # needle_pose_w, _ = needle_detection_res
            # needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = needle_detection_res
            # print("needle_pose\n", needle_pose_w)

            # if vis:
            #     visualize_needle_3d_live(ax, needle_pose_w, start_3d, end_3d, points_3d, center_3d)
            #     overlay_L, overlay_R = visualize_frame_projection(n_det.image_L, n_det.image_R, needle_pose_w, n_det.P_L, n_det.P_R, points_3d, start_3d, center_3d, end_3d)
            #     cv2.imshow("Stereo Overlay", ImgUtils.stack_stereo_img(overlay_L, overlay_R, 0.5))
            #####################################################


            needle_pose_w_flatten = flatten_T(T=needle_pose_w)  # [x, y, z, qx, qy, qz, qw]
            if init_flag:
                state = np.array([-10, -10, -10])
                state = np.append(state, needle_pose_w_flatten)
                init_flag = False
            else:
                # state = np.append(gp_state, needle_pose_w_flatten)    # <- 실제로는 detection한 needle pose 사용.
                # tmp
                state = np.append(gp_state, flatten_T(new_needle_pose_w))
            print("state:", state)

            # inference
            action = rl.infer(state)
            new_state, reward, done, new_needle_pose_w, joint_pos_1, joint_pos_2 = rl.step(action, state, needle_pose_w)
            print("reward:", reward)
            if reward <= -1:
                print("Reward less than -1")  # exit signal
                # 일단 계속.
            #     break

            # cmd
            if prev_gh is None:
                if int(new_state[2]) == 0:
                    robot = 'PSM1'
                elif int(new_state[2]) == 1:
                    robot = 'PSM2'
                else:
                    print("no robot selected for picking up")
                    break

                # cmd
                # this robot will do joint_pos_1, joint_pos_2 in order
                # cmd.set_joint(q_targ=joint_pos_1, which=robot)
                cmd.set_joint_rel(q_targ=joint_pos_1, which=robot)

                print("pickup")
                time.sleep(1)


                parent = '/PSM1_ee' if robot == 'PSM1' else '/PSM2_ee'
                cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject(parent), True)
                # cmd.set_joint(q_targ=joint_pos_2, which=robot)
                cmd.set_joint_rel(q_targ=joint_pos_2, which=robot)

                # cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(), cmd.sim.getObject(cmd.world))
                # cmd.update()
                print("pickup done")
                time.sleep(1)

            else:
                if gp_state[2] == 0 and new_state[2] == 1:
                    left_to_right = True
                    giver_robot = 'PSM1'
                    receiver_robot = 'PSM2'
                elif gp_state[2] == 1 and new_state[2] == 0:
                    left_to_right = False
                    giver_robot = 'PSM2'
                    receiver_robot = 'PSM1'
                else:
                    print("Invalid GH occured")  # exit signal

                # cmd
                if left_to_right:
                    # cmd.set_joint(q_targ=joint_pos_1, which='PSM1')
                    cmd.set_joint_rel(q_targ=joint_pos_1, which='PSM1')
                    # cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(), cmd.sim.getObject(cmd.world))
                    # cmd.update()
                    print("ready to handover")
                    time.sleep(1)
                    # cmd.set_joint(q_targ=joint_pos_2, which='PSM2')
                    cmd.set_joint_rel(q_targ=joint_pos_2, which='PSM2')


                    parent = '/PSM2_ee'
                    cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject(parent), True)
                    print("handover finished")
                    time.sleep(5)
                else:
                    # cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(), cmd.sim.getObject(cmd.world))
                    # cmd.update()
                    # cmd.set_joint(q_targ=joint_pos_2, which='PSM2')
                    cmd.set_joint_rel(q_targ=joint_pos_2, which='PSM2')

                    print("ready to handover")
                    time.sleep(2)
                    # cmd.set_joint(q_targ=joint_pos_1, which='PSM1')
                    cmd.set_joint_rel(q_targ=joint_pos_1, which='PSM1')


                    parent = '/PSM1_ee'
                    cmd.sim.setObjectParent(cmd.sim.getObject(cmd.needle), cmd.sim.getObject(parent), True)
                    print("handover finished")
                    time.sleep(5)

            prev_gh = state[2]
            gp_state = new_state[:3]

            if done:
                print("Reached the goal")
                quit()

    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    # main()
    main_coppelia()
#     aa = np.array(
# [[ 9.99999982e-01, -1.45821419e-04, -1.20288867e-04, -2.49431320e-02],
#  [-1.60143467e-06, -6.42851847e-01,  7.65990537e-01, -8.55640182e-03],
#  [-1.89025748e-04, -7.65990523e-01, -6.42851836e-01,  3.00922960e-01],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
#     print(flatten_T(aa))

    # pose_w2cam = np.array([-0.0025006371292415957, 0.18311638869233257, 0.9556953390164797, -0.9063255023394088, 4.0664985549832e-05, 8.5321061260836e-05, -0.42258025850232767])
    # Tw_cam = np.identity(4)
    # Tw_cam[:3, -1] = pose_w2cam[:3]
    # Tw_cam[:3, :3] = R.from_quat(pose_w2cam[3:]).as_matrix()
    # Tw_ho_pos = np.identity(4)
    # Tw_ho_pos[:3, -1] = [0.0, 0.1, 0.05]
    # Tcam_ho_pos = np.linalg.inv(Tw_cam) @ Tw_ho_pos
    # print(Tcam_ho_pos)
    # Tcam_ho_pos = np.array(
    # [[ 9.99999982e-01, -1.60143467e-06, -1.89025748e-04, -2.51024856e-03],
    #  [-1.45821419e-04, -6.42851847e-01, -7.65990523e-01, 8.05316778e-02],
    #  [-1.20288867e-04,  7.65990537e-01, -6.42851836e-01,  2.50151801e-01],
    #  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

