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


from Kinematics.dvrkKinematics import dvrkKinematics


def get_waypoint(joint_pos, distance=0.03):
    Trb_targ = dvrkKinematics.fk(joint_pos)[0][-1]
    Ttarg_targ = np.identity(4)
    Ttarg_targ[2, -1] = distance
    return dvrkKinematics.ik(Trb_targ @ Ttarg_targ)

def get_waypoint_T(Trb_targ, distance=0.02):
    Ttarg_targ = np.identity(4)
    z_axis = Trb_targ[:3, 2]
    Trb_targ[:3, 3]
    pos = Trb_targ[:3, 3]
    new_pos = pos + z_axis * distance

    T_new = Trb_targ.copy()
    T_new[:3, 3] = new_pos
    return T_new


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
                state = np.append(gp_state, needle_pose_w_flatten)  # <- 실제로는 detection한 needle pose 사용.
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
                    cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(),
                                          cmd.sim.getObject(cmd.world))
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
                    cmd.sim.setObjectPose(cmd.sim.getObject(cmd.needle), flatten_T(new_needle_pose_w).tolist(),
                                          cmd.sim.getObject(cmd.world))
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


def get_key_input():
    """
    이미지(img)를 띄우면서 화살표키 / WASD / q 입력 대기
    q: 종료
    반환: 'UP', 'DOWN', 'LEFT', 'RIGHT', 'QUIT' 또는 None
    """
    img = np.zeros((1, 1))
    cv2.imshow("get_key_input", img)
    print("WASDFB, q")

    while True:
        key = cv2.waitKey(0) & 0xFF  # 0 → 입력까지 대기
        if key in (82, ord('w'), ord('W')):
            return 'UP'
        elif key in (84, ord('s'), ord('S')):
            return 'DOWN'
        elif key in (81, ord('a'), ord('A')):
            return 'LEFT'
        elif key in (83, ord('d'), ord('D')):
            return 'RIGHT'
        elif key in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            return 'QUIT'
        elif key in (ord('f'), ord('F')):
            return 'FORWARD'
        elif key in (ord('b'), ord('B')):
            return 'BACKWARD'
        else:
            print(f"Unknown key: {key}")

def main():
    from detection import NeedleDetection
    from needleDetect.visualize import visualize_needle_3d_live, visualize_frame_projection
    from needleDetect.ImgUtils import ImgUtils
    from dvrk_ctrl import dvrkCmd
    import sys, os
    sys.path.append("/home/surglab/icra26_nh_system/segment-anything-2-real-time")
    from DQN_cam._config import Tcam_rbBlue, Tcam_rbYellow
    '''
    !! World = Cam !!
    '''

    cmd = dvrkCmd()

    cmd.open_jaw(which='PSM1')
    cmd.open_jaw(which='PSM2')
    cmd.close_jaw(which='PSM1')
    cmd.close_jaw(which='PSM2')
    cmd.set_joint_positions_rel(q_targ=[-1.0, -0.35, 0.15, 0.0, 0.0, 0.0], which='PSM2')
    cmd.set_joint_positions_rel(q_targ=[1.0, -0.35, 0.15, 0.0, 0.0, 0.0], which='PSM1')
    # quit()


    # n_det = NeedleDetection()
    rl = RL()
    vis = False

    # cv2.namedWindow('img_data', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('img_data', 1024, 512)

    prev_gp = None
    prev_go = None
    prev_gh = None

    if vis:
        plt.ion()  # interactive mode 켜기
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    init_flag = True


    ############
    # === 로그 설정 ===
    import os
    import time
    import threading
    import pickle

    # === 로그 폴더 이름을 숫자 카운트 방식으로 ===
    LOG_ROOT = "./logs"
    os.makedirs(LOG_ROOT, exist_ok=True)

    # logs 폴더 안의 숫자 폴더 리스트
    existing_dirs = [d for d in os.listdir(LOG_ROOT) if d.isdigit()]
    if existing_dirs:
        run_id = str(max(map(int, existing_dirs)) + 1)
    else:
        run_id = "0"

    LOG_DIR = os.path.join(LOG_ROOT, run_id)
    IMG_DIR = os.path.join(LOG_DIR, "images")
    os.makedirs(IMG_DIR, exist_ok=True)

    LOG_PATH_PICKLE = os.path.join(LOG_DIR, "experiment_log.pkl")


    BACKUP_EVERY = 200  # n 프레임마다 백업
    LOG_HZ = 20  # 로깅 주기 (Hz)
    dt = 1.0 / LOG_HZ
    SAVE_IMAGES = True  # 이미지 파일 저장 여부 (용량 큼)
    IMAGE_EVERY = 1.0  # 이미지 저장 주기(초)


    all_data = []  # 프레임 로그 누적
    frame_count = 0

    # 메인스레드 <-> 로깅스레드 공유 변수
    shared = {
        "state": None,
        "action": None,
        "stop": False
    }
    last_img_save_t = 0.0

    def backup():
        # 안전한 백업
        try:
            with open(LOG_PATH_PICKLE, "wb") as f:
                pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"[LOG] backup: {len(all_data)} frames -> {LOG_PATH_PICKLE}")
        except Exception as e:
            print(f"[LOG] backup failed: {e}")

    def logger_loop(n_det_local):
        nonlocal frame_count, last_img_save_t
        while not shared["stop"]:
            ts = time.time()
            try:
                blue_joint = cmd.get_joint_positions(which='PSM2')
                yellow_joint = cmd.get_joint_positions(which='PSM1')
                blue_jaw = cmd.psm2.jaw.measured_js()[0]
                yellow_jaw = cmd.psm1.jaw.measured_js()[0]

                # 최신 이미지 갱신
                imgL = n_det_local.get_image(which='L')
                imgR = n_det_local.get_image(which='R')

                imgL_path = imgR_path = None
                if SAVE_IMAGES and (ts - last_img_save_t) >= IMAGE_EVERY:
                    if imgL is not None:
                        imgL_path = os.path.join(IMG_DIR, f"L_{int(ts * 1000)}.jpg")
                        cv2.imwrite(imgL_path, imgL)
                    if imgR is not None:
                        imgR_path = os.path.join(IMG_DIR, f"R_{int(ts * 1000)}.jpg")
                        cv2.imwrite(imgR_path, imgR)
                    last_img_save_t = ts

                frame_data = {
                    "timestamp": ts,
                    "blue_joint": blue_joint,
                    "yellow_joint": yellow_joint,
                    "blue_jaw": blue_jaw,
                    "yellow_jaw": yellow_jaw,
                    "state": shared["state"],
                    "action": shared["action"],
                    "img_L_path": imgL_path,
                    "img_R_path": imgR_path,
                }
                all_data.append(frame_data)
                frame_count += 1
                if frame_count % BACKUP_EVERY == 0:
                    backup()

            except Exception as e:
                print(f"[LOGGER] frame capture error: {e}")

            # 로깅 주기 유지
            sleep_left = dt - (time.time() - ts)
            if sleep_left > 0:
                time.sleep(sleep_left)

    while True:
        n_det = NeedleDetection()

        ##
        # 로거 시작
        # shared["stop"] = False
        # log_thread = threading.Thread(target=logger_loop, args=(n_det,), daemon=True)
        # log_thread.start()
        # ##

        # DATA
        # blue_joint = cmd.get_joint_positions(which='PSM2') or rostopic /PSM2/measured_js
        # yellow_joint = cmd.get_joint_positions(which='PSM1') or rostopic /PSM1/measured_js
        # blue_jaw = cmd.psm2.jaw.measured_js()[0] or /PSM2/jaw/measured_js
        # yellow_jaw = cmd.psm1.jaw.masured_js()[0] or rostopic /PSM1/jaw/measured_js

        # ehdo_L, R = rostopic /dvrk/left/image_raw/compressed
        # img_L = n_det.image_L
        # img_R = n_det.image_R
        # state
        # action



        try:
            needle_detection_res = n_det.get_needle_frame()
            if needle_detection_res is None:
                print("needle_pose_w is None")
                continue

            needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = needle_detection_res
            print("needle_pose\n", needle_pose_w.tolist())



            if vis:
                visualize_needle_3d_live(ax, needle_pose_w, start_3d, end_3d, points_3d, center_3d)
                overlay_L, overlay_R = visualize_frame_projection(n_det.image_L, n_det.image_R, needle_pose_w,
                                                                  n_det.P_L, n_det.P_R, points_3d, start_3d, center_3d,
                                                                  end_3d)
                cv2.imshow("Stereo Overlay", ImgUtils.stack_stereo_img(overlay_L, overlay_R, 0.5))
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    quit()

            needle_pose_w_flatten = flatten_T(T=needle_pose_w)  # [x, y, z, qx, qy, qz, qw]
            if init_flag:
                state = np.array([-10, -10, -10])
                state = np.append(state, needle_pose_w_flatten)
                init_flag = False
            else:
                state = np.append(gp_state, needle_pose_w_flatten)    # <- 실제로는 detection한 needle pose 사용.
            print("state:", state)

            shared["state"] = state


            # inference
            action = rl.infer(state)
            shared["action"] = action
            new_state, reward, done, new_needle_pose_w, Tw_targ1, Tw_targ2, reward_type = rl.step(action, state, needle_pose_w)
            print("reward:", reward)
            if reward < -1:
                cause = list(reward_type.keys())  # 현재 reward 원인 리스트
                if 'CollisionStrategy_ground' not in cause:
                    print("Reward less than -1")  # exit signal
                    break



            # cmd
            if prev_gh is None:
                if int(new_state[2]) == 0:
                    robot = 'PSM2'
                elif int(new_state[2]) == 1:
                    robot = 'PSM1'
                else:
                    print("no robot selected for picking up")
                    break

                if robot == 'PSM2':
                    Trb_targ1 = np.linalg.inv(Tcam_rbBlue) @ Tw_targ1
                elif robot == 'PSM1':
                    Trb_targ1 = np.linalg.inv(Tcam_rbYellow) @ Tw_targ1
                Trb_targ1_prev = get_waypoint_T(Trb_targ1, distance=0.01)
                print("cur_q:", cmd.get_joint_positions(which=robot))
                print("des_q_prev:", dvrkKinematics.ik(Trb_targ1_prev))

                # 1. Approach
                keyboard_input = input("1명령 입력 (n 입력 시 set command 실행, q 종료): ")
                if keyboard_input == 'n':
                    print("set command 실행 중...")
                    cmd.close_jaw(which=robot)
                    # cmd.set_joint_positions_rel(q_targ=joint_pos_1_prev, which=robot)
                    cmd.set_pose(Trb_targ1_prev, which=robot)
                else:
                    print("입력 없음 또는 다른 입력 → 종료")
                    quit()

                # 3. OPEN JAW
                cmd.open_jaw(which=robot, angle=70.0)

                # 2. Servoing
                key = get_key_input()
                while not key == 'QUIT':
                    cmd.keyboard_servo(key, which=robot)
                    key = get_key_input()

                cmd.close_jaw(which=robot)
                time.sleep(1)
                #
                # cmd.set_joint_positions_rel(q_targ=joint_pos_1_prev, which=robot)

                T_targ = get_waypoint_T(cmd.get_ee_pose(which=robot), distance=0.01)
                print(dvrkKinematics.ik(T_targ))
                keyboard_input = input("1명령 입력 (n 입력 시 set command 실행, q 종료): ")
                if keyboard_input == 'n':
                    cmd.close_jaw(which=robot)
                    cmd.set_pose(T_targ, which=robot)
                else:
                    print("입력 없음 또는 다른 입력 → 종료")
                    quit()

                if robot == 'PSM2':
                    Trb_targ2 = np.linalg.inv(Tcam_rbBlue) @ Tw_targ2
                elif robot == 'PSM1':
                    Trb_targ2 = np.linalg.inv(Tcam_rbYellow) @ Tw_targ2

                joint_pos_2 = dvrkKinematics.ik(Trb_targ2)
                print("joint_pos_2:", joint_pos_2)
                keyboard_input = input("명령 입력 (n 입력 시 set command 실행, q 종료): ")
                if keyboard_input == 'n':
                    print("set command 실행 중...")
                    # cmd.set_joint_positions_rel(q_targ=joint_pos_2, which=robot)
                    cmd.close_jaw(which=robot)
                    cmd.set_pose(Trb_targ2, which=robot)    # joint_rel로 변경?
                else:
                    quit()

                print("pickup")
                time.sleep(1)


            else:
                # Tw_targ1은 psmblue, Tw_targ2 psmyellow always.
                if gp_state[2] == 0 and new_state[2] == 1:
                    left_to_right = True
                    giver_robot = 'PSM2'
                    receiver_robot = 'PSM1'
                elif gp_state[2] == 1 and new_state[2] == 0:
                    left_to_right = False
                    giver_robot = 'PSM1'
                    receiver_robot = 'PSM2'
                else:
                    print("Invalid GH occured")  # exit signal
                    quit()

                if left_to_right:
                     Trb_giver = np.linalg.inv(Tcam_rbBlue) @ Tw_targ1
                else:
                    Trb_giver = np.linalg.inv(Tcam_rbYellow) @ Tw_targ2

                # 1. Prepare to Give
                print("cur_q:", cmd.get_joint_positions(which=giver_robot))
                print("des_q_prev:", dvrkKinematics.ik(Trb_giver))
                keyboard_input = input("1명령 입력 (n 입력 시 set command 실행, q 종료): ")
                if keyboard_input == 'n':
                    print("set command 실행 중...")
                    cmd.close_jaw(which=giver_robot)
                    cmd.set_pose(Trb_giver, which=giver_robot)
                else:
                    print("입력 없음 또는 다른 입력 → 종료")
                    quit()

                # 2. Prepare to Receive
                if left_to_right:
                    Trb_receive = np.linalg.inv(Tcam_rbYellow) @ Tw_targ2
                else:
                    Trb_receive = np.linalg.inv(Tcam_rbBlue) @ Tw_targ1

                Trb_receive_approach = get_waypoint_T(Trb_receive, distance=0.03)
                print("receiver_q_cur:", cmd.get_joint_positions(which=receiver_robot))
                print("receiver_q_des:", dvrkKinematics.ik(Trb_receive_approach))
                keyboard_input = input("1명령 입력 (n 입력 시 set command 실행, q 종료): ")
                if keyboard_input == 'n':
                    cmd.open_jaw(which=receiver_robot)
                    cmd.set_pose(Trb_receive_approach, which=receiver_robot)
                else:
                    print("입력 없음 또는 다른 입력 → 종료")
                    quit()



                # 2. Servoing
                key = get_key_input()
                while not key == 'QUIT':
                    cmd.keyboard_servo(key, which=receiver_robot)
                    key = get_key_input()

                cmd.close_jaw(which=receiver_robot)
                time.sleep(1)

                keyboard_input = input("1명령 입력 (n 입력 시 giver robot opens jaw, q 종료): ")
                if keyboard_input == 'n':
                    cmd.open_jaw(which=giver_robot)
                else:
                    print('wrong input')
                    quit()

                Tgiver_back = cmd.get_ee_pose(which=giver_robot)
                Tgiver_back = get_waypoint_T(Tgiver_back, distance=0.03)
                print("giver_q_cur:", cmd.get_joint_positions(which=giver_robot))
                print("giver_q_des:", dvrkKinematics.ik(Tgiver_back))
                keyboard_input = input("1명령 입력 (n 입력 시 set command 실행, q 종료): ")
                if keyboard_input == 'n':
                    cmd.set_pose(Tgiver_back, which=giver_robot)
                else:
                    print('wrong input')
                    quit()


            prev_gh = state[2]
            gp_state = new_state[:3]

            if done:
                print("Reached the goal")
                quit()

        finally:
            # 항상 close
            n_det.close()
            del n_det
            print("===== NeedleDetection cycle ended. Restarting... =====")


if __name__ == "__main__":
    main()
    # main_coppelia()
