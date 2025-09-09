from logging import raiseExceptions
import threading
from pypylon import pylon
import cv2
import numpy as np
# from imutils.face_utils import visualize_facial_landmarks
from scipy.spatial.transform import Rotation

from cam_utils import *
from scipy.spatial.transform import Rotation as R
from itertools import combinations
from img_proc import *
from particle_filter import SimpleParticleFilter
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import Image

import rospy
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import panda.pandaKinematics as panda
# from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class ros_cmd:
    def __init__(self):
        rospy.init_node('coppeliasim_remote_zmq', anonymous=True)
        self.bridge = CvBridge()
        self.panda = panda.pandaKinematics()

        self.q_panda = rospy.wait_for_message('/panda1/franka_state_controller/joint_states', JointState).position
        print('Program started1')
        # self.q_cam = rospy.wait_for_message('/fr3/joint_states', JointState).position
        print('Program started2')
        self.q_psm = rospy.wait_for_message('/panda1/dvrk/joint_states', JointState).position
        print('Program started3')
        rospy.Subscriber('/panda1/dvrk/joint_states', JointState, self.joint_cb_psm)
        # rospy.Subscriber('/fr3/joint_states', JointState, self.joint_cb_cam)
        rospy.Subscriber('/panda1/franka_state_controller/joint_states', JointState, self.joint_cb_panda)

        self.T_base2cam = np.load('T_base2cam.npy')
        
        self.T_base2cbase = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, -17*0.025],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]]) # 고정

        self.T_ee2cam = np.eye(4)
        self.T_ee2cam[:3, :3] = Rotation.from_euler(seq='ZXY', angles=[np.pi / 4, 0, 0]).as_matrix()
        # self.T_ee2cam[:3, :3] = Rotation.from_euler(seq='ZXY', angles=[-np.pi / 4, np.pi, np.pi / 2]).as_matrix()
        # self.T_ee2cam[:3, :3] = Rotation.from_euler(seq='XY', angles=[np.pi, np.pi / 2]).as_matrix()
        self.T_ee2cam[0, 3] = 0.017677
        self.T_ee2cam[1, 3] = -0.017677
        self.T_ee2cam[2, 3] = 0.15
        print(self.T_ee2cam)

        jointNames = ['/joint1', '/joint2', '/joint3', '/joint4', '/joint5', '/joint6', '/joint7', '/panda_joint_roll',
                      '/panda_joint_wrist', '/panda_joint_jaw1', '/panda_joint_jaw2']

    def joint_cb_panda(self, msg:JointState):
        self.q_panda = np.array(msg.position)

    # def joint_cb_cam(self, msg:JointState):
    #     self.q_cam = np.array(msg.position)
    #     T_cbase2ee, _ = self.panda.fk(self.q_cam)
    #     T_cbase2cam = T_cbase2ee[-1] @ self.T_ee2cam
    #     self.T_base2cam = self.T_base2cbase.dot(T_cbase2cam)
    #     # np.save('T_base2cam.npy', self.T_base2cam)

    def joint_cb_psm(self, msg:JointState):
        self.q_psm = np.array(msg.position)
    
    def get_joint_pos(self):
        cur_joint = np.concatenate([self.q_panda, self.q_psm])
        return cur_joint
    
    def get_base2cam(self):
        return self.T_base2cam

class Basler(threading.Thread):
    def __init__(self, serial_number):
        threading.Thread.__init__(self)
        self.stop_flag = False

        # Get the transport layer factory.
        tlf = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlf.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        for d in devices:
            if d.GetSerialNumber() == serial_number:
                print(f"Model= {d.GetModelName()}, Serial= {d.GetSerialNumber()}")
                self.cam = pylon.InstantCamera(tlf.CreateDevice(d))
                self.cam.Open()

        # Grabing Continusely (video) with minimal delay
        self.cam.StartGrabbing()

        # converting to opencv bgr format
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.image = []

    def __del__(self):
        self.stop()

    def run(self):
        while self.cam.IsGrabbing():
            if self.stop_flag:
                print ("stop flag detected")
                break
            grabResult = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                self.image = image.GetArray()

        # Releasing the resource
        self.cam.StopGrabbing()
        self.cam.Close()
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_flag = True

print('Program started')
import time
cmd = ros_cmd()
cam_L = Basler(serial_number="40262045")
cam_L.start()
time.sleep(0.1)
print('Program started')

####################### 수정 ########################
####################### 수정 ########################
####################### 수정 ########################
K = np.array([
    [1.82685922e+03, 0.00000000e+00, 9.81013446e+02],
     [0.00000000e+00, 1.82642713e+03, 5.71633006e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
D = np.zeros(5)
####################### 수정 ########################
####################### 수정 ########################
####################### 수정 ########################

pf = SimpleParticleFilter(num_particles=200, state_dim=6)
lumped_error = np.zeros(6)
max_std =0.0000000000007
min_std =0.001
decay_rate = 0.005
noise_std = max_std
pf.predict(noise_std=noise_std)
itr = 0

w, h = 480, 680
fps = 10

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
i = 0
fps = []


start_time = time.time()    # 만약 std decay rate를 시간 기준으로 하고 싶을 때

while not rospy.is_shutdown():
    # try:
        st = time.time()

        img = cam_L.image
        img = np.array(img)
        # cv2.imwrite("sample_img2.jpg", img)

        # cv2.imshow('image', img)
        # fps.append(int(1 / (time.time() - st)))
        # # print(f'len_fps : {len(fps)}')
        # if len(fps) > 20:
        #     fps.pop(0)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
        #
        # continue

        # b_obs_s = True
        # show_obs_s = True
        # #
        # observed_centrs, img_mask = keypoint_segmentation_centroid(img, visualize=True)
        #
        # observed_shaft_lines = None
        # observed_shaft_lines, img = detect_shaft_with_hough(img, visualize=show_obs_s)
        # observed_shaft_lines = np.array(observed_shaft_lines)
        #
        # flag_keyP, flag_line = True, True
        # if len(observed_centrs) == 0: flag_keyP = False
        # if len(observed_shaft_lines) == 0: flag_line = False
        # # obs = observed_centrs
        #
        # if flag_keyP:      # key point가 아무것도 안보이는 경우에는 particle filter 가만히 냅두기
        #     pf.predict(noise_std=0.001)
        # cur_joint = cmd.get_joint_pos()
        #
        #
        #
        # def compute_likelihood(particles, gamma=0.1, c_thrsd=50, s_thrsd=[30, 0.5]):
        #     projected_p, _ = key_point_projection_vectorized(cur_joint, particles, cmd.get_base2cam())
        #     projected_pts_shaft, _ = shaft_projection_vectorized(cur_joint, particles, cmd.get_base2cam())
        #
        #     C_all = np.linalg.norm(projected_p[:, :, None, :] - observed_centrs[None, None, :, :], axis=3)
        #
        #     S_rho, S_phi = None, None
        #     if b_obs_s and flag_line:
        #         projected_pts_shaft, _ = shaft_projection_vectorized(cur_joint, particles, cmd.get_base2cam())
        #         S_rho = np.abs(projected_pts_shaft[:, :, None, 0] - observed_shaft_lines[None, None, :, 0])
        #         S_phi = np.abs(projected_pts_shaft[:, :, None, 1] - observed_shaft_lines[None, None, :, 1])
        #
        #     likelihoods = []
        #     for p_idx in range(len(projected_p)):  # index of particle
        #         C = C_all[p_idx]
        #         C_row_idx, C_col_idx = linear_sum_assignment(C)
        #
        #         valid = C[C_row_idx, C_col_idx] < c_thrsd  # Only keep assignments with a cost below the threshold.
        #         C_row_idx = C_row_idx[valid]
        #         C_col_idx = C_col_idx[valid]
        #
        #         m = C.shape[0]  # number of projected points for this particle
        #         likelihood = np.sum(np.exp(-gamma * C[C_row_idx, C_col_idx])) + (m - len(C_row_idx)) * np.exp(
        #             -gamma * c_thrsd)
        #         # point도 안보이면 그냥 update를 안할 예정이라서... 그냥 무조건 point에 대해서는 평가하도록 구현함
        #
        #         if b_obs_s and flag_line:
        #             coeff = 20
        #             S = S_rho[p_idx] + coeff * S_phi[p_idx]
        #             S_row_idx, S_col_idx = linear_sum_assignment(S)
        #
        #             valid = np.logical_and(
        #                 (S_rho[p_idx, S_row_idx, S_col_idx] < s_thrsd[0]),
        #                 (S_phi[p_idx, S_row_idx, S_col_idx] < s_thrsd[1])
        #             )
        #             S_row_idx = S_row_idx[valid]
        #             S_col_idx = S_col_idx[valid]
        #
        #             n = 2  # number of observed line
        #             likelihood += np.sum(np.exp(-gamma * S[S_row_idx, S_col_idx]))
        #             likelihood += (n - len(S_row_idx)) * np.exp(-gamma * (s_thrsd[0] + coeff * s_thrsd[1]))
        #
        #         likelihoods.append(likelihood)
        #     return np.array(likelihoods)

        observed_centrs, img = keypoint_segmentation_centroid(img, visualize=True)
        observed_shaft_lines, img = detect_shaft_with_hough(img, visualize=True)

        # cv2.imwrite("before hole Filling.jpg", img)
        # cv2.imshow('image', img)
        # fps.append(int(1 / (time.time() - st)))
        # # print(f'len_fps : {len(fps)}')
        # if len(fps) > 20:
        #     fps.pop(0)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
        #
        # continue

        flag_keyP, flag_line = True, True

        if len(observed_centrs) == 0:
            flag_keyP = False
        if len(observed_shaft_lines) == 0:
            if flag_keyP:
                obs = observed_centrs
            flag_line = False
        else:
            if flag_keyP:
                obs = np.vstack((observed_centrs, observed_shaft_lines))

        # obs = observed_centrs


        noise_std = 0.0007
        if flag_keyP:
            pf.predict(noise_std=noise_std)
        cur_joint = cmd.get_joint_pos() #+ err
        # print(f'err : {np.rad2deg(err)}')
        # flag_keyP, flag_line = True, True
        # if len(observed_centrs) == 0: flag_keyP = False
        # if len(observed_shaft_lines) == 0: flag_line = False



        def compute_likelihood(particles, gamma=0.1, threshold=100):
            projected_pts_key, _ = key_point_projection_vectorized(cur_joint, particles, cmd.get_base2cam())
            projected_pts_shaft, _ = shaft_projection_vectorized(cur_joint, particles, cmd.get_base2cam())
            if flag_line:
                projected_pts = np.concatenate([projected_pts_key, projected_pts_shaft], axis=1)

            else:
                projected_pts = projected_pts_key

            # projected_pts = projected_pts_key
            C_all = np.linalg.norm(projected_pts[:, :, None, :] - obs[None, None, :, :], axis=3)
            likelihoods = []
            for C in C_all:
                row_idx, col_idx = linear_sum_assignment(C)

                # Only keep assignments with a cost below the threshold.
                valid = C[row_idx, col_idx] < threshold
                row_idx = row_idx[valid]
                col_idx = col_idx[valid]

                m = C.shape[0]  # number of projected points for this particle
                likelihood = (5* np.sum(np.exp(-gamma * C[row_idx, col_idx])) +
                              (m - len(row_idx)) * np.exp(-gamma * threshold))
                likelihoods.append(likelihood)

            return np.array(likelihoods)


        if flag_keyP:      # 점이 아무것도 안보이는 경우에는 particle filter 가만히 냅두기
            pf.update(compute_likelihood)
        # pf.update(compute_likelihood)

        lumped_error = pf.estimate()
        print(f'lumped error :', lumped_error)


        # projected_pts_key, _ = key_point_projection(cur_joint, lumped_error, cmd.get_base2cam())
        # projected_pts_shaft, _ = shaft_projection(cur_joint, lumped_error, cmd.get_base2cam())
        #
        # if flag_line:
        #     projected_pts = np.vstack([projected_pts_key, projected_pts_shaft])
        # else:
        #     projected_pts = np.array(projected_pts_key)
        #
        # C = np.linalg.norm(projected_pts[:, None, :] - obs[None, :, :], axis=2)
        #
        # row_idx, col_idx = linear_sum_assignment(C)
        #
        # # Only keep assignments with a cost below the threshold.
        # valid = C[row_idx, col_idx] < 50
        # row_idx = row_idx[valid]
        # col_idx = col_idx[valid]







        proj_ps = []
        projected_p, _ = key_point_projection(cur_joint, lumped_error, cmd.get_base2cam())  # lumped error를 포함하여 예측
        for p_ in projected_p:
        # for ind_p in range(len(row_idx)):
        #     if row_idx[ind_p] >= len(projected_p): continue
        #     p_ = projected_p[row_idx[ind_p]]
            cv2.circle(img, p_, 1, (0, 0, 255), thickness=10)

        projected_p, _ = key_point_projection(cur_joint, np.zeros(6), cmd.get_base2cam())  # lumped error를 포함하지 않고 예측
        # print(projected_p)
        for p_ in projected_p:
        # for ind_p in range(len(row_idx)):
        #     if row_idx[ind_p] >= len(projected_p): continue
        #     p_ = projected_p[row_idx[ind_p]]
            cv2.circle(img, p_, 1, (255, 0, 255), thickness=10)

        projected_p, _ = shaft_projection(cur_joint, lumped_error, cmd.get_base2cam(), img, visualize=True)
        cv2.putText(img, f'{np.mean(fps, dtype=int)} fps', (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.imshow('image', img)
        fps.append(int(1 / (time.time() - st)))
        # print(f'len_fps : {len(fps)}')
        if len(fps) > 20:
            fps.pop(0)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print(f'duration : {time.time() - st} sec')
        print()

    # except:
    #     st = time.time()
    #
    #     img = cam_L.image
    #     img = np.array(img)
    #     projected_p, _ = key_point_projection(cur_joint, lumped_error, cmd.get_base2cam())  # lumped error를 포함하여 예측
    #     for p_ in projected_p:
    #     # for ind_p in range(len(row_idx)):
    #     #     if row_idx[ind_p] >= len(projected_p): continue
    #     #     p_ = projected_p[row_idx[ind_p]]
    #         cv2.circle(img, p_, 1, (0, 0, 255), thickness=10)
    #
    #     projected_p, _ = key_point_projection(cur_joint, np.zeros(6), cmd.get_base2cam())  # lumped error를 포함하지 않고 예측
    #     # print(projected_p)
    #     for p_ in projected_p:
    #     # for ind_p in range(len(row_idx)):
    #     #     if row_idx[ind_p] >= len(projected_p): continue
    #     #     p_ = projected_p[row_idx[ind_p]]
    #         cv2.circle(img, p_, 1, (255, 0, 255), thickness=10)
    #
    #     projected_p, _ = shaft_projection(cur_joint, lumped_error, cmd.get_base2cam(), img, visualize=True)
    #     cv2.putText(img, f'{np.mean(fps, dtype=int)} fps', (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2,
    #                 lineType=cv2.LINE_AA)
    #     cv2.imshow('image', img)
    #     fps.append(int(1 / (time.time() - st)))
    #     # print(f'len_fps : {len(fps)}')
    #     if len(fps) > 20:
    #         fps.pop(0)
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    #
    #     print(f'duration : {time.time() - st} sec')
    #     print()
    #
    #     pass

# out.release()
cur_q = cmd.get_joint_pos()
cur_q[0] = 0
cur_q[-4:] = np.zeros(4)
print('Closing Scene')