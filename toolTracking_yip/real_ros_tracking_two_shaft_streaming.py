import cv2
import copy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
import sys, os
sys.path.append('/home/surglab/icra26_nh_system/')
from img_proc_two_shaft import *
from particle_filter import SimpleParticleFilter
from needleDetect.Basler import Basler
import crtk
import numpy as np


sys.path.append("/home/surglab/pycharmprojects")
from dvrk_surglab.motion.psm import psm
import time




ral = crtk.ral(node_name='dvrkCtrl')

ral.check_connections()
ral.spin()

psm1 = psm(ral=ral, arm_name='PSM1')
psm2 = psm(ral=ral, arm_name='PSM2')

# tvec, rvec = [0.009620040250292524, 0.0001692052469176161, 0.15000193981221874], [2.00664984346439,
#                                                                                   0.005120409365822809,
#                                                                                   0.09097453687127555]

# TrbYellow_ee = kin.fk(psm1.measured_js()[0])[0][-1]
# # print(TrbYellow_ee)
# Tcam_ee = np.identity(4)
# Tcam_ee[:3, -1] = tvec
# Tcam_ee[:3, :3] = R.from_euler('XYZ', rvec).as_matrix()
# Tcam_rbYellow = Tcam_ee @ np.linalg.inv(TrbYellow_ee)
# print(Tcam_rbYellow.tolist())
# quit()


# TrbBlue_ee = kin.fk(psm2.measured_js()[0])[0][-1]
# Tcam_ee = np.identity(4)
# Tcam_ee[:3, -1] = tvec
# Tcam_ee[:3, :3] = R.from_euler('XYZ', rvec).as_matrix()
# Tcam_rbBlue = Tcam_ee @ np.linalg.inv(TrbBlue_ee)
# print(Tcam_rbBlue.tolist())
# quit()

# Initialize
pf_blue = SimpleParticleFilter(num_particles=200, state_dim=6)
pf_yellow = SimpleParticleFilter(num_particles=200, state_dim=6)
lumped_error = np.zeros(6)

# video_path = "data_1/Left.avi"
# cap = cv2.VideoCapture(video_path)
#
# joint_blue_ary = np.load("data_1/psm1_joint_pos.npy")
# joint_green_ary = np.load("data_1/psm2_joint_pos.npy")

cam_L = Basler(serial_number="40262045")
# cam_R = Basler(serial_number="40268300")
cam_L.start()
# cam_R.start()q
time.sleep(0.1)



def pose2T(pose):
    T = np.eye(4)
    T[:3, -1] = pose[:3]
    T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    return T
Tcam_rbBlue = np.array(
    [[0.958227689670898, -0.21768249007965917, -0.18551018371154965, 0.17551387734934729],
     [-0.09430895426216324, 0.37185710016414775, -0.9234869345061079, -0.17092286002679188],
     [0.27001021442521606, 0.9024060231238654, 0.33579436197741447, 0.19878618409315504], [0.0, 0.0, 0.0, 1.0]]

)
Tcam_rbYellow = np.array(
    [[0.994042862572803, -0.013175090869826397, -0.10819059269936301, -0.2179224449418338],
     [-0.09753726435562539, 0.33541623381749125, -0.9370071676106384, -0.16470283851810743],
     [0.048634035716651244, 0.9419779015838229, 0.3321330508962478, 0.2105987610433221], [0.0, 0.0, 0.0, 1.0]]
)

# TrbBlue2rbBlue = pose2T([0.007071970785560316, 0.008979825616317329, -0.011368533475678259, -0.032487093509209274, 0.047368678832232086, 0.04126209962348458])
# Tcam_rbBlue = Tcam_rbBlue @ TrbBlue2rbBlue

TrbYellow_rbYellow = pose2T([0.02161299290230202, -0.061331324884405236, -0.025230796235099287, 0.023009687863923413, 0.05763043654491389, 0.02936060978489949])
Tcam_rbYellow = Tcam_rbYellow @ TrbYellow_rbYellow


# print(Tcam_rbBlue.tolist())
print(Tcam_rbYellow.tolist())
quit()

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('image', cv2.WINDOW_FULLSCREEN)


# i = -1
# while cap.isOpened():

def soft_likelihood(proj, obs, gamma=0.5):
    if len(proj) == 0 or len(obs) == 0: return 1e-6
    dist = np.linalg.norm(proj[:, None, :] - obs[None, :, :], axis=2)
    score = np.exp(-gamma * dist).max(axis=1)
    return np.mean(score)


init_flag = 0
decay_d = 0.01
while True:
    # i += 1
    # ret, img = cap.read()
    # if not ret:
    #     break
    img = cam_L.image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    observed_centrs_blue, img = keypoint_segmentation_centroid(img, "Blue", visualize=True)
    observed_centrs_yellow, img = keypoint_segmentation_centroid(img, "Yellow", visualize=True)
    flag_keyP_blue, flag_keyP_yellow, flag_line = True, True, True
    if len(observed_centrs_blue) == 0:
        flag_keyP_blue = False
    if len(observed_centrs_yellow) == 0:
        flag_keyP_yellow = False

    observed_shaft_lines, img = detect_shaft_with_hough(img, visualize=False)

    if len(observed_shaft_lines) == 0 or len(observed_shaft_lines) != 4:
        if flag_keyP_blue:
            obs_blue = observed_centrs_blue
        if flag_keyP_yellow:
            obs_yellow = observed_centrs_yellow
        flag_line = False
        flag_line = False
    else:
        if flag_keyP_blue:
            obs_blue = np.vstack((observed_centrs_blue, observed_shaft_lines[2:]))
        if flag_keyP_yellow:
            obs_yellow = np.vstack((observed_centrs_yellow, observed_shaft_lines[:2]))

    ####
    noise_std = 0.001
    # noise_std = 0.002
    ####
    if flag_keyP_blue:
        pf_blue.predict(noise_std=noise_std)

    if flag_keyP_yellow:
        pf_yellow.predict(noise_std=noise_std)

    # cur_joint_blue = joint_blue_ary[i]
    # cur_joint_yellow = joint_green_ary[i]

    cur_joint_blue = psm2.measured_js()[0]
    cur_joint_yellow = psm1.measured_js()[0]
    cur_joint_blue_jaw = psm2.jaw.measured_js()[0]
    cur_joint_yellow_jaw = psm1.jaw.measured_js()[0]


    def compute_likelihood_blue(particles, gamma=0.1, threshold=100):
        # projected_pts_key, _ = key_point_projection_vectorized(cur_joint_blue, particles, Tcam_rbBlue, img=img)
        projected_pts_key, _ = key_point_projection_vectorized(cur_joint_blue, particles, Tcam_rbBlue, img=img,
                                                               jaw=cur_joint_blue_jaw)

        N = projected_pts_key.shape[0]
        for i in range(N):
            for pt in projected_pts_key[i]:
                cv2.circle(img, tuple(pt), radius=2, color=(0, 255, 255), thickness=1)  # 하늘색

        # projected_pts_shaft, _ = shaft_projection_vectorized(cur_joint_blue, particles, Tcam_rbBlue)
        projected_pts_shaft = []
        if flag_line:
            projected_pts = np.concatenate([projected_pts_key, projected_pts_shaft], axis=1)
        else:
            projected_pts = projected_pts_key

        # projected_pts = projected_pts_key
        C_all = np.linalg.norm(projected_pts[:, :, None, :] - obs_blue[None, None, :, :], axis=3)
        likelihoods = []
        for C in C_all:
            row_idx, col_idx = linear_sum_assignment(C)

            # Only keep assignments with a cost below the threshold.
            valid = C[row_idx, col_idx] < threshold
            row_idx = row_idx[valid]
            col_idx = col_idx[valid]

            m = C.shape[0]  # number of projected points for this particle
            likelihood = (5 * np.sum(np.exp(-gamma * C[row_idx, col_idx])) +
                          (m - len(row_idx)) * np.exp(-gamma * threshold))
            likelihoods.append(likelihood)

        return np.array(likelihoods)


    def compute_likelihood_yellow(particles, gamma=0.1, threshold=100):
        # projected_pts_key, _ = key_point_projection_vectorized(cur_joint_yellow, particles, Tcam_rbYellow, img=img)
        projected_pts_key, _ = key_point_projection_vectorized(cur_joint_yellow, particles, Tcam_rbYellow, img=img,
                                                               jaw=cur_joint_yellow_jaw)

        projected_pts_key = projected_pts_key.astype(int)
        N = projected_pts_key.shape[0]
        for i in range(N):
            for pt in projected_pts_key[i]:
                x, y = pt
                if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
                    continue  # 이미지 범위 밖 좌표는 무시
                cv2.circle(img, tuple(pt), radius=2, color=(255, 255, 0), thickness=1)  # 노란색
        # projected_pts_shaft, _ = shaft_projection_vectorized(cur_joint_yellow, particles, Tcam_rbYellow)
        projected_pts_shaft = []
        if flag_line:
            projected_pts = np.concatenate([projected_pts_key, projected_pts_shaft], axis=1)

        else:
            projected_pts = projected_pts_key

        # projected_pts = projected_pts_key
        C_all = np.linalg.norm(projected_pts[:, :, None, :] - obs_yellow[None, None, :, :], axis=3)
        likelihoods = []
        for C in C_all:
            row_idx, col_idx = linear_sum_assignment(C)

            # Only keep assignments with a cost below the threshold.
            valid = C[row_idx, col_idx] < threshold
            row_idx = row_idx[valid]
            col_idx = col_idx[valid]

            m = C.shape[0]  # number of projected points for this particle
            likelihood = (5 * np.sum(np.exp(-gamma * C[row_idx, col_idx])) +
                          (m - len(row_idx)) * np.exp(-gamma * threshold))
            likelihoods.append(likelihood)

        return np.array(likelihoods)

    #
    # def compute_likelihood_yellow_soft(particles):
    #     likelihoods = []
    #     obs = obs_yellow
    #     for p in particles:
    #         proj, _ = key_point_projection_vectorized(cur_joint_yellow, p[None], Tcam_rbYellow,
    #                                                   jaw=cur_joint_yellow_jaw)
    #         proj = proj[0]
    #         proj = proj[np.isfinite(proj).all(axis=1)]
    #         likelihoods.append(soft_likelihood(proj, obs))
    #     return np.array(likelihoods)


    if flag_keyP_blue:  # 점이 아무것도 안보이는 경우에는 particle filter 가만히 냅두기
        pf_blue.update(compute_likelihood_blue)
    if flag_keyP_yellow:
        pf_yellow.update(compute_likelihood_yellow)

    lumped_error_blue = pf_blue.estimate()
    lumped_error_yellow = pf_yellow.estimate()
    print(f'lumped error Blue :', lumped_error_blue.tolist())
    print(f'lumped error Yellow :', lumped_error_yellow.tolist())

    # Blue
    projected_p, _ = key_point_projection(cur_joint_blue, lumped_error_blue, Tcam_rbBlue,
                                          jaw=cur_joint_blue_jaw)  # lumped error를 포함하여 예측
    for p_ in projected_p:
        # for ind_p in range(len(row_idx)):
        #     if row_idx[ind_p] >= len(projected_p): continue
        #     p_ = projected_p[row_idx[ind_p]]
        cv2.circle(img, p_, 5, (255, 0, 0), thickness=10)

    # projected_p, _ = key_point_projection(cur_joint_blue, np.zeros(6), Tcam_rbBlue)  # lumped error를 포함하지 않고 예측
    # # print(projected_p)
    # for p_ in projected_p:
    # # for ind_p in range(len(row_idx)):
    # #     if row_idx[ind_p] >= len(projected_p): continue
    # #     p_ = projected_p[row_idx[ind_p]]
    #     cv2.circle(img, p_, 1, (0, 255, 0), thickness=10)

    # projected_p, _ = shaft_projection(cur_joint_blue, lumped_error_blue, cmd.get_base2cam_blue(), img, visualize=True)

    # Yellow
    projected_p, _ = key_point_projection(cur_joint_yellow, lumped_error_yellow, Tcam_rbYellow,
                                          jaw=cur_joint_yellow_jaw)  # lumped error를 포함하여 예측
    for p_ in projected_p:
        # for ind_p in range(len(row_idx)):
        #     if row_idx[ind_p] >= len(projected_p): continue
        #     p_ = projected_p[row_idx[ind_p]]
        cv2.circle(img, p_, 5, (255, 100, 100), thickness=10)

    # projected_p, _ = key_point_projection(cur_joint_yellow, np.zeros(6), Tcam_rbYellow)  # lumped error를 포함하지 않고 예측
    # # print(projected_p)
    # for p_ in projected_p:
    # # for ind_p in range(len(row_idx)):
    # #     if row_idx[ind_p] >= len(projected_p): continue
    # #     p_ = projected_p[row_idx[ind_p]]
    #     cv2.circle(img, p_, 1, (0, 255, 0), thickness=10)

    # projected_p, _ = shaft_projection(cur_joint_yellow, lumped_error_yellow, cmd.get_base2cam_yellow(), img, visualize=True)
    # cv2.putText(img, f'{np.mean(fps, dtype=int)} fps', (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2,
    #             lineType=cv2.LINE_AA)

    cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    continue

cv2.destroyAllWindows()
quit()
# out.release()
