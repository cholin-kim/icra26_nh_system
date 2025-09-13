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

from pycpd import RigidRegistration


def cpd_match(projected_pts, observed_pts, max_iter=50, tolerance=1e-5):
    """
    projected_pts: (N,2) numpy array
    observed_pts: (M,2) numpy array
    returns: correspondence probabilities (N,M)
    """
    reg = RigidRegistration(X=observed_pts, Y=projected_pts, max_iterations=max_iter, tolerance=tolerance)
    TY, (s, R, t) = reg.register()
    # P matrix (NxM) : correspondence probabilities
    P = reg.P  # from EM step
    return TY, P


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
    [[0.9553300608635766, -0.08034125246891978, -0.2844112479528908, 0.14979430848282968],
     [-0.23423944051178844, 0.3809728437608613, -0.894422482289822, -0.19533547004104282],
     [0.18021198439371292, 0.9210890160394315, 0.34513571998909726, 0.17552939516244073],
     [0.0, 0.0, 0.0, 1.0]]
)
TrbBlue2rbBlue = pose2T([ 0.01150112 , 0.00919402, -0.0074815   ,0.05454021,  0.0188427,   0.06605713])
Tcam_rbBlue = Tcam_rbBlue @ TrbBlue2rbBlue
TrbBlue2rbBlue = pose2T([0.006586822557409947, 0.008576872085468375, -0.0029563590403755224, 0.014256251136074195, 0.034689057749241296, 0.024206632675366466])
Tcam_rbBlue = Tcam_rbBlue @ TrbBlue2rbBlue




Tcam_rbYellow = np.array(
    [[0.9871629728014177, -0.005432605252827723, -0.15962378247001394, -0.2091070980192582],
     [-0.1510453377860188, 0.2930666973085455, -0.9440853864252792, -0.1364412677764036],
     [0.05190925796979646, 0.9560765647838193, 0.2884840224487254, 0.13940613021486847],
     [0.0, 0.0, 0.0, 1.0]]
)
TrbYellow_rbYellow = pose2T([ 0.00194834, -0.00276434,  0.0042234,   0.01130029, -0.01008123,  0.00120943])
Tcam_rbYellow = Tcam_rbYellow @ TrbYellow_rbYellow
TrbYellow_rbYellow = pose2T([0.010279771865884425, -0.010399723700597864, -0.004296959179342556, 0.0055935908481995434, 0.07552836668194551, -0.026764137920276966])
Tcam_rbYellow = Tcam_rbYellow @ TrbYellow_rbYellow

print(Tcam_rbBlue.tolist())
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
