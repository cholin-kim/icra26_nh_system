import numpy as np
import cv2

K_L = np.load("cmtx0.npy")
K_R = np.load("cmtx1.npy")
distort_L = np.load("dist0.npy")
distort_R = np.load("dist1.npy")

rot = np.load("R.npy")
trans = np.load("T.npy") * 0.01 # cm -> m

P_left = np.hstack([K_L, np.zeros((3,1))])        # K [I|0]
P_right = np.hstack([K_R @ rot, K_R @ trans.reshape(3,1)]) # K [R|t]

map1_l = np.load("map1_l.npy")
map2_l = np.load("map2_l.npy")
map1_r = np.load("map1_r.npy")
map2_r = np.load("map2_r.npy")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    print("Left")
    print("K\n", K_L)
    print("dist\n", distort_L)
    print("P\n", P_left)

    print("Right")
    print("K\n", K_R)
    print("dist\n", distort_R)
    print("P\n", P_right)

    print("R\n")
    print(rot)
    from scipy.spatial.transform import Rotation as R
    print(R.from_matrix(rot).as_euler('xyz', degrees=True))
    print("T\n", trans)

    w, h = 1920, 1200
    new_camera_matrix_l, _ = cv2.getOptimalNewCameraMatrix(K_L, distort_L, (w, h), 1, (w, h))
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_L, distort_L, None, new_camera_matrix_l, (w, h), cv2.CV_32FC1)

    new_camera_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(K_R, distort_R, (w, h), 1, (w, h))
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_R, distort_R, None, new_camera_matrix_r, (w, h), cv2.CV_32FC1)

    # np.save("map1_l.npy", map1_l)
    # np.save("map2_l.npy", map2_l)
    # np.save("map1_r.npy", map1_r)
    # np.save("map2_r.npy", map2_r)

