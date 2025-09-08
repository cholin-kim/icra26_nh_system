import copy
import cv2
import numpy as np

from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class Aruco_Detect():
    def __init__(self):

        self.distCoeffs = np.load("needleDetect/camera_calibration/dist0.npy")
        self.camMatrix = np.array("needleDetect/camera_calibration/cmtx0.npy")





        ## Initialize ##
        self.cv2_img = 0
        self.corners = 0
        self.ids = 0
        self.rejected = 0
        self.aruco_visualization = 0
        self.detect_flag = False



        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        arucoParam = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)




    def find_aruco(self, img):
        self.corners, self.ids, self.rejected = self.detector.detectMarkers(img)
        print("Detected markers: ", self.ids, "#", len(self.corners))

        if len(self.corners) > 0 and np.max(self.ids) <= 5 and np.min(self.ids) >= 0:
            self.ids = self.ids.flatten()  ## {NoneType} object has no attribute 'flatten', convert to shape (n,)
            self.detect_flag = True
            self.publish_aruco()
            translate_x = []
            translate_y = []
            translate_z = []
            roll_x_lst = []
            pitch_y_lst = []
            yaw_z_lst = []

            if self.detect_flag == True:
                for i in range(len(self.corners)):
                    pose = np.zeros(7)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], 0.06,
                                                                                   self.camMatrix, self.distCoeffs)
                    x, y, z = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
                    pose[:3] = [x, y, z]
                    [[quat_x, quat_y, quat_z, quat_w]] = R.from_rotvec(rvec.reshape(1,
                                                                                    3)).as_quat()  ## R.from_{np.ndarray.shape (1,n)} > return type array([[_, _, _, _]]) (1, m)
                    pose[3:] = [quat_x, quat_y, quat_z, quat_w]

                    # translate_x.append(round(x * 100, 2))
                    # translate_y.append(round(y * 100, 2))
                    # translate_z.append(round(z * 100, 2))
                    # [roll_x, pitch_y, yaw_z] = R.from_quat(np.array([quat_x, quat_y, quat_z, quat_w])).as_euler('xyz',
                    #                                                                                             degrees=True)  ## input type (4,) > output type (3,)
                    # roll_x_lst.append(round(roll_x, 2))
                    # pitch_y_lst.append(round(pitch_y, 2))
                    # yaw_z_lst.append(round(yaw_z, 2))

                    cv2.aruco.drawDetectedMarkers(img, self.corners)
                    cv2.drawFrameAxes(img, self.camMatrix, self.distCoeffs, rvec, tvec, 0.03, 1)

                self.put_text([translate_x, translate_y, translate_z, roll_x_lst, pitch_y_lst, yaw_z_lst])
        else:
            self.detect_flag = False
            print("searching again")

            translate_x = []
            translate_y = []
            translate_z = []
            roll_x_lst = []
            pitch_y_lst = []
            yaw_z_lst = []

            if self.detect_flag == True:
                for i in range(len(self.corners)):
                    pose = np.zeros(7)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], 0.06,
                                                                                   self.camMatrix, self.distCoeffs)
                    x, y, z = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
                    pose[:3] = [x, y, z]
                    [[quat_x, quat_y, quat_z, quat_w]] = R.from_rotvec(rvec.reshape(1,
                                                                                    3)).as_quat()  ## R.from_{np.ndarray.shape (1,n)} > return type array([[_, _, _, _]]) (1, m)
                    pose[3:] = [quat_x, quat_y, quat_z, quat_w]


                    # translate_x.append(round(x * 100, 2))
                    # translate_y.append(round(y * 100, 2))
                    # translate_z.append(round(z * 100, 2))
                    # [roll_x, pitch_y, yaw_z] = R.from_quat(np.array([quat_x, quat_y, quat_z, quat_w])).as_euler('xyz',
                    #                                                                                             degrees=True)  ## input type (4,) > output type (3,)
                    # roll_x_lst.append(round(roll_x, 2))
                    # pitch_y_lst.append(round(pitch_y, 2))
                    # yaw_z_lst.append(round(yaw_z, 2))

                    cv2.aruco.drawDetectedMarkers(img, self.corners)
                    cv2.drawFrameAxes(img, self.camMatrix, self.distCoeffs, rvec, tvec, 0.03, 1)

                self.put_text([translate_x, translate_y, translate_z, roll_x_lst, pitch_y_lst, yaw_z_lst])
            return pose


    def put_text(self, pose_lst):


        pose_lst = np.array(pose_lst).reshape(6, -1)


        for j in range(len(self.corners)):
            corner = self.corners[j].flatten()[0:2]

            cv2.putText(self.frame, f"{self.ids[j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]-60)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[0][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]-40)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[1][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]-20)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[2][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1])), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[3][j]}",(int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]+20)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[4][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]+40)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[5][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]+60)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))


if __name__ == '__main__':
    ad = Aruco_Detect()
    img = #
    pose = ad.find_aruco(img)
    print("_________________________________")