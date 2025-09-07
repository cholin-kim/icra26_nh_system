import cv2
import numpy as np
from cv2 import waitKey

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# from dvrk.dvrk_Kinematics import dvrkKinematics


class CoppeliaCmd:
    def __init__(self):
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.l_vision_handle = self.sim.getObject('/Vision_sensor_left')
        self.r_vision_handle = self.sim.getObject('/Vision_sensor_right')
        self.PSM1_joint_handls = [self.sim.getObject('/J1_PSM1'), self.sim.getObject('/J2_PSM1'), self.sim.getObject('/J3_PSM1'),
                                  self.sim.getObject('/J1_TOOL1'), self.sim.getObject('/J2_TOOL1'),
                                  self.sim.getObject('/J3_dx_TOOL1'), self.sim.getObject('/J3_sx_TOOL1')]
        self.PSM2_joint_handls = [self.sim.getObject('/J1_PSM2'), self.sim.getObject('/J2_PSM2'), self.sim.getObject('/J3_PSM2'),
                                  self.sim.getObject('/J1_TOOL2'), self.sim.getObject('/J2_TOOL2'),
                                  self.sim.getObject('/J3_dx_TOOL2'), self.sim.getObject('/J3_sx_TOOL2')]
        self.needle = '/Needle_origin'
        self.world = '/World'
        self.RCM_PSM1 = '/RCM_PSM1'
        self.RCM_PSM2 = '/RCM_PSM2'

        self.break_flag = False
        self.sim.startSimulation()
        print('Program started')

        Tw_psm1 = self.sim.getObjectPose(self.sim.getObject(self.RCM_PSM1), self.sim.getObject(self.world))  # rb1_w [x y z qx qy qz qw]
        Tw_psm2 = self.sim.getObjectPose(self.sim.getObject(self.RCM_PSM2), self.sim.getObject(self.world))   #

        self.Tw_psm1 = np.identity(4)
        self.Tw_psm2 = np.identity(4)
        self.Tw_psm1[:3, -1] = Tw_psm1[:3]
        self.Tw_psm1[:3, :3] = R.from_quat(Tw_psm1[3:]).as_matrix()
        self.Tw_psm2[:3, -1] = Tw_psm2[:3]
        self.Tw_psm2[:3, :3] = R.from_quat(Tw_psm2[3:]).as_matrix()

    
    def __del__(self):
        self.sim.stopSimulation()

    def get_image(self, which='stereo'):
        rgbBuffer_l, resolution_l = self.sim.getVisionSensorImg(self.l_vision_handle)
        img_data_l = np.frombuffer(rgbBuffer_l, dtype=np.uint8).reshape(resolution_l[1], resolution_l[0], 3)
        img_data_l = img_data_l[::-1, :, :]
        rgbBuffer_r, resolution_r = self.sim.getVisionSensorImg(self.r_vision_handle)
        img_data_r = np.frombuffer(rgbBuffer_r, dtype=np.uint8).reshape(resolution_r[1], resolution_r[0], 3)
        img_data_r = img_data_r[:, ::-1, :]
        if which  == 'L':
            img_data = img_data_l
        elif which  == 'R':
            img_data = img_data_r
        else:
            img_data = np.hstack((img_data_l, img_data_r))
        self.img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        return self.img_data

    def update(self):
        self.sim.step()
        self.show_image()

    def get_joints(self, which='PSM1'):
        if which == 'PSM1':
            jhs = self.PSM1_joint_handls
        if which == 'PSM2':
            jhs = self.PSM2_joint_handls

        return np.array([self.sim.getJointPosition(jh) for jh in jhs])

    def get_ee_pose(self, which='PSM1'):
        Trcm_ee = self.kin.fk(self.get_joints(which=which)[:6])
        if which == 'PSM1':
            Tw2ee = self.T_w2rcm1 @ Trcm_ee
        elif which == 'PSM2':
            Tw2ee = self.T_w2rcm2 @ Trcm_ee
        return Tw2ee

    def get_needle_pose(self):
        T_w2n = self.get_pose(self.world, self.needle)
        return T_w2n

    def pose2T(self, pose):
        T = np.eye(4)
        T[:3, -1] = pose[:3]
        T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        return T

    def get_pose(self, ref, targ):
        assert type(ref) == type(targ)
        assert type(ref) == str
        pose = self.sim.getObjectPose(self.sim.getObject(targ), self.sim.getObject(ref))
        return self.pose2T(pose)

    def set_ee_pose_direct(self, T_targ, which='PSM1'):
        """
        Set robot target pose. Reference frame of the target pose is /World
        :param T_targ: World to target pose
        :return:
        """
        if which == 'PSM1':
            T_w2rcm = self.T_w2rcm1
        elif which == 'PSM2':
            T_w2rcm = self.T_w2rcm2
        T_rcm2targ = np.linalg.inv(T_w2rcm) @ T_targ
        q_targ = self.kin.ik(T_rcm2targ)[0]
        self.set_joint(q_targ, which=which)
        self.update()

    def set_ee_pose(self, T_targ, which='PSM1'):
        """
        Set robot target pose. Reference frame of the target pose is /World
        :param T_targ: World to target pose
        :return:
        """
        if which == 'PSM1':
            T_w2rcm = self.T_w2rcm1
        elif which == 'PSM2':
            T_w2rcm = self.T_w2rcm2
        T_rcm2targ = np.linalg.inv(T_w2rcm) @ T_targ
        T_rcm2cur = np.linalg.inv(T_w2rcm) @ self.get_ee_pose(which=which)

        T_interp = self.interpolate_transform(T_rcm2cur, T_rcm2targ)
        q_cur   = kin.ik(T_rcm2cur)[0]
        q_targ  = kin.ik(T_rcm2targ)[0]
        q_interp = np.linspace(q_cur, q_targ, 30)

        for q in q_interp:
            self.set_joint(q, which=which)


    def set_joint(self, q_targ, which='PSM1'):
        if which == 'PSM1':
            jhs = self.PSM1_joint_handls
        elif which == 'PSM2':
            jhs = self.PSM2_joint_handls
        if len(q_targ) != 7:
            q_targ = np.append(q_targ, -q_targ[-1])
        for i in range(7): self.sim.setJointPosition(jhs[i], q_targ[i])
        self.update()

    def open_jaw(self, which='PSM1'):
        if which == 'PSM1':
            jhs = self.PSM1_joint_handls
        elif which == 'PSM2':
            jhs = self.PSM2_joint_handls
        q_targ = self.get_joints(which=which)
        if len(q_targ) != 7:
            q_targ = np.append(q_targ, -q_targ[-1])
        q_targ[-1] += np.deg2rad(45)
        q_targ[-2] += np.deg2rad(45)
        for i in range(7): self.sim.setJointPosition(jhs[i], q_targ[i])
        self.update()

    def close_jaw(self, which='PSM1'):
        if which == 'PSM1':
            jhs = self.PSM1_joint_handls
        elif which == 'PSM2':
            jhs = self.PSM2_joint_handls
        q_targ = self.get_joints(which=which)
        if len(q_targ) != 7:
            q_targ = np.append(q_targ, -q_targ[-1])
        q_targ[-1] -= np.deg2rad(45)
        q_targ[-2] -= np.deg2rad(45)
        for i in range(7): self.sim.setJointPosition(jhs[i], q_targ[i])
        self.update()

    def set_joint_rel(self, q_targ, which='PSM1', jaw='OPEN'):
        if which == 'PSM1':
            jhs = self.PSM1_joint_handls
        elif which == 'PSM2':
            jhs = self.PSM2_joint_handls
        if len(q_targ) != 7:
            q_targ = np.append(q_targ, -q_targ[-1])
        reached = False
        step = 0.25  # 0.1 rad = 5.7 deg
        threshold = 0.01
        if jaw == 'OPEN':
            q_targ[-1] += np.deg2rad(45)
            q_targ[-2] += np.deg2rad(45)
        elif jaw == 'CLOSE':
            q_targ[-1] -= np.deg2rad(45)
            q_targ[-2] -= np.deg2rad(45)
        while not reached:
            q_cur = self.get_joints(which=which)
            reached = ((q_targ - q_cur) < threshold).all()
            q_targ_rel = q_cur + (q_targ - q_cur) * step
            for i in range(7): self.sim.setJointPosition(jhs[i], q_targ_rel[i])
            self.update()


    def show_image(self):
        self.img_data = self.get_image()
        cv2.imshow('img_data', self.img_data)
        if cv2.waitKey(1) & 0xFF == 27:
            self.break_flag = True


# sim.startSimulation()
# print('Program started')

# from dvrk.dvrk_Kinematics import dvrkKinematics

# cv2.namedWindow('img_data', cv2.WINDOW_KEEPRATIO)
# cv2.resizeWindow('img_data', 1024, 512)

# cmd = CoppeliaCmd()
# kin = dvrkKinematics()
# running = False

# import time

# def main():
#     try:
#         while not cmd.break_flag:
#             random_angle = np.random.uniform(low=np.pi * 1 / 5, high=np.pi * 4 / 5, size=1).item()
#             random_robot = np.random.uniform(0,1)
#             if random_robot > 0.5: robot='PSM1'
#             else: robot='PSM2'
#             # print(f'Random Needle Grasping Angle: {robot}, {random_angle}')

#             if cmd.grasp_flag:
#                 print('Needle Grasping')
#                 T_w2n = cmd.get_needle_pose()
#                 T_n2gr = np.eye(4)
#                 T_n2gr[:2, -1] = 0.012 * np.array([np.cos(random_angle), np.sin(random_angle)])
#                 if T_w2n[2, 2] > 0:
#                     T_n2gr[:3, :3] = R.from_euler('Z', random_angle).as_matrix()
#                     T_n2gr[2, -1] += 0.005
#                 else:
#                     T_n2gr[:3, :3] = R.from_euler('ZY', [random_angle, np.pi]).as_matrix()
#                     T_n2gr[2, -1] -= 0.005
#                 T_w2gr = T_w2n@T_n2gr
#                 cmd.set_ee_pose(T_w2gr, which=robot)

#                 T_gr2g = np.eye(4)
#                 T_gr2g[2, -1] -= 0.005
#                 T_w2g = T_w2gr @ T_gr2g
#                 cmd.set_ee_pose(T_w2g, which=robot)
#                 sim.setObjectParent(sim.getObject(cmd.needle), sim.getObject('/' + robot+'_ee'))
#                 sim.setObjectInt32Param(sim.getObject(cmd.needle), sim.shapeintparam_static, 1)
#                 sim.step()
#                 time.sleep(1)

#                 T_g2u = np.eye(4)
#                 T_g2u[2, -1] += 0.01
#                 T_w2u = T_w2g @ T_g2u
#                 cmd.set_ee_pose(T_w2u @ T_g2u, which=robot)

#                 cmd.break_flag = True

#                 # needle 잡으러 내려가야함
#                 # needle 잡으면 child 관계 만들어서 강제 부착
#     except Exception as e:
#         import traceback
#         print(e)
#         traceback.print_exc()
#         sim.setObjectInt32Param(sim.getObject(cmd.needle), sim.shapeintparam_static, 0)
#         sim.step()
#         sim.stopSimulation()
#     sim.setObjectInt32Param(sim.getObject(cmd.needle), sim.shapeintparam_static, 0)
#     sim.step()
#     sim.stopSimulation()

# if __name__ == '__main__':
#     main()