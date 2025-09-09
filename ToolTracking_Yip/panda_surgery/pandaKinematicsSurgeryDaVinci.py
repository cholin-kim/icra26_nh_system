from panda.pandaKinematics import pandaKinematics
from panda_surgery.pandaKinematicsRCMDaVinci import pandaKinematicsRCMDavinci
import panda_surgery.pandaVarSurgeryDaVinci as pandaVar
from scipy.spatial.transform import Rotation
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=np.inf)
import time


class pandaKinematicsSurgeryDavinci:
    @classmethod
    def DH_transform(cls, dhparams):  # stacks transforms of neighbor frame, following the modified DH convention
        Ts = [np.array([[np.cos(theta), -np.sin(theta), 0, a],
                        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                         -np.sin(alpha) * d],
                        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]]) for [alpha, a, d, theta] in dhparams]
        return Ts

    @classmethod
    def fk(cls, joints):
        """
        joints = (10,)
        Ts = (Tb1, T12, T23, ...)
        """
        # Calculate forward kin.
        Ts = pandaKinematicsSurgeryDavinci.DH_transform(pandaVar.dhparam(joints))     # Tb1, T12, T23, ...
        Tbi = np.eye(4)
        Tbs = []
        for T in Ts:
            Tbi = Tbi.dot(T)
            Tbs.append(Tbi)
        return Tbs, Ts


    @classmethod
    def ik_RCM(cls, Tb_rcm, Tb_ed=None, T_rcm_ed=None, q_curr=None):
        """
        This is IK for 10 DoF surgical Panda with RCM constraints.
        Only one input can be given among Tb_ed or T_rcm_ed.

        Tb_ed = transform from base to the desired end effector
        Trcm_ed = transform from RCM to the desired end effector
        Tb_rcm = transform from base to RCM (given)
        """
        assert ((Tb_ed is not None) or (T_rcm_ed is not None))
        assert not ((Tb_ed is not None) and (T_rcm_ed is not None))

        if Tb_ed is not None:
            T_rcm_ed = np.linalg.inv(Tb_rcm).dot(Tb_ed)

        # Solve IK of RCM kinematics
        qk_RCM = pandaKinematicsRCMDavinci.ik(Tb_ed=T_rcm_ed)

        # Obtain transform from base to flange
        Ts = pandaKinematicsRCMDavinci.DH_transform(pandaVar.dhparam_RCM(joints=qk_RCM))
        T_rcm_t = Ts[0].dot(Ts[1]).dot(Ts[2])
        T_t_GB = pandaKinematicsRCMDavinci.DH_transform([[0, 0, -pandaVar.L6, np.pi]])[0]
        ###################################
        Tf_gb = np.identity(4)
        ###################################
        Tb_fd = Tb_rcm.dot(T_rcm_t).dot(T_t_GB).dot(np.linalg.inv(Tf_gb))

        # Solve IK for Panda Arm
        if q_curr is None:
            q_curr = [0.0, -1.0, -0.0, -2.5, -0.0, 1.57, 0.0, 0.0, 0.0]
        qk_arm = pandaKinematics.ik_analytic(Tb_ed=Tb_fd, q_curr=q_curr[:7], q7=None)
        qk = np.r_[qk_arm, qk_RCM[3:]]
        assert ~np.isnan(qk).any()
        return qk


if __name__ == "__main__":
    import time
    xyz = "0 0 0"
    rpy = "0 0 0"

    xyz1 = "0 0 0"
    rpy1 = "0 0 0"
    base_transform = [xyz, rpy, xyz1, rpy1]

    Tb_rcm = np.identity(4)
    Tb_rcm[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
    Tb_rcm[:3, -1] = np.array([0.35, 0.0, 0.2])


    # # Verify FK
    # while True:
    #     q_des = (np.random.rand(11) - 0.5) * np.pi
    #     q_des[-1] = q_des[-2]
    #     q_des = [0.0]*11
    #     print("q_des=", q_des)
    #     motion.set_joint_position_direct(joints=q_des)
    #
    #     # FK
    #     Tbe_des = pandaKinematicsSurgeryDavinci.fk(joints=q_des[:10])[0][-1]
    #     pos_des = Tbe_des[:3, -1]
    #     quat_des = Rotation.from_matrix(Tbe_des[:3, :3]).as_quat()
    #     print("pose_des=", pos_des, quat_des)
    #
    #     # RCM FK
    #     q_des_RCM = [0.0]*6
    #     Trcm_ed = pandaKinematicsRCMDavinci.fk(joints=q_des_RCM)[0][-1]
    #     pos_des = Trcm_ed[:3, -1]
    #     quat_des = Rotation.from_matrix(Trcm_ed[:3, :3]).as_quat()
    #     print("pose_RCM_des=", pos_des, quat_des)
    #     input()

    # Verify RCM Kinematics
    while True:
        # RCM FK
        q_des_RCM_min = np.array([-np.pi/5, -np.pi/5, 0.2, -np.pi/2, -np.pi/2, -np.pi/2])
        q_des_RCM_max = np.array([np.pi/5, np.pi/5, 0.3, np.pi/2, np.pi/2, np.pi/2])
        q_des_RCM = np.random.uniform(low=q_des_RCM_min, high=q_des_RCM_max, size=6)
        print ("q_des_RCM=", q_des_RCM)
        Trcm_ed = pandaKinematicsRCMDavinci.fk(joints=q_des_RCM)[0][-1]
        print ("Trcm_ed=", Trcm_ed)

        # print (Rotation.from_euler(seq='y', angles=np.pi).as_matrix())
        #
        # IK
        # Define rob base to RCM transform
        q_ik = pandaKinematicsSurgeryDavinci.ik_RCM(Tb_rcm=Tb_rcm, Tb_ed=None, T_rcm_ed=Trcm_ed)
        # motion.set_joint_position_direct(joints=np.append(q_ik, 0.0))
        print("q_ik =", q_ik)

        # FK to verify the IK
        Tbe_ik = pandaKinematicsSurgeryDavinci.fk(joints=q_ik)[0][-1]
        Trcm_ed_ik = np.linalg.inv(Tb_rcm).dot(Tbe_ik)
        print("Trcm_ed_ik =", Trcm_ed_ik)
        print("error=", np.linalg.inv(Trcm_ed).dot(Trcm_ed_ik))
        print("======================================================================")
        input()
