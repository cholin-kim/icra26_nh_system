import panda_surgery.pandaVarSurgeryDaVinci as pandaVar
from scipy.spatial.transform import Rotation
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=np.inf)


class pandaKinematicsRCMDavinci:
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
        joints = (6,)
        Ts = (Tb1, T12, T23, ...)
        """
        # Calculate forward kin.
        Ts = pandaKinematicsRCMDavinci.DH_transform(pandaVar.dhparam_RCM(joints))
        Tbi = np.eye(4)
        Tbs = []
        for T in Ts:
            Tbi = Tbi.dot(T)
            Tbs.append(Tbi)
        return Tbs, Ts

    @classmethod
    def ik(cls, Tb_ed):
        T = np.linalg.inv(Tb_ed)
        L3 = pandaVar.L7     # pitch ~ yaw (m)
        L4 = pandaVar.dj    # yaw ~ tip (m)

        x74 = T[0, 3]
        y74 = T[1, 3]
        z74 = T[2, 3]
        q6 = np.arctan2(-x74, -z74 - L4)
        temp = -L3 + np.sqrt(x74 ** 2 + (z74 + L4) ** 2)
        q3 = 0 + np.sqrt(y74 ** 2 + temp ** 2)
        q5 = np.arctan2(-y74, temp)
        R74 = np.array([[-np.sin(q5) * np.sin(q6), np.cos(q6), -np.cos(q5) * np.sin(q6)],
                        [np.cos(q5), 0, -np.sin(q5)],
                        [-np.cos(q6) * np.sin(q5), -np.sin(q6), -np.cos(q5) * np.cos(q6)]])
        R70 = T[:3, :3]
        R40 = R74.T.dot(R70)
        n32 = R40[2, 1]
        n31 = R40[2, 0]
        n33 = R40[2, 2]
        n22 = R40[1, 1]
        n12 = R40[0, 1]
        q2 = np.arcsin(n32)
        q1 = np.arctan2(-n31, n33)
        q4 = np.arctan2(n22, n12)
        joint = [q1, q2, q3, q4, q5, q6]
        assert ~np.isnan(joint).any()
        return joint


if __name__ == "__main__":
    # L1, L2, L3, L4, L5, L6, L7, dj, offset, t, h = pandaVar.params
    while True:
        q_des = (np.random.rand(6)-0.5) * 2 * np.pi/2.5     # if either yaw or pitch angle approaches to 90(deg), the ik solution can diverges.
        q_des[2] = np.random.rand(1) + 0.05
        print("q_des=", q_des)

        # FK
        Tbe_des = pandaKinematicsRCMDavinci.fk(joints=q_des)[0][-1]
        pos_des = Tbe_des[:3, -1]
        quat_des = Rotation.from_matrix(Tbe_des[:3, :3]).as_quat()
        print("pose_des=", pos_des, quat_des)
        print ("")

        # Tbe_des = np.identity(4)
        # x_180 = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        # z_180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        # Tbe_des[:3, :3] = x_180 @ z_180
        # Tbe_des[:3, -1] = [0.0, 0.0, -0.03]

        # IK
        q_ik = np.array(pandaKinematicsRCMDavinci.ik(Tb_ed=Tbe_des))
        print("q_ik =", q_ik)
        print("error =", np.abs(q_ik-q_des))
        input()

        # # FK to verify the IK
        # Tbe_ik = pandaKinematicsRCM.fk(joints=q_ik)[0][-1]
        # pos_ik = Tbe_ik[:3, -1]
        # quat_ik = Rotation.from_matrix(Tbe_ik[:3, :3]).as_quat()
        # print("pose_ik=", pos_ik, quat_ik)
        # print("======================================================================")
        # input()
        # print (q_des - q_ik)
        # if np.linalg.norm(q_des - q_ik) > 0.01:
        #     print ("detected")
        #     print (np.rad2deg(q_des))
        #     print (q_des[2])
        #     print (np.rad2deg(q_ik))
        #     print (q_ik[2])
        #     import pdb; pdb.set_trace()
