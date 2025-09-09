import panda_surgery.pandaVarSurgery as pandaVar
from scipy.spatial.transform import Rotation
import scipy.misc
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=np.inf)
import time


class pandaKinematicsRCM:
    @classmethod
    def DH_transform(cls, dhparams):  # stacks transforms of neighbor frame, following the modified DH convention
        Ts = [np.array([[np.cos(theta), -np.sin(theta), 0, a],
                        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                         -np.sin(alpha) * d],
                        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]]) for [alpha, a, d, theta] in dhparams]
        return Ts

    @classmethod
    def fk(cls, joints, n_module=16, vars=pandaVar):
        """
        joints = (6,)
        Ts = (Tb1, T12, T23, ...)
        """
        qy, qp, qt, qr, q8, q9 = joints
        L1, L2, L3, L4, L5, L6, L7, dj, offset, t, h = vars.params

        Ts = pandaKinematicsRCM.DH_transform(pandaVar.dhparam_RCM(joints[:4]))

        # Elastic Segment
        trans_q8 = t + h
        T_m1 = np.array([[np.cos(q8 / (2 * n_module)), -np.sin(q8 / (2 * n_module)), 0, 0],
                         [np.sin(q8 / (2 * n_module)), np.cos(q8 / (2 * n_module)), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T_m2 = np.array([[1, 0, 0, trans_q8],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T_m3 = np.array([[np.cos(q8 / (2 * n_module)), -np.sin(q8 / (2 * n_module)), 0, 0],
                         [np.sin(q8 / (2 * n_module)), np.cos(q8 / (2 * n_module)), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T_m = T_m1 @ T_m2 @ T_m3
        T_seg = np.identity(4)
        for i in range(n_module):
            T_seg = T_seg @ T_m
        Ts.append(T_seg)

        Ts_seg = []
        Ts_seg.append(T_m1)
        Ts_seg.append(T_m2)
        Ts_seg.append(T_m3)

        # Jaw (yaw)
        T_yaw = np.array([[np.sin(q9), np.cos(q9), 0, L7],
                          [0, 0, 1, 0],
                          [np.cos(q9), -np.sin(q9), 0, 0],
                          [0, 0, 0, 1]])
        T_ee = np.array([[1, 0, 0, 0],
                         [0, 0, 1, dj],
                         [0, -1, 0, 0],
                         [0, 0, 0, 1]])
        Ts.append(T_yaw)
        Ts.append(T_ee)

        # Calculate forward kin.
        # Never use multi_dot for speed-up. It slows down the computation A LOT! Use "for-loop" instead.
        Tbi = np.eye(4)
        Tbs = []
        for T in Ts:
            # Tbe = Tbe @ T
            Tbi = Tbi.dot(T)
            Tbs.append(Tbi)

        # # Tbe = np.linalg.multi_dot(Ts)   # from base to end-effector
        # Tbs = np.array([np.linalg.multi_dot(Ts[:i]) if i > 1 else Ts[0] for i in range(1, len(Ts)+1)])  # Tb1, Tb2, Tb3, ...
        # # Tbs[-1]: from base to end effector
        return Tbs, Ts, Ts_seg

    @classmethod
    def jacobian(cls, result_fk, n_module=16):
        """
        Tbs: Tb0, Tb1, Tb2, ...
        """
        Tbs, Ts, Ts_seg = result_fk
        Tbe = Tbs[-1]

        J = np.zeros((6, 4))
        for i in range(4):
            Zi = Tbs[i][:3, 2]  # vector of actuation axis
            if i == 2:  # for prismatic
                J[3:, i] = np.zeros(3)  # Jw
                J[:3, i] = Zi  # Jv
            else:
                J[3:, i] = Zi  # Jw
                Pin = (Tbe[:3, -1] - Tbs[i][:3, -1])  # pos vector from (i) to (n)
                J[:3, i] = np.cross(Zi, Pin)  # Jv

        # Elastic Segment
        C=0
        Jw8 = np.array([0.0, 0.0, 0.0])
        Jv8 = np.array([0.0, 0.0, 0.0])
        Tbi = Tbs[6]
        for i in range(n_module):
            # rotate by q5/2
            Tbi = Tbi @ Ts_seg[0]
            Jw8i = Tbi[:3, 2] / n_module / 2
            Jw8 = Jw8 + Jw8i
            Jv8 = Jv8 + np.cross(Jw8i, (Tbe[:3, -1] - Tbi[:3, -1]))

            # translate by trans_q5
            Tbi = Tbi @ Ts_seg[1]
            Jv8 = Jv8 + Tbi[:3, 0] * C

            # rotate by q5/2
            Tbi = Tbi @ Ts_seg[2]
            Jw8i = Tbi[:3, 2] / n_module / 2
            Jw8 = Jw8 + Jw8i
            Jv8 = Jv8 + np.cross(Jw8i, (Tbe[:3, -1] - Tbi[:3, -1]))

        Tbi = Tbs[-2]
        Jw9 = Tbi[:3, 2]
        Jv9 = np.cross(Jw9, Tbe[:3, -1] - Tbi[:3, -1])

        # Compose a jacobian matrix
        Jv = np.c_[J[:3], Jv8, Jv9]
        Jw = np.c_[J[3:], Jw8, Jw9]
        J = np.r_[Jv, Jw]
        return J

    @classmethod
    def ik_analytic_assumption(cls, Tb_ed):
        # Analytic solution, assuming the continuum joint as hinge joint.
        # The hinge joint is assumed to located at the middle point of the continuum joint.
        T = np.linalg.inv(Tb_ed)
        L3 = 16*(pandaVar.t+pandaVar.h)/2 + pandaVar.L7     # pitch ~ yaw (m)
        L4 = pandaVar.dj    # yaw ~ tip (m)
        x74 = T[0, 3]
        y74 = T[1, 3]
        z74 = T[2, 3]
        q6 = np.arctan2(-x74, -z74 - L4)
        temp = -L3 + np.sqrt(x74 ** 2 + (z74 + L4) ** 2)
        q3 = -L3/2 + np.sqrt(y74 ** 2 + temp ** 2)
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

    @classmethod
    def ik(cls, Tb_ed, n_module=16, vars=pandaVar, q0=None, RRMC=False, k=0.001):     # inverse kinematics using Newton-Raphson Method
        """
        Tb_ed = transform from base to desired end effector
        q0 = initial configuration for iterative N-R method
        RRMC = Resolved-rate Motion Control
        k = step size (scaling) of cartesian error
        """
        if q0 is None:
            q0 = []
        assert Tb_ed.shape == (4, 4)
        st = time.time()
        if q0 == []:
            qk = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])  # initial guess
        else:
            qk = np.array(q0)
        iter = 0
        reached = False
        while not reached:
            # Define Cartesian error
            result_fk = pandaKinematicsRCM.fk(joints=qk, n_module=n_module, vars=vars)
            Tbs, Ts, Ts_seg = result_fk
            Tb_ec = Tbs[-1]  # base to current ee
            Tec_ed = np.linalg.inv(Tb_ec).dot(Tb_ed)     #   transform from current ee to desired ee
            pos_err = Tb_ec[:3, :3].dot(Tec_ed[:3, -1])     # pos err in the base frame
            # rot_err = Tb_ec[:3, :3].dot(Rotation.from_dcm(Tec_ed[:3, :3]).as_rotvec())  # rot err in the base frame
            rot_err = Tb_ec[:3, :3].dot(Rotation.from_matrix(Tec_ed[:3, :3]).as_rotvec())  # rot err in the base frame
            err_cart = np.concatenate((pos_err, rot_err))

            # Inverse differential kinematics (Newton-Raphson method)
            J = pandaKinematicsRCM.jacobian(result_fk=result_fk, n_module=n_module)
            Jp = np.linalg.pinv(J)
            qk_next = qk + Jp.dot(err_cart*k)
            qk = qk_next

            # Convergence condition
            if np.linalg.norm(err_cart) < 10e-5:
                reached = True
            else:
                iter += 1
            if RRMC:
                reached = True

        # print ("iter=", iter, "time=", time.time() - st)
        assert ~np.isnan(qk).any()
        return qk


if __name__ == "__main__":
    L1, L2, L3, L4, L5, L6, L7, dj, offset, t, h = pandaVar.params
    while True:
        q_des = (np.random.rand(6)-0.5) * 2 * np.pi/2.5     # if either yaw or pitch angle approaches to 90(deg), the ik solution can diverges.
        q_des[2] = np.random.rand(1) + 0.05
        print("q_des=", q_des)

        # FK
        Tbe_des = pandaKinematicsRCM.fk(joints=q_des)[0][-1]
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
        q_ik = pandaKinematicsRCM.ik(Tb_ed=Tbe_des, n_module=16, vars=pandaVar, q0=None, RRMC=False, k=0.5)
        q_ik_ = np.array(pandaKinematicsRCM.ik_analytic_assumption(Tb_ed=Tbe_des))
        print("q_ik =", q_ik)
        print("error_numerical =", np.abs(q_ik-q_des))
        print("")
        print("q_ik_=", q_ik_)
        print("error_analytic_assumption =", np.abs(q_ik_-q_des))
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
