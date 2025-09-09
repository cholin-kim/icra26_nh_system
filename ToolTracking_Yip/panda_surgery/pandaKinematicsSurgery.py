from panda.pandaKinematics import pandaKinematics
from panda_surgery.pandaKinematicsRCM import pandaKinematicsRCM
import panda_surgery.pandaVarSurgery as pandaVar
from scipy.spatial.transform import Rotation
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=np.inf)
import time


class pandaKinematicsSurgery:
    @classmethod
    def DH_transform(cls, dhparams):  # stacks transforms of neighbor frame, following the modified DH convention
        Ts = [np.array([[np.cos(theta), -np.sin(theta), 0, a],
                        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                         -np.sin(alpha) * d],
                        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]]) for [alpha, a, d, theta] in dhparams]
        return Ts

    @classmethod
    def fk(cls, joints, theta_offset=0.0, n_module=16, vars=pandaVar):
        """
        joints = (9,)
        Ts = (Tb1, T12, T23, ...)
        """
        q1, q2, q3, q4, q5, q6, q7, q8, q9 = joints
        L1, L2, L3, L4, L5, L6, L7, dj, offset, t, h = vars.params

        # Panda arm
        Ts = pandaKinematicsSurgery.DH_transform(pandaVar.dhparam_surgery(joints[:7], theta_offset=theta_offset))     # Tb1, T12, T23, ...

        # Elastic Segment
        trans_q8 = t+h
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

        # Panda manipulator
        J = np.zeros((6, 7))
        for i in range(7):
            Zi = Tbs[i][:3, 2]  # vector of actuation axis
            J[3:, i] = Zi  # Jw
            Pin = (Tbe[:3, -1] - Tbs[i][:3, -1])  # pos vector from (i) to (n)
            J[:3, i] = np.cross(Zi, Pin)  # Jv

        # Elastic Segment
        C=0
        Jw8 = np.array([0.0, 0.0, 0.0])
        Jv8 = np.array([0.0, 0.0, 0.0])
        Tbi = Tbs[10]
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
    def ik(cls, Tb_ed, theta_offset=0.0, n_module=16, vars=pandaVar, q0=None, RRMC=False, k=0.5):     # inverse kinematics using Newton-Raphson Method
        """
        This is IK for 10 DoF panda with surgical gripper.
        Tb_ed = transform from base to desired end effector
        q0 = initial configuration for iterative N-R method
        RRMC = Resolved-rate Motion Control
        k = step size (scaling) of cartesian error
        """
        if q0 is None:
            q0 = []
        assert Tb_ed.shape == (4, 4)
        # st = time.time()
        if q0 == []:
            qk = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.0, 0.0])  # initial guess
        else:
            qk = np.array(q0)
        iter = 0
        reached = False
        while not reached:
            # Define Cartesian error
            result_fk = pandaKinematicsSurgery.fk(joints=qk, theta_offset=theta_offset, n_module=n_module, vars=vars)
            Tbs, Ts, Ts_seg = result_fk
            Tb_ec = Tbs[-1]  # base to current ee
            Tec_ed = np.linalg.inv(Tb_ec).dot(Tb_ed)     #   transform from current ee to desired ee
            pos_err = Tb_ec[:3, :3].dot(Tec_ed[:3, -1])     # pos err in the base frame
            # rot_err = Tb_ec[:3, :3].dot(Rotation.from_dcm(Tec_ed[:3, :3]).as_rotvec())  # rot err in the base frame
            rot_err = Tb_ec[:3, :3].dot(Rotation.from_matrix(Tec_ed[:3, :3]).as_rotvec())  # rot err in the base frame
            err_cart = np.concatenate((pos_err, rot_err))

            # Inverse differential kinematics (Newton-Raphson method)
            J = pandaKinematicsSurgery.jacobian(result_fk=result_fk, n_module=n_module)
            Jp = np.linalg.pinv(J)
            qk_next = qk + Jp.dot(err_cart*k)
            qk = qk_next

            # Convergence condition
            if np.linalg.norm(err_cart) < 10e-4:
                reached = True
            else:
                iter += 1
            if RRMC:
                reached = True

        # print ("iter=", iter, "time=", time.time() - st)
        assert ~np.isnan(qk).any()
        return qk

    @classmethod
    def ik_RCM(cls, Tb_rcm, Tb_ed=None, T_rcm_ed=None, theta_offset=0.0, n_module=16, vars=pandaVar, q0=None, RRMC=False, k=0.5):     # inverse kinematics using Newton-Raphson Method
        """
        This is IK for surgical Panda with RCM constraints.
        Only one input can be given among Tb_ed or T_rcm_ed.

        Tb_ed = transform from base to the desired end effector
        Trcm_ed = transform from RCM to the desired end effector
        Tb_rcm = transform from base to RCM (given)
        q0, q0_RCM = initial configuration for iterative N-R method
        RRMC = Resolved-rate Motion Control
        k = step size (scaling) of cartesian error
        """
        assert ((Tb_ed is not None) or (T_rcm_ed is not None))
        assert not ((Tb_ed is not None) and (T_rcm_ed is not None))

        if Tb_ed is not None:
            T_rcm_ed = np.linalg.inv(Tb_rcm) @ Tb_ed

        # Solve IK for RCM
        # q0_RCM = pandaKinematicsRCM.ik_analytic_assumption(Tb_ed=T_rcm_ed)  # get the course ik solution
        qk_RCM = pandaKinematicsRCM.ik_analytic_assumption(Tb_ed=T_rcm_ed)  # get the course ik solution
        # qk_RCM = pandaKinematicsRCM.ik(Tb_ed=T_rcm_ed, n_module=n_module, vars=vars, q0=q0_RCM, RRMC=RRMC, k=k)     # refine the ik solution

        # Obtain transform from base to flange
        Ts = pandaKinematicsRCM.DH_transform(pandaVar.dhparam_RCM(joints=qk_RCM[:4]))
        T_rcm_gb = Ts[0].dot(Ts[1]).dot(Ts[2]).dot(Ts[3]).dot(Ts[4])
        Tf_gb = np.identity(4)
        Tf_gb[:3, :3] = Rotation.from_euler('z', theta_offset).as_matrix()
        Tb_fd = Tb_rcm.dot(T_rcm_gb).dot(np.linalg.inv(Tf_gb))

        # Solve IK for Panda Arm
        if q0 is None:
            q0 = [0.0, -1.0, -0.0, -2.5, -0.0, 1.57, 0.0, 0.0, 0.0]
        qk_arm = pandaKinematics.ik_analytic(Tb_ed=Tb_fd, q_curr=q0[:7], q7=qk_RCM[3])
        qk = np.r_[qk_arm, qk_RCM[4:]]
        # print (qk_arm, qk_RCM)
        assert ~np.isnan(qk).any()
        return qk


if __name__ == "__main__":
    from panda_surgery.ros.pandaRvizSurgery import pandaRvizSurgery
    from panda_surgery.ros.pandaRvizMotionSurgery import pandaRvizMotionSurgery
    panda = pandaRvizSurgery(is_core=False, use_GUI=False)
    motion = pandaRvizMotionSurgery(ns='panda1', q0=None, Tb_rcm=None)
    while True:
        q_des = (np.random.rand(9) - 0.5) * np.pi
        # q_des = [0.0]*9
        # q_des[7] = np.pi/2
        print("q_des=", q_des)
        motion.set_joint_position_direct(joints=np.append(q_des, 0.0))

        # FK
        Tbe_des = pandaKinematicsSurgery.fk(joints=q_des[:9], n_module=16, theta_offset=0)[0][-1]
        pos_des = Tbe_des[:3, -1]
        quat_des = Rotation.from_matrix(Tbe_des[:3, :3]).as_quat()
        print("pose_des=", pos_des, quat_des)
        input()

        # IK
        q_ik = pandaKinematicsSurgery.ik(Tb_ed=Tbe_des, n_module=16, vars=pandaVar, q0=None, RRMC=False, k=0.5)
        # motion.set_joint_position_direct(joints=np.append(q_ik, 0.0))
        print("q_ik =", q_ik)

        # FK to verify the IK
        Tbe_ik = pandaKinematicsSurgery.fk(joints=q_ik)[0][-1]
        pos_ik = Tbe_ik[:3, -1]
        quat_ik = Rotation.from_matrix(Tbe_ik[:3, :3]).as_quat()
        print("pose_ik=", pos_ik, quat_ik)
        print("======================================================================")
