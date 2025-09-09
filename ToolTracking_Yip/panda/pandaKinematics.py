import panda.pandaVar as pandaVar
from scipy.spatial.transform import Rotation
import scipy.misc
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import time
from scipy.optimize import minimize_scalar


class pandaKinematics:
    # @classmethod
    # def DH_transform(cls, dh_params):  # stacks transforms of neighbor frame, following the modified DH convention
    #     """
    #     dh_params: N_conf x N_dh x 4
    #     N_conf = # of configurations
    #     N_dh = # of coordinate transforms per DH-param group (9 for Panda: T01, ..., T89)
    #     4 = (alpha, a, d, theta)
    #     """
    #     N_conf, N_dh, _ = np.shape(dh_params)
    #     alpha, a, d, theta = dh_params.reshape(-1, 4).T
    #     Ts = np.zeros((N_conf, N_dh, 4, 4))
    #     Ts[:, :, 0, 0] = np.cos(theta)
    #     Ts[:, :, 0, 1] = -np.sin(theta)
    #     Ts[:, :, 0, 2] = 0.0
    #     Ts[:, :, 0, 3] = a
    #     Ts[:, :, 1, 0] = np.sin(theta) * np.cos(alpha)
    #     Ts[:, :, 1, 1] = np.cos(theta) * np.cos(alpha)
    #     Ts[:, :, 1, 2] = -np.sin(alpha)
    #     Ts[:, :, 1, 3] = -np.sin(alpha) * d
    #     Ts[:, :, 2, 0] = np.sin(theta) * np.sin(alpha)
    #     Ts[:, :, 2, 1] = np.cos(theta) * np.sin(alpha)
    #     Ts[:, :, 2, 2] = np.cos(alpha)
    #     Ts[:, :, 2, 3] = np.cos(alpha) * d
    #     Ts[:, :, 3, 0] = 0.0
    #     Ts[:, :, 3, 1] = 0.0
    #     Ts[:, :, 3, 2] = 0.0
    #     Ts[:, :, 3, 3] = 1.0
    #     return Ts

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
        joints = (7,)
        Ts = (Tb1, T12, T23, ...)
        """
        Ts = pandaKinematics.DH_transform(pandaVar.dhparam(joints))     # Tb1, T12, T23, ...
        # Tbe = np.linalg.multi_dot(Ts)   # from base to end-effector
        Tbs = np.array([np.linalg.multi_dot(Ts[:i]) if i > 1 else Ts[0] for i in range(1, len(Ts)+1)])  # Tb1, Tb2, Tb3, ...
        # Tbs[-1]: from base to end effector
        return Tbs, Ts

    @classmethod
    def jacobian(cls, Tbs):
        """
        Tbs: Tb0, Tb1, Tb2, ...
        """
        Tbe = Tbs[-1]
        J = np.zeros((6, 7))
        for i in range(7):
            Zi = Tbs[i, :3, 2]  # vector of actuation axis
            J[3:, i] = Zi  # Jw
            Pin = (Tbe[:3, -1] - Tbs[i, :3, -1])  # pos vector from (i) to (n)
            J[:3, i] = np.cross(Zi, Pin)  # Jv
        return J

    @classmethod
    def ik(cls, Tb_ed, q0=None, RRMC=False, k=0.5):     # inverse kinematics using Newton-Raphson Method
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
            qk = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0])  # initial guess
        else:
            qk = np.array(q0)
        iter = 0
        reached = False
        while not reached:
            Tbs = pandaKinematics.fk(joints=qk)[0]

            # Define Cartesian error
            Tb_ec = Tbs[-1]  # base to current ee
            Tec_ed = np.linalg.inv(Tb_ec).dot(Tb_ed)     #   transform from current ee to desired ee
            pos_err = Tb_ec[:3, :3].dot(Tec_ed[:3, -1])     # pos err in the base frame
            # rot_err = Tb_ec[:3, :3].dot(Rotation.from_dcm(Tec_ed[:3, :3]).as_rotvec())  # rot err in the base frame
            rot_err = Tb_ec[:3, :3].dot(Rotation.from_matrix(Tec_ed[:3, :3]).as_rotvec())  # rot err in the base frame
            err_cart = np.concatenate((pos_err, rot_err))

            # Inverse differential kinematics (Newton-Raphson method)
            J = pandaKinematics.jacobian(Tbs)
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
    # "Complete" Inverse Kinematics
    # Compute four candidates of solutions, and select the final one that is the closest to the q_curr
    def ik_analytic_complete(cls, Tb_ed, q_curr, q7):
        qs = np.empty((4, 7))   # Four combinations of solution candidates in total
        qs[:] = np.nan

        if q7 <= pandaVar.q_min[6] or q7 >= pandaVar.q_max[6]:
            qs[:, 6] = np.nan
        else:
            qs[:, 6] = q7

        R_be = Tb_ed[:3, :3]
        z_be = R_be[:3, 2]
        p_be = Tb_ed[:3, -1]

        # Compute q4 (unique feasible solution)
        p_b7 = p_be - pandaVar.L5*z_be
        x_e6 = np.array([np.cos(q7), -np.sin(q7), 0.0])
        x_b6 = R_be @ x_e6
        p_b6 = p_b7 - pandaVar.L4*x_b6

        p_b2 = np.array([0.0, 0.0, pandaVar.L1])
        v_26 = p_b6 - p_b2
        L26 = np.linalg.norm(v_26)
        L24 = np.sqrt(pandaVar.offset**2 + pandaVar.L2**2)
        L46 = np.sqrt(pandaVar.offset**2 + pandaVar.L3**2)
        if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
            print("L24=", L24)
            print("L46=", L46)
            print("L26=", L26)
            raise ValueError

        ang_H46 = np.arctan(pandaVar.L3/pandaVar.offset)
        ang_342 = np.arctan(pandaVar.L2/pandaVar.offset)
        ang_46H = np.arctan(pandaVar.offset/pandaVar.L3)
        ang_246 = np.arccos((L24**2 + L46**2 - L26**2)/(2.0*L24*L46))
        q4 = ang_246 + ang_H46 + ang_342 - 2.0*np.pi
        if q4 <= pandaVar.q_min[3] or q4 >= pandaVar.q_max[3]:
            qs[:, 3] = np.nan
        else:
            qs[:, 3] = q4

        # Compute q6 (two solutions)
        ang_462 = np.arccos((L26**2 + L46**2 - L24**2)/(2.0*L26*L46))
        ang_26H = ang_46H + ang_462

        z_b6 = np.cross(z_be, x_b6)
        y_b6 = np.cross(z_b6, x_b6)
        R_b6 = np.c_[x_b6, y_b6, z_b6]
        v_6_62 = R_b6.T @ (-v_26)

        phi6 = np.arctan2(v_6_62[1], v_6_62[0])
        psi6 = np.arcsin((-L26*np.cos(ang_26H))/np.sqrt(v_6_62[0]**2 + v_6_62[1]**2))

        # Two candidates
        q6_c1 = np.pi - psi6 - phi6
        q6_c2 = psi6 - phi6
        for i, q6 in enumerate([q6_c1, q6_c2]):
            if q6 <= pandaVar.q_min[5]:
                q6 += 2.0*np.pi
            elif q6 >= pandaVar.q_max[5]:
                q6 -= 2.0*np.pi
            if q6 <= pandaVar.q_min[5] or q6 >= pandaVar.q_max[5]:
                qs[2*i, 5] = np.nan
                qs[2*i+1, 5] = np.nan
            else:
                qs[2*i, 5] = q6
                qs[2*i+1, 5] = q6

        # Compute q1 & q2
        ang_324 = np.pi/2 - ang_342
        ang_426 = np.pi - ang_246 - ang_462
        ang_P26 = ang_324 + ang_426
        ang_2P6 = np.pi - ang_P26 - ang_26H
        LP6 = L26 * np.sin(ang_P26)/np.sin(ang_2P6)

        z_b5_set = np.zeros((4, 3))
        v_2P_set = np.zeros((4, 3))
        for i, q6 in enumerate([q6_c1, q6_c2]):
            z_65 = np.array([np.sin(q6), np.cos(q6), 0.0])
            z_b5 = R_b6 @ z_65
            v_2P = (p_b6 - p_b2) - LP6*z_b5

            z_b5_set[2*i] = z_b5
            z_b5_set[2*i + 1] = z_b5
            v_2P_set[2*i] = v_2P
            v_2P_set[2*i + 1] = v_2P
            L2P = np.linalg.norm(v_2P)

            if np.abs(v_2P[2]/L2P) > 0.99999:     # Singularity by q2 = 0. In this case, q1 and q3 has infinite solutions.
                print ("Singularity!")
                qs[2*i, 0] = q_curr[0]  # q1 gets the same value with q0(reference)
                qs[2*i, 1] = 0.0
                qs[2*i + 1, 0] = q_curr[0]
                qs[2*i + 1, 1] = 0.0
            else:
                qs[2*i, 0] = np.arctan2(v_2P[1], v_2P[0])
                qs[2*i, 1] = np.arccos(v_2P[2]/L2P)
                if qs[2*i, 0] < 0:
                    qs[2*i+1, 0] = qs[2*i, 0] + np.pi
                else:
                    qs[2*i+1, 0] = qs[2*i, 0] - np.pi
                qs[2*i+1, 1] = -qs[2*i, 1]

        for i in range(4):
            if qs[i, 0] <= pandaVar.q_min[0] or qs[i, 0] >= pandaVar.q_max[0] or qs[i, 1] <= pandaVar.q_min[1] or qs[i, 1] >= pandaVar.q_max[1]:
                qs[i, 0] = np.nan
                qs[i, 1] = np.nan

            # Compute q3
            z_b3 = v_2P_set[i]/np.linalg.norm(v_2P_set[i])
            y_b3 = np.cross(-v_26, v_2P_set[i])
            y_b3 = y_b3/np.linalg.norm(y_b3)
            x_b3 = np.cross(y_b3, z_b3)
            c1 = np.cos(qs[i, 0])
            s1 = np.sin(qs[i, 0])
            R_b1 = np.array([[c1,  -s1,  0.0],
                             [s1,   c1,  0.0],
                             [0.0,  0.0,  1.0]])
            c2 = np.cos(qs[i, 1])
            s2 = np.sin(qs[i, 1])
            R_12 = np.array([[c2,  -s2, 0.0],
                             [0.0,  0.0, 1.0],
                             [-s2,  -c2, 0.0]])
            R_b2 = R_b1 @ R_12
            x_23 = R_b2.T @ x_b3
            qs[i, 2] = np.arctan2(x_23[2], x_23[0])

            if qs[i, 2] <= pandaVar.q_min[2] or qs[i, 2] >= pandaVar.q_max[2]:
                qs[i, 2] = np.nan

            # Compute q5
            v_H4 = p_b2 + pandaVar.L2*z_b3 + pandaVar.offset*x_b3 - p_b6 + pandaVar.L3*z_b5_set[i]
            c6 = np.cos(qs[i, 5])
            s6 = np.sin(qs[i, 5])
            R_56 = np.array([[c6,  -s6,  0.0],
                             [0.0,  0.0, -1.0],
                             [s6,   c6,  0.0]])
            R_b5 = R_b6 @ R_56.T
            v_5_H4 = R_b5.T @ v_H4

            qs[i, 4] = -np.arctan2(v_5_H4[1], v_5_H4[0])
            if qs[i, 4] <= pandaVar.q_min[4] or qs[i, 4] >= pandaVar.q_max[4]:
                qs[i, 4] = np.nan

        q_result = np.array([q for q in qs if ~np.isnan(q).any()])    # feasible solutions with removed NaN
        # q_result = np.array([q for q in q_result if -np.pi/2 <= q[0] <= np.pi/2 and -np.pi/2 <= q[2] <= np.pi/2])
        if len(q_result) == 0:
            print (qs)
            import pdb; pdb.set_trace()
        norms = np.linalg.norm(q_result - q_curr, axis=1)
        return q_result[np.argmin(norms)]   # Finally select the solution that is closest to the q_curr

    @classmethod
    # "Case-Consistent" IK solution
    # Algorithm proposed in the paper
    def ik_analytic(cls, Tb_ed, q_curr, q7=None):
        # If q7 is not given, select q7 that makes the closest configuration to the neutral.
        # However, it will take more time for optimizing. (about 4~7 ms)
        if q7 is None:
            res = minimize_scalar(
                pandaKinematics.joint_margin,
                args=(Tb_ed, q_curr), bounds=(pandaVar.q_min[-1], pandaVar.q_max[-1]), method='bounded')
            q7 = res.x
        q_NaN = [np.NaN]*7

        # return NAN if input q7 is out of range
        if q7 <= pandaVar.q_min[6] or q7 >= pandaVar.q_max[6]:
            return q_NaN

        # FK for getting current case id
        c1_a = np.cos(q_curr[0]); s1_a = np.sin(q_curr[0])
        c2_a = np.cos(q_curr[1]); s2_a = np.sin(q_curr[1])
        c3_a = np.cos(q_curr[2]); s3_a = np.sin(q_curr[2])
        c4_a = np.cos(q_curr[3]); s4_a = np.sin(q_curr[3])
        c5_a = np.cos(q_curr[4]); s5_a = np.sin(q_curr[4])
        c6_a = np.cos(q_curr[5]); s6_a = np.sin(q_curr[5])

        Ts_a = []
        Ts_a.append(np.array([[c1_a, -s1_a,  0.0,  0.0],
                              [s1_a,  c1_a,  0.0,  0.0],
                              [0.0,   0.0,  1.0,   pandaVar.L1],
                              [0.0,   0.0,  0.0,  1.0]]))
        Ts_a.append(np.array([[c2_a, -s2_a,  0.0,  0.0],
                          [0.0,   0.0,  1.0,  0.0],
                          [-s2_a, -c2_a,  0.0,  0.0],
                          [0.0,   0.0,  0.0,  1.0]]))
        Ts_a.append(np.array([[c3_a, -s3_a,  0.0,  0.0],
                          [0.0,   0.0, -1.0,  -pandaVar.L2],
                          [s3_a,  c3_a,  0.0,  0.0],
                          [0.0,   0.0,  0.0,  1.0]]))
        Ts_a.append(np.array([[c4_a, -s4_a,  0.0,   pandaVar.offset],
                          [0.0,   0.0, -1.0,  0.0],
                          [s4_a,  c4_a,  0.0,  0.0],
                          [0.0,   0.0,  0.0,  1.0]]))
        Ts_a.append(np.array([[1.0,   0.0,  0.0,  -pandaVar.offset],
                          [0.0,   1.0,  0.0,  0.0],
                          [0.0,   0.0,  1.0,  0.0],
                          [0.0,   0.0,  0.0,  1.0]]))
        Ts_a.append(np.array([[c5_a, -s5_a,  0.0,  0.0],
                          [0.0,   0.0,  1.0,   pandaVar.L3],
                          [-s5_a, -c5_a,  0.0,  0.0],
                          [0.0,   0.0,  0.0,  1.0]]))
        Ts_a.append(np.array([[c6_a, -s6_a,  0.0,  0.0],
                          [0.0,   0.0, -1.0,  0.0],
                          [s6_a,  c6_a,  0.0,  0.0],
                          [0.0,   0.0,  0.0,  1.0]]))

        # Never use multi_dot for speed-up. It slows down the computation A LOT! Use "for-loop" instead.
        Tbs_a = []
        T_temp = np.eye(4)
        for T in Ts_a:
            T_temp = T_temp.dot(T)
            Tbs_a.append(T_temp)

        # identify q6 case
        v_26_a = Tbs_a[6][:3, -1] - Tbs_a[1][:3, -1]
        x_b5_a = np.array([Tbs_a[5][2, -1], 0, 0])
        is_case6_0 = np.dot(v_26_a, x_b5_a) <= 0

        # identify q1 case
        is_case1_1 = q_curr[1] < 0

        # IK: compute p_6
        R_be = Tb_ed[:3, :3]
        z_be = R_be[:3, 2]
        p_be = Tb_ed[:3, -1]

        # Compute q4 (unique feasible solution)
        p_b7 = p_be - pandaVar.L5 * z_be
        x_e6 = np.array([np.cos(q7), -np.sin(q7), 0.0])
        x_b6 = R_be @ x_e6
        p_b6 = p_b7 - pandaVar.L4 * x_b6

        # IK: compute q4
        p_b2 = np.array([0.0, 0.0, pandaVar.L1])
        v_26 = p_b6 - p_b2
        L26 = np.linalg.norm(v_26)
        L24 = np.sqrt(pandaVar.offset ** 2 + pandaVar.L2 ** 2)
        L46 = np.sqrt(pandaVar.offset ** 2 + pandaVar.L3 ** 2)
        if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
            print("L24=", L24)
            print("L46=", L46)
            print("L26=", L26)
            return q_NaN

        ang_H46 = np.arctan(pandaVar.L3 / pandaVar.offset)
        ang_342 = np.arctan(pandaVar.L2 / pandaVar.offset)
        ang_46H = np.arctan(pandaVar.offset / pandaVar.L3)
        ang_246 = np.arccos((L24 ** 2 + L46 ** 2 - L26 ** 2) / (2.0 * L24 * L46))
        q4 = ang_246 + ang_H46 + ang_342 - 2.0 * np.pi
        if q4 <= pandaVar.q_min[3] or q4 >= pandaVar.q_max[3]:
            print("q4 value is invalid")
            return q_NaN

        # Compute q6 (two solutions)
        ang_462 = np.arccos((L26 ** 2 + L46 ** 2 - L24 ** 2) / (2.0 * L26 * L46))
        ang_26H = ang_46H + ang_462

        z_b6 = np.cross(z_be, x_b6)
        y_b6 = np.cross(z_b6, x_b6)
        R_b6 = np.c_[x_b6, y_b6, z_b6]
        v_6_62 = R_b6.T @ (-v_26)

        phi6 = np.arctan2(v_6_62[1], v_6_62[0])
        psi6 = np.arcsin((-L26 * np.cos(ang_26H)) / np.sqrt(v_6_62[0] ** 2 + v_6_62[1] ** 2))
        # print ("arcsin", (-L26 * np.cos(ang_26H)) / np.sqrt(v_6_62[0] ** 2 + v_6_62[1] ** 2))

        # Two candidates
        q6_c1 = np.pi - psi6 - phi6
        q6_c2 = psi6 - phi6
        if is_case6_0:
            q6 = q6_c1
        else:
            q6 = q6_c2

        if q6 <= pandaVar.q_min[5]:
            q6 += 2.0 * np.pi
        elif q6 >= pandaVar.q_max[5]:
            q6 -= 2.0 * np.pi

        if q6 <= pandaVar.q_min[5] or q6 >= pandaVar.q_max[5]:
            print("q6 value is invalid")
            return q_NaN

        # Compute q1 & q2
        ang_324 = np.pi / 2 - ang_342
        ang_426 = np.pi - ang_246 - ang_462
        ang_P26 = ang_324 + ang_426
        ang_2P6 = np.pi - ang_P26 - ang_26H
        LP6 = L26 * np.sin(ang_P26) / np.sin(ang_2P6)

        z_65 = np.array([np.sin(q6), np.cos(q6), 0.0])
        z_b5 = R_b6 @ z_65
        v_2P = (p_b6 - p_b2) - LP6 * z_b5
        L2P = np.linalg.norm(v_2P)

        if np.abs(v_2P[2] / L2P) > 0.99999:  # Singularity by q2 = 0. In this case, q1 and q3 has infinite solutions.
            print("Singularity!")
            q1 = q_curr[0]  # q1 gets the same value with q0(reference)
            q2 = 0.0
        else:
            q1 = np.arctan2(v_2P[1], v_2P[0])
            q2 = np.arccos(v_2P[2] / L2P)
            if is_case1_1:
                if q1 < 0:
                    q1 = q1 + np.pi
                else:
                    q1 = q1 - np.pi
                q2 = -q2
        if q1 <= pandaVar.q_min[0] or q1 >= pandaVar.q_max[0] or q2 <= pandaVar.q_min[1] or q2 >= pandaVar.q_max[1]:
            print("q1 & q2 values are invalid")
            return q_NaN

        # Compute q3
        z_b3 = v_2P / np.linalg.norm(v_2P)
        y_b3 = np.cross(-v_26, v_2P)
        y_b3 = y_b3 / np.linalg.norm(y_b3)
        x_b3 = np.cross(y_b3, z_b3)
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        R_b1 = np.array([[c1, -s1, 0.0],
                         [s1, c1, 0.0],
                         [0.0, 0.0, 1.0]])
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        R_12 = np.array([[c2, -s2, 0.0],
                         [0.0, 0.0, 1.0],
                         [-s2, -c2, 0.0]])
        R_b2 = R_b1 @ R_12
        x_23 = R_b2.T @ x_b3
        q3 = np.arctan2(x_23[2], x_23[0])
        if q3 <= pandaVar.q_min[2] or q3 >= pandaVar.q_max[2]:
            print("q3 value is invalid")
            return q_NaN

        # Compute q5
        v_H4 = p_b2 + pandaVar.L2 * z_b3 + pandaVar.offset * x_b3 - p_b6 + pandaVar.L3 * z_b5
        c6 = np.cos(q6)
        s6 = np.sin(q6)
        R_56 = np.array([[c6, -s6, 0.0],
                         [0.0, 0.0, -1.0],
                         [s6, c6, 0.0]])
        R_b5 = R_b6 @ R_56.T
        v_5_H4 = R_b5.T @ v_H4
        q5 = -np.arctan2(v_5_H4[1], v_5_H4[0])
        if q5 <= pandaVar.q_min[4] or q5 >= pandaVar.q_max[4]:
            print ("q5 value is invalid")
            return q_NaN
        return [q1, q2, q3, q4, q5, q6, q7]

    @classmethod
    def joint_margin(cls, q7, Tb_ed, q_curr):
        q_ik = pandaKinematics.ik_analytic(Tb_ed=Tb_ed, q_curr=q_curr, q7=q7)
        if np.isnan(q_ik).any():
            score = 99999
        else:
            q_avr = (pandaVar.q_max + pandaVar.q_min)/2
            score = np.linalg.norm(q_avr - q_ik)
        return score

    @classmethod
    def null_space_control(cls, joints, crit='joint_limit'):    # Null-space control input
        Tbs = pandaKinematics.fk(joints=joints)[0]
        J = pandaKinematics.jacobian(Tbs)
        Jp = np.linalg.pinv(J)
        Jn = np.eye(7) - Jp.dot(J)
        k=0.1
        if crit == 'joint_limit':   # distance to joint limits
            qk_null_dot = [k*pandaKinematics.partial_derivative(pandaKinematics.distance_to_joint_limits, i, joints)
                           for i in range(len(joints))]
        elif crit == 'manipulability':
            qk_null_dot = [k * pandaKinematics.partial_derivative(pandaKinematics.manipulability, i, joints)
                           for i in range(len(joints))]
        elif crit == 'obstacle_avoidance':
            qk_null_dot = [k * pandaKinematics.partial_derivative(pandaKinematics.obstacle_avoidance, i, joints)
                           for i in range(len(joints))]
        else:
            raise ValueError
        return Jn.dot(qk_null_dot)

    @classmethod
    def manipulability(cls, q1, q2, q3, q4, q5, q6, q7):
        J = pandaKinematics.jacobian([q1, q2, q3, q4, q5, q6, q7])
        det = np.linalg.det(J.dot(J.T))
        return np.sqrt(det)

    @classmethod
    def distance_to_joint_limits(cls, q1, q2, q3, q4, q5, q6, q7):
        q = [q1, q2, q3, q4, q5, q6, q7]
        dist = [((q - (q_max+q_min)/2)/(q_max - q_min))**2 for q, q_max, q_min in zip(q, q_max, q_min)]
        return -np.sum(dist)/7/2

    @classmethod
    def obstacle_avoidance(cls, q1, q2, q3, q4, q5, q6, q7):
        q = [q1, q2, q3, q4, q5, q6, q7]
        Tbs = pandaKinematics.fk(joints=q)[0]
        p04 = Tbs[3][:3, -1]   # we can define multiple points on the robot
        p_obj = np.array([0.5, -0.5, 0.3])
        return np.linalg.norm(p04 - p_obj)

    @classmethod
    def partial_derivative(cls, func, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return scipy.misc.derivative(wraps, point[var], dx=1e-6)


# Test IK
if __name__ == "__main__":
    from panda.ros.pandaRviz import pandaRviz
    from panda.ros.pandaRvizMotion import pandaRvizMotion
    import numpy as np
    import time
    import panda.pandaVar as pandaVar

    sim = pandaRviz(is_core=True, is_dual=False, use_GUI=False)
    motion = pandaRvizMotion(ns='panda1')

    T_des = np.array([[0.7071, -0.7071, 0., 0.375],
                      [-0.7071, -0.7071, 0., -0.1],
                      [0., -0., -1., 0.211],
                      [0., 0., 0., 1.]])

    q_ik_num = pandaKinematics.ik(Tb_ed=T_des, q0=None)
    print("q_ik_num=", q_ik_num)
    T_fk_num = pandaKinematics.fk(joints=q_ik_num)
    print("T_fk_num=", T_fk_num[0][-1])
    print()

    q_ik_anal = pandaKinematics.ik_analytic(Tb_ed=T_des, q_curr=q_ik_num, q7=None)
    print("q_ik_anal=", q_ik_anal)
    T_fk_anal = pandaKinematics.fk(joints=q_ik_anal)
    print("T_fk_anal=", T_fk_anal[0][-1])
    print()

    motion.set_joint_position(joints=q_ik_anal)
    input()


# if __name__ == "__main__":
#     # FK
#     joints = [0.95, -0.702, -0.678, -2.238, -0.365, 0.703, 0.808]
#     Tbe = pandaKinematics.fk(joints=joints)[0][-1]
#
#     # IK
#     import time
#     st = time.time()
#     print("q_des=", joints)
#     print ("q_ik =", pandaKinematics.ik(Tb_ed=Tbe))
#     print ("t_comp=", time.time() - st)