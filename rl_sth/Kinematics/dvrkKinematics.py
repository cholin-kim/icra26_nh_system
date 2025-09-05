import numpy as np
import sympy as sp

# from Kinematics import dvrkVar
from ..Kinematics import dvrkVar


class dvrkKinematics:
    @ classmethod
    def DH_transform(cls, dhparams): # stacks transforms of neighbor frame, following the modified DH convention
        Ts = [np.array([[np.cos(theta), -np.sin(theta), 0, a],
                        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                         -np.sin(alpha) * d],
                        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha),
                         np.cos(alpha) * d],
                        [0, 0, 0, 1]]) for [a, alpha, d, theta] in dhparams]
        return Ts

    @classmethod
    def fk(cls, joints):
        """
        joints = (6,)
        Ts = (Tb1, T12, T23, ...)
        """
        Ts = dvrkKinematics.DH_transform(dvrkVar.dhparam(joints))     # Tb1, T12, T23, ...
        Tbi = np.eye(4)
        Tbs = []
        for T in Ts:
            Tbi = Tbi.dot(T)
            Tbs.append(Tbi)

        return Tbs, Ts



    @classmethod
    def ik(self, T):
        # analytic method, based on "Automating Surgical Peg Transfer: Calibration with Deep Learning Can Exceed Speed, Accuracy, and Consistency of Humans"
        T = np.linalg.inv(T)
        t_x = T[0, -1]
        t_y = T[1, -1]
        t_z = T[2, -1]
        q6 = np.arctan2(t_x, t_z - dvrkVar.L4)
        p = -dvrkVar.L3 + np.sqrt(t_x ** 2 + (t_z - dvrkVar.L4) ** 2)

        ## 일단 교수님것으로 사용.
        q5 = np.arctan2(-t_y, p)
        q3 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(t_y ** 2 + p ** 2)

        R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
                        [np.cos(q5), 0, -np.sin(q5)],
                        [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
        R80 = T[:3, :3]
        R40 = R84.T @ R80
        n32 = R40[2, 1]
        n31 = R40[2, 0]
        n33 = R40[2, 2]
        n22 = R40[1, 1]
        n12 = R40[0, 1]
        q2 = np.arcsin(n32)
        q1 = np.arctan2(-n31, n33)
        q4 = np.arctan2(n22, n12)
        joint = np.array([q1, q2, q3, q4, q5, q6]).T
        assert ~np.isnan(joint).any()

        return joint


    # @classmethod
    # def ik(self, T):
    #     # analytic method, based on "Automating Surgical Peg Transfer: Calibration with Deep Learning Can Exceed Speed, Accuracy, and Consistency of Humans"
    #     T = np.linalg.inv(T)
    #     t_x = T[0, -1]
    #     t_y = T[1, -1]
    #     t_z = T[2, -1]
    #     q6 = np.arctan2(t_x, t_z - dvrkVar.L4)
    #     p = -dvrkVar.L3 + np.sqrt(t_x ** 2 + (t_z - dvrkVar.L4) ** 2)
    #     p_prime = -dvrkVar.L3 - np.sqrt(t_x ** 2 + (t_z - dvrkVar.L4) ** 2)
    #     q5_cand1 = np.arctan2(-t_y, p)
    #     q5_cand2 = np.arctan2(-t_y, p_prime)
    #     q3_cand1 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(t_y ** 2 + p ** 2)
    #     q3_cand2 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(t_y ** 2 + p_prime ** 2)
    #     q3_cand3 = dvrkVar.L1 - dvrkVar.L2 - np.sqrt(t_y ** 2 + p ** 2)
    #     q3_cand4 = dvrkVar.L1 - dvrkVar.L2 - np.sqrt(t_y ** 2 + p_prime ** 2)
    #     joint_lst = []
    #
    #     q5_cand_lst = np.array([q5_cand1, q5_cand2])
    #     q3_cand_lst = np.array([q3_cand1, q3_cand2, q3_cand3, q3_cand4])
    #
    #     for q5 in q5_cand_lst:
    #         for q3 in q3_cand_lst:
    #             R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
    #                             [np.cos(q5), 0, -np.sin(q5)],
    #                             [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
    #             R80 = T[:3, :3]
    #             R40 = R84.T @ R80
    #             n32 = R40[2, 1]
    #             n31 = R40[2, 0]
    #             n33 = R40[2, 2]
    #             n22 = R40[1, 1]
    #             n12 = R40[0, 1]
    #             q2 = np.arcsin(n32)
    #             q1 = np.arctan2(-n31, n33)
    #             q4 = np.arctan2(n22, n12)
    #             joint = np.array([q1, q2, q3, q4, q5, q6]).T
    #             assert ~np.isnan(joint).any()
    #             # if np.any(joint > dvrkVar.joint_range_upper_limit) or np.any(joint < dvrkVar.joint_range_lower_limit):
    #             #     pass
    #             # else:
    #             #     joint_lst.append(joint)
    #             joint_lst.append(joint)
    #
    #     return joint_lst



    @classmethod
    def jacobian(cls, joints):  # 가져온 것. 아직 이해 못함.
        q1, q2, q3, q4, q5, q6 = np.array(joints).T
        J = np.zeros((6, 6))
        J[0, 0] = np.cos(q6) * ((np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2))) / 200 + (np.cos(q1) * np.cos(q2) * np.cos(q5)) / 200) - np.sin(q6) * ((np.cos(q4) * np.sin(q1)) / 200 + (np.cos(q1) * np.sin(q2) * np.sin(q4)) / 200) + np.sin(q5) * ((91 * np.sin(q1) * np.sin(q4)) / 10000 - (91 * np.cos(q1) * np.cos(q4) * np.sin(q2)) / 10000) + q3 * np.cos(q1) * np.cos(q2) + (91 * np.cos(q1) * np.cos(q2) * np.cos(q5)) / 10000
        J[0, 1] = -(np.sin(q1) * (91 * np.cos(q5) * np.sin(q2) + 10000 * q3 * np.sin(q2) + 91 * np.cos(q2) * np.cos(q4) * np.sin(q5) + 50 * np.cos(q5) * np.cos(q6) * np.sin(q2) + 50 * np.cos(q2) * np.sin(q4) * np.sin(q6) + 50 * np.cos(q2) * np.cos(q4) * np.cos(q6) * np.sin(q5))) / 10000
        J[0, 2] = np.cos(q2) * np.sin(q1)
        J[0, 3] = - (91 * np.sin(q5) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) / 10000 - np.sin(q6) * ((np.cos(q1) * np.sin(q4)) / 200 + (np.cos(q4) * np.sin(q1) * np.sin(q2)) / 200) - (np.cos(q6) * np.sin(q5) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) / 200
        J[0, 4] = -((50 * np.cos(q6) + 91) * (np.cos(q1) * np.cos(q5) * np.sin(q4) + np.cos(q2) * np.sin(q1) * np.sin(q5) + np.cos(q4) * np.cos(q5) * np.sin(q1) * np.sin(q2))) / 10000
        J[0, 5] = np.sin(q6) * ((np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2))) / 200 - (np.cos(q2) * np.cos(q5) * np.sin(q1)) / 200) + np.cos(q6) * ((np.cos(q1) * np.cos(q4)) / 200 - (np.sin(q1) * np.sin(q2) * np.sin(q4)) / 200)

        J[1, 0] = 0
        J[1, 1] = (np.sin(q2) * np.sin(q4) * np.sin(q6)) / 200 - np.cos(q6) * ((np.cos(q2) * np.cos(q5)) / 200 - (np.cos(q4) * np.sin(q2) * np.sin(q5)) / 200) - q3 * np.cos(q2) - (91 * np.cos(q2) * np.cos(q5)) / 10000 + (91 * np.cos(q4) * np.sin(q2) * np.sin(q5)) / 10000
        J[1, 2] = -np.sin(q2)
        J[1, 3] = (np.cos(q2) * (91 * np.sin(q4) * np.sin(q5) - 50 * np.cos(q4) * np.sin(q6) + 50 * np.cos(q6) * np.sin(q4) * np.sin(q5))) / 10000
        J[1, 4] = ((np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5)) * (50 * np.cos(q6) + 91)) / 10000
        J[1, 5] = np.sin(q6) * ((np.cos(q5) * np.sin(q2)) / 200 + (np.cos(q2) * np.cos(q4) * np.sin(q5)) / 200) - (np.cos(q2) * np.cos(q6) * np.sin(q4)) / 200

        J[2, 0] = np.sin(q6) * ((np.cos(q1) * np.cos(q4)) / 200 - (np.sin(q1) * np.sin(q2) * np.sin(q4)) / 200) - np.cos(q6) * ((np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2))) / 200 - (np.cos(q2) * np.cos(q5) * np.sin(q1)) / 200) - np.sin(q5) * ((91 * np.cos(q1) * np.sin(q4)) / 10000 + (91 * np.cos(q4) * np.sin(q1) * np.sin(q2)) / 10000) + q3 * np.cos(q2) * np.sin(q1) + (91 * np.cos(q2) * np.cos(q5) * np.sin(q1)) / 10000
        J[2, 1] = (np.cos(q1) * (91 * np.cos(q5) * np.sin(q2) + 10000 * q3 * np.sin(q2) + 91 * np.cos(q2) * np.cos(q4) * np.sin(q5) + 50 * np.cos(q5) * np.cos(q6) * np.sin(q2) + 50 * np.cos(q2) * np.sin(q4) * np.sin(q6) + 50 * np.cos(q2) * np.cos(q4) * np.cos(q6) * np.sin(q5))) / 10000
        J[2, 2] = -np.cos(q1) * np.cos(q2)
        J[2, 3] = - (np.sin(q6) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2))) / 200 - np.sin(q5) * ((91 * np.cos(q4) * np.sin(q1)) / 10000 + (91 * np.cos(q1) * np.sin(q2) * np.sin(q4)) / 10000) - (np.cos(q6) * np.sin(q5) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) / 200
        J[2, 4] = ((50 * np.cos(q6) + 91) * (np.cos(q1) * np.cos(q2) * np.sin(q5) - np.cos(q5) * np.sin(q1) * np.sin(q4) + np.cos(q1) * np.cos(q4) * np.cos(q5) * np.sin(q2))) / 10000
        J[2, 5] = np.sin(q6) * ((np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2))) / 200 + (np.cos(q1) * np.cos(q2) * np.cos(q5)) / 200) + np.cos(q6) * ((np.cos(q4) * np.sin(q1)) / 200 + (np.cos(q1) * np.sin(q2) * np.sin(q4)) / 200)

        J[3, 0] = (np.sin(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) + np.cos(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q4)) + (np.cos(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) - np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q2) * np.sin(q4) * np.sin(q6)) - (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q2) * np.sin(q1) * np.sin(q5)) * (np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5))
        J[3, 1] = - (np.cos(q6) * (np.cos(q1) * np.cos(q5) * np.sin(q2) + np.cos(q1) * np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q1) * np.cos(q2) * np.sin(q4) * np.sin(q6)) * (np.cos(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q2) * np.sin(q4) * np.sin(q6)) - (np.sin(q6) * (np.cos(q1) * np.cos(q5) * np.sin(q2) + np.cos(q1) * np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q1) * np.cos(q2) * np.cos(q6) * np.sin(q4)) * (np.sin(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q4)) - (np.cos(q1) * np.sin(q2) * np.sin(q5) - np.cos(q1) * np.cos(q2) * np.cos(q4) * np.cos(q5)) * (np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5))
        J[3, 2] = 0
        J[3, 3] = (np.sin(q6) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q6) * np.sin(q5) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q2) * np.sin(q4) * np.sin(q6)) - (np.cos(q6) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.sin(q5) * np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q4)) - np.cos(q5) * (np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5)) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))
        J[3, 4] = (np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5)) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.cos(q1) * np.cos(q2) * np.sin(q5)) * (np.cos(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q2) * np.sin(q4) * np.sin(q6)) + np.sin(q6) * (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.cos(q1) * np.cos(q2) * np.sin(q5)) * (np.sin(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q4))
        J[3, 5] = (np.cos(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) - np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q4)) - (np.sin(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q2) * np.sin(q4) * np.sin(q6))

        J[4, 0] = - (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.cos(q1) * np.cos(q2) * np.sin(q5)) ** 2 - (np.sin(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) ** 2 - (np.cos(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) - np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) ** 2
        J[4, 1] = (np.sin(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.cos(q5) * np.sin(q1) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q1) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q1) * np.sin(q4)) + (np.cos(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) - np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.cos(q5) * np.sin(q1) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q1) * np.sin(q5)) + np.cos(q2) * np.sin(q1) * np.sin(q4) * np.sin(q6)) - (np.sin(q1) * np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5) * np.sin(q1)) * (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.cos(q1) * np.cos(q2) * np.sin(q5))
        J[4, 2] = 0
        J[4, 3] = (np.cos(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) - np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q6) * np.sin(q5) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) - (np.sin(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.sin(q5) * np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) + np.cos(q5) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4)) * (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.cos(q1) * np.cos(q2) * np.sin(q5))
        J[4, 4] = np.cos(q6) * (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q2) * np.sin(q1) * np.sin(q5)) * (np.cos(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) - np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) - (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) * (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) - np.cos(q1) * np.cos(q2) * np.sin(q5)) + np.sin(q6) * (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q2) * np.sin(q1) * np.sin(q5)) * (np.sin(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4)))
        J[4, 5] = (np.sin(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) + np.cos(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) - np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) - (np.cos(q6) * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q1) * np.cos(q4) * np.sin(q2)) + np.cos(q1) * np.cos(q2) * np.cos(q5)) - np.sin(q6) * (np.cos(q4) * np.sin(q1) + np.cos(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) + np.cos(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4)))

        J[5, 0] = 0
        J[5, 1] = (np.sin(q6) * (np.cos(q2) * np.cos(q5) - np.cos(q4) * np.sin(q2) * np.sin(q5)) + np.cos(q6) * np.sin(q2) * np.sin(q4)) * (np.sin(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) + np.cos(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) + (np.cos(q6) * (np.cos(q2) * np.cos(q5) - np.cos(q4) * np.sin(q2) * np.sin(q5)) - np.sin(q2) * np.sin(q4) * np.sin(q6)) * (np.cos(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) - np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) - (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q2) * np.sin(q1) * np.sin(q5)) * (np.cos(q2) * np.sin(q5) + np.cos(q4) * np.cos(q5) * np.sin(q2))
        J[5, 2] = 0
        J[5, 3] = (np.cos(q2) * np.cos(q4) * np.sin(q6) - np.cos(q2) * np.cos(q6) * np.sin(q4) * np.sin(q5)) * (np.cos(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) - np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) - (np.cos(q2) * np.cos(q4) * np.cos(q6) + np.cos(q2) * np.sin(q4) * np.sin(q5) * np.sin(q6)) * (np.sin(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) + np.cos(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) - np.cos(q2) * np.cos(q5) * np.sin(q4) * (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q2) * np.sin(q1) * np.sin(q5))
        J[5, 4] = - (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) + np.cos(q2) * np.sin(q1) * np.sin(q5)) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q6) * (np.cos(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) - np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5)) - np.sin(q6) * (np.sin(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) + np.cos(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q2) * np.sin(q5) - np.cos(q2) * np.cos(q4) * np.cos(q5))
        J[5, 5] = (np.sin(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) + np.cos(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) * (np.cos(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) + np.cos(q2) * np.sin(q4) * np.sin(q6)) - (np.cos(q6) * (np.sin(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * np.sin(q1) * np.sin(q2)) - np.cos(q2) * np.cos(q5) * np.sin(q1)) - np.sin(q6) * (np.cos(q1) * np.cos(q4) - np.sin(q1) * np.sin(q2) * np.sin(q4))) * (np.sin(q6) * (np.cos(q5) * np.sin(q2) + np.cos(q2) * np.cos(q4) * np.sin(q5)) - np.cos(q2) * np.cos(q6) * np.sin(q4))

        return J

    @classmethod
    def DH_transform_sym_flipped(cls, dhparams): # stacks transforms of neighbor frame, following the modified DH convention
        Ts = [sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0, a],
                        [sp.sin(theta) * sp.cos(alpha), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha),
                         -sp.sin(alpha) * d],
                        [sp.sin(theta) * sp.sin(alpha), sp.cos(theta) * sp.sin(alpha), sp.cos(alpha),
                         sp.cos(alpha) * d],
                        [0, 0, 0, 1]]) for [a, alpha, d, theta] in dhparams]
        return Ts

    @ classmethod
    def fk_sym_flipped(cls):
        dhparams = dvrkVar.dhparam_sym_flipped()
        Ts = dvrkKinematics.DH_transform_sym_flipped(dhparams=dhparams)

        # Initialize the transformation matrix (identity matrix)
        Tbi = sp.eye(4)
        Tbs = []
        for T in Ts:
            Tbi = Tbi * (T)
            Tbs.append(Tbi)

        return Tbs, Ts

if __name__ == "__main__":
    dvrkkin = dvrkKinematics()
    np.set_printoptions(suppress=True, precision=3)

    TW_RB1 = np.identity(4)
    TW_RB2 = np.identity(4)

    TW_RB1[:3, -1] = np.array([-0.12, 0, 0.20]).T
    TW_RB2[:3, -1] = np.array([0.12, 0, 0.20]).T

    Tw_targ = np.array([[-0.5 ,     0.86603,  0. ,     -0.012  ],
 [-0.86603, -0.5 ,     0.   ,   -0.02078],
 [ 0.     ,  0.   ,    1.   ,    0.     ],
 [ 0.  ,     0.      , 0.   ,    1.     ]])

    Trb_targ = np.linalg.inv(TW_RB2) @ Tw_targ
    joint_pos = dvrkkin.ik(Trb_targ)

    # Tw_pickup_after
    # : [[0.5 - 0.86603  0.       0.012]
    #    [0.86603  0.5      0.       0.12078]
    # [0.
    # 0.
    # 1.
    # 0.05]
    # [0.       0.       0.       1.]]

    print(joint_pos)
    quit()

    # 1. random joints
    # init_joint = np.random.uniform(dvrkVar.joint_range_lower_limit, dvrkVar.joint_range_upper_limit)
    # print("joints:", init_joint)
    #
    #
    # # 2. forward kinematics
    # Tbs, Ts = dvrkkin.fk(init_joint)
    # # print("Tbe:\n", Tbs[-1])
    # print("Tbe_trans\n", Tbs[-1][:3, -1])
    #
    # # 2'. forward kinematics, symbolic(flipped)
    # # Tbs_sym, Ts_sym = dvrkkin.fk_sym_flipped()
    # # Tb3_sym = Tbs[2]
    # # for i in range(len(Tbs_sym)):
    # #     print(f"Tb{i}_sym")
    # #     print(sp.latex(Tbs_sym[i]))
    #
    # # Tb0, T12, T23, T34, T45, T56, T67, T78 = Ts_sym[0], Ts_sym[1], Ts_sym[2], Ts_sym[3], Ts_sym[3], Ts_sym[5], Ts_sym[6], Ts_sym[7]
    # # R48 = (T45 * T56 * T67 * T78)[:3, :3]
    # # print("R48:", sp.latex(R48))
    #
    # # 3. inverse kinematics(analytic)
    # Tb_ed = Tbs[-1].copy()
    # joint_lst = dvrkkin.ik(Tb_ed)
    # print("joint_lst:\n", joint_lst)
    #
    # # 3-1. check whether fk(ik_result) = fk_ground
    # for joint in joint_lst:
    #     # print(dvrkkin.fk(joint)[0][-1])
    #     # since q3, q5 only affects translation, checking translation only.
    #     print(dvrkkin.fk(joint)[0][-1][:3, -1])
    # print("---------------------------------------")
    # # see translation difference with fk_ground
    # cart_err_lst = []
    # for joint in joint_lst:
    #     res = dvrkkin.fk(joint)[0][-1][:3, -1] - Tb_ed[:3, -1]
    #     total_diff = np.linalg.norm(res)
    #     cart_err_lst.append(total_diff)
    #     total_diff_joint = np.linalg.norm(joint-init_joint)
    #     print("total_difference_in_cartesian:", round(total_diff, 5), "m")
    #     print("total_difference_in_joint:", total_diff_joint)
    #     # print("difference in each axis:", np.abs(res))
    # print("best_ik_joint:", joint_lst[np.argmin(cart_err_lst)])

    ###############################################
    ## Visualize result
    ###############################################
    # 1. generate ik dataset with random trajectory
    traj_size = 20
    random_joint_traj = np.random.uniform(dvrkVar.joint_range_lower_limit, dvrkVar.joint_range_upper_limit, size=(traj_size, 6))

    T_traj = np.zeros((traj_size, 4, 4))
    joint_lst_stacked = np.zeros((traj_size, 8, 6))
    for s in range(traj_size):
        T_traj[s] = dvrkkin.fk(random_joint_traj[s])[0][-1]
        joint_lst_stacked[s] = dvrkkin.ik(T_traj[s])

    # 2. visualize result
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))


    t = np.linspace(0, 10, traj_size)
    axs[0, 0].set_title("Random_joint_traj")
    axs[0, 1].set_title("IK_result, q1")
    axs[0, 2].set_title("IK_result, q2")
    axs[0, 3].set_title("IK_result, q3")
    axs[1, 0].set_title("IK_result, q4")
    axs[1, 1].set_title("IK_result, q5")
    axs[1, 2].set_title("IK_result, q6")

    for i in range(6):
        axs[0, 0].plot(t, random_joint_traj[:, i], label=f'q{i}')
    axs[0, 0].legend(loc='upper left', bbox_to_anchor=(-0.5, 1))


    ik_q1 = joint_lst_stacked[:, :, 0]  # (traj_size, 8)
    ik_q2 = joint_lst_stacked[:, :, 1]
    ik_q3 = joint_lst_stacked[:, :, 2]
    ik_q4 = joint_lst_stacked[:, :, 3]
    ik_q5 = joint_lst_stacked[:, :, 4]
    ik_q6 = joint_lst_stacked[:, :, 5]

    ik_lst = [ik_q1, ik_q2, ik_q3, ik_q4, ik_q5, ik_q6]
    axs_lst = [axs[0, 1], axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2]]

    for a in range(len(axs_lst)):
        ax = axs_lst[a]
        ik_qi = ik_lst[a]
        ax.plot(t, np.ones_like(t) * dvrkVar.joint_range_lower_limit[a], c='gray', alpha=0.3, linestyle='--')
        ax.plot(t, np.ones_like(t) * dvrkVar.joint_range_upper_limit[a], c='gray', alpha=0.3, linestyle='--')
        for p in range(np.shape(ik_q1)[-1]):    # 0-7
            if p == 0:
                ax.plot(t, ik_qi[:, p], c='red', linewidth=2, label='ik_prof')
            else:
                # ax.plot(t, ik_qi[:, p], alpha=0.3, label=f'ik_{p}')
                ax.plot(t, ik_qi[:, p], label=f'ik_{p}')
        if a == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1))

    # plot joint error
    fig2, axs2 = plt.subplots(2, 4, figsize=(12, 8))
    fig2.suptitle("Absolute Joint Err")

    ik_prof = joint_lst_stacked[:, 0, :]  # (traj_size, 6)
    ik_2 = joint_lst_stacked[:, 1, :]
    ik_3 = joint_lst_stacked[:, 2, :]
    ik_4 = joint_lst_stacked[:, 3, :]
    ik_5 = joint_lst_stacked[:, 4, :]
    ik_6 = joint_lst_stacked[:, 5, :]
    ik_7 = joint_lst_stacked[:, 6, :]
    ik_8 = joint_lst_stacked[:, 7, :]


    axs2[0, 0].set_title("IK_prof")
    axs2[0, 1].set_title("IK_2")
    axs2[0, 2].set_title("IK_3")
    axs2[0, 3].set_title("IK_4")
    axs2[1, 0].set_title("IK_5")
    axs2[1, 1].set_title("IK_6")
    axs2[1, 2].set_title("IK_7")
    axs2[1, 3].set_title("IK_8")

    axs_lst2 = [axs2[0, 0], axs2[0, 1], axs2[0, 2], axs2[0, 3], axs2[1, 0], axs2[1, 1], axs2[1, 2], axs2[1, 3]]
    ik_lst2 = [ik_prof, ik_2, ik_3, ik_4, ik_5, ik_6, ik_7, ik_8]

    for a in range(len(axs_lst2)):
        ax = axs_lst2[a]
        diff = np.abs(random_joint_traj - ik_lst2[a])
        for p in range(np.shape(ik_prof)[-1]):  # 0-5
            ax.plot(t, diff[:, p], label=f'q_{p+1}(rad, m)')
        if a == 3:
            ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1))
        ax.set_ylim(0, 6)

    # plot cartesian err
    fig3, axs3 = plt.subplots(2, 4, figsize=(12, 8))
    fig3.suptitle("Absolute Cartesin Err")

    axs3[0, 0].set_title("IK_prof")
    axs3[0, 1].set_title("IK_2")
    axs3[0, 2].set_title("IK_3")
    axs3[0, 3].set_title("IK_4")
    axs3[1, 0].set_title("IK_5")
    axs3[1, 1].set_title("IK_6")
    axs3[1, 2].set_title("IK_7")
    axs3[1, 3].set_title("IK_8")

    axs_lst3 = [axs3[0, 0], axs3[0, 1], axs3[0, 2], axs3[0, 3], axs3[1, 0], axs3[1, 1], axs3[1, 2], axs3[1, 3]]
    for a in range(len(axs_lst3)):
        ax = axs_lst3[a]
        diff_lst = np.zeros((traj_size, 3))
        total_diff_lst = np.zeros((traj_size))
        for s in range(traj_size):
            T = dvrkkin.fk(ik_lst2[a][s])[0][-1]
            diff_lst[s] = np.abs(T[:3, -1] - T_traj[s][:3, -1])
            total_diff_lst[s] = np.linalg.norm(diff_lst[s])

        ax.plot(t, diff_lst[:, 0], c='r', label='x(m)')
        ax.plot(t, diff_lst[:, 1], c='g', label='y')
        ax.plot(t, diff_lst[:, 2], c='b', label='z')
        ax.plot(t, total_diff_lst, c='gray', label='total')
        ax.set_ylim(0, 0.5)
        if a == 3:
            ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1))


    plt.show()






