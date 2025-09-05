import numpy as np
import dvrkVar as dvrkVar


class dvrkKinematics():
    @classmethod
    def DH_transform(cls, a, alpha, d, theta, unit='rad'):  # modified DH convention
        if unit == 'deg':
            alpha = np.deg2rad(alpha)
            theta = np.deg2rad(theta)
        variables = [a, alpha, d, theta]
        N = 0
        for var in variables:
            if type(var) == np.ndarray or type(var) == list:
                N = len(var)
        if N == 0:
            T = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                          [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                           -np.sin(alpha) * d],
                          [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                          [0, 0, 0, 1]])
        else:
            T = np.zeros((N, 4, 4))
            T[:, 0, 0] = np.cos(theta)
            T[:, 0, 1] = -np.sin(theta)
            T[:, 0, 2] = 0.0
            T[:, 0, 3] = a
            T[:, 1, 0] = np.sin(theta) * np.cos(alpha)
            T[:, 1, 1] = np.cos(theta) * np.cos(alpha)
            T[:, 1, 2] = -np.sin(alpha)
            T[:, 1, 3] = -np.sin(alpha)*d
            T[:, 2, 0] = np.sin(theta) * np.sin(alpha)
            T[:, 2, 1] = np.cos(theta) * np.sin(alpha)
            T[:, 2, 2] = np.cos(alpha)
            T[:, 2, 3] = np.cos(alpha)*d
            T[:, 3, 0] = 0.0
            T[:, 3, 1] = 0.0
            T[:, 3, 2] = 0.0
            T[:, 3, 3] = 1.0
        return T

    @classmethod
    def fk(cls, joints, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4):
        q1, q2, q3, q4, q5, q6 = np.array(joints).T
        T01 = dvrkKinematics.DH_transform(0, np.pi / 2, 0, q1 + np.pi / 2)
        T12 = dvrkKinematics.DH_transform(0, -np.pi / 2, 0, q2 - np.pi / 2)
        T23 = dvrkKinematics.DH_transform(0, np.pi / 2, q3 - L1 + L2, 0)
        T34 = dvrkKinematics.DH_transform(0, 0, 0, q4)
        T45 = dvrkKinematics.DH_transform(0, -np.pi / 2, 0, q5 - np.pi / 2)
        T56 = dvrkKinematics.DH_transform(L3, -np.pi / 2, 0, q6 - np.pi / 2)
        T67 = dvrkKinematics.DH_transform(0, -np.pi / 2, L4, 0)
        # T78 = dvrkKinematics.DH_transform(0, np.pi, 0, np.pi)
        T08 = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(T67)#.dot(T78)
        return T08

    @classmethod
    def ik(self, T, method='analytic'):
        if method=='analytic':
            T = np.linalg.inv(T)
            if np.shape(T) == (4,4):
                x84 = T[0, 3]
                y84 = T[1, 3]
                z84 = T[2, 3]
                q6 = np.arctan2(x84, z84 - dvrkVar.L4)
                temp = -dvrkVar.L3 + np.sqrt(x84 ** 2 + (z84 - dvrkVar.L4) ** 2)
                q3 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(y84 ** 2 + temp ** 2)
                q5 = np.arctan2(-y84, temp)
                R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
                                [np.cos(q5), 0, -np.sin(q5)],
                                [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
                R80 = T[:3, :3]
                R40 = R84.T.dot(R80)
                n32 = R40[2, 1]
                n31 = R40[2, 0]
                n33 = R40[2, 2]
                n22 = R40[1, 1]
                n12 = R40[0, 1]
                q2 = np.arcsin(n32)
                q1 = np.arctan2(-n31, n33)
                q4 = np.arctan2(n22, n12)
                joint = [[q1, q2, q3, q4, q5, q6]]
            else:
                x84 = T[:, 0, 3]
                y84 = T[:, 1, 3]
                z84 = T[:, 2, 3]
                q6 = np.arctan2(x84, z84 - dvrkVar.L4)
                temp = -dvrkVar.L3 + np.sqrt(x84 ** 2 + (z84 - dvrkVar.L4) ** 2)
                q3 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(y84 ** 2 + temp ** 2)
                q5 = np.arctan2(-y84, temp)
                R84 = np.zeros((len(T), 3, 3))
                R84[:, 0, 0] = np.sin(q5)*np.sin(q6)
                R84[:, 0, 1] = -np.cos(q6)
                R84[:, 0, 2] = np.cos(q5) * np.sin(q6)
                R84[:, 1, 0] = np.cos(q5)
                R84[:, 1, 1] = 0.0
                R84[:, 1, 2] = -np.sin(q5)
                R84[:, 2, 0] = np.cos(q6) * np.sin(q5)
                R84[:, 2, 1] = np.sin(q6)
                R84[:, 2, 2] = np.cos(q5) * np.cos(q6)
                # R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
                #                 [np.cos(q5), 0, -np.sin(q5)],
                #                 [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
                R80 = T[:, :3, :3]
                R40 = np.matmul(R84.transpose(0, 2, 1), R80)
                n32 = R40[:, 2, 1]
                n31 = R40[:, 2, 0]
                n33 = R40[:, 2, 2]
                n22 = R40[:, 1, 1]
                n12 = R40[:, 0, 1]
                q2 = np.arcsin(n32)
                q1 = np.arctan2(-n31, n33)
                q4 = np.arctan2(n22, n12)
                joint = np.array([q1, q2, q3, q4, q5, q6]).T
        # elif method=='numerical':
        #     q0 = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial guess
        #     ik_sol = self.ikine(T, q0)
        #     joint = [ik_sol[0, 0], ik_sol[0, 1], ik_sol[0, 2], ik_sol[0, 3], ik_sol[0, 3], ik_sol[0, 5]]
        assert ~np.isnan(joint).any()
        return joint

if __name__ == "__main__":
    dvrkkin = dvrkKinematics()

    # 1. random joints
    joints = np.random.uniform(dvrkVar.joint_range_lower_limit, dvrkVar.joint_range_upper_limit)
    print("joints:", joints)

    # 2. forward kinematics <- 결과 동일.
    Tbe = dvrkkin.fk(joints)
    print("Tbe:\n", Tbe)
    print("Tbe_trans\n", Tbe[:3, -1])

    # 3. inverse kinematics(analytic)
    Tb_ed = Tbe.copy()
    joint_ik = dvrkkin.ik(Tb_ed)
    print("ik result:\n", joint_ik)


