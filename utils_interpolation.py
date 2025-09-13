import numpy as np
from Kinematics.dvrkKinematics import dvrkKinematics
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import RotationSpline


dvrkkin = dvrkKinematics()

class Interpolationutils:
    def __init__(self):
        pass

    @staticmethod
    def get_cart_T_traj(start_T, end_T, duration, visualize=False):
        '''
        start_T, end_T 모두 {rb}
        :param visualize:
        :return: shortest trajectory from start_T to end_T
        '''
        t = np.array([0, duration])
        ts = np.linspace(0, duration, int(duration * 5))

        pos = start_T[:3, -1]
        pos_des = end_T[:3, -1]
        pos_lst = np.array([pos, pos_des])

        cs = []
        pos_traj = []

        for i in range(len(pos)):
            cs.append(CubicSpline(t, pos_lst[:, i], bc_type='clamped'))

            pos_traj_individual = cs[i](ts)
            pos_traj.append(pos_traj_individual)
        pos_traj = np.array(pos_traj).T

        RS = RotationSpline(t, R.from_matrix([start_T[:3, :3], end_T[:3, :3]]))
        # Limitation: The angular acceleration is continuous, but not smooth.

        ori_traj = RS(ts).as_matrix()
        Ts = np.zeros((len(pos_traj), 4, 4))
        Ts[:, -1, -1] = 1
        Ts[:, :3, -1] = pos_traj
        Ts[:, :3, :3] = ori_traj

        if visualize:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Ts[0, 0, -1], Ts[0, 1, -1], Ts[0, 2, -1], s=30, label='start')
            ax.scatter(Ts[-1, 0, -1], Ts[-1, 1, -1], Ts[-1, 2, -1], s=30, label='end')
            for T in Ts:
                position = T[:3, -1]
                orientation = T[:3, :3]
                ax.scatter(position[0], position[1], position[2], s=5, c='black')

                axis_length = 0.03
                x_axis = orientation[:, 0] * axis_length
                y_axis = orientation[:, 1] * axis_length
                z_axis = orientation[:, 2] * axis_length

                ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r',
                          length=axis_length, normalize=True)
                ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g',
                          length=axis_length, normalize=True)
                ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b',
                          length=axis_length, normalize=True)

            # Set labels and display the plot
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_aspect('equal')

            ax.legend()
            plt.show()

        return ts, Ts


    @staticmethod
    def get_cart_q_traj(T_traj):
        """T_traj_rb -> q_traj"""
        q_traj = []
        for T in T_traj:
            q_traj.append(dvrkkin.ik(T))
        return np.array(q_traj)

    @staticmethod
    def get_jacob_traj(q_traj):
        """관절 궤적에 대한 자코비안 궤적 계산"""
        J_traj = []
        for q in q_traj:
            J_traj.append(dvrkkin.jacobian(q))
        return np.array(J_traj)

