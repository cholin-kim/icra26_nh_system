import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_T(pose, scalar_first=False):
    """
    Args:
        pose: [x, y, z, qx, qy, qz, qw]
        scalar_first: True => qw, qx, qy, qz
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    pos = pose[:3]
    quat = pose[3:]
    if scalar_first:
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    T = np.identity(4)
    T[:3, -1] = pos
    T[:3, :3] = R.from_quat(quat).as_matrix()
    return T

def T_to_pose(T, scalar_first=False):
    """
    Args: numpy.ndarray: 4x4 transformation matrix
        pose:
        scalar_first: True => qw, qx, qy, qz
    Returns:
        [x, y, z, qx, qy, qz, qw]
    """
    pos = T[:3, -1]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    if scalar_first:
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    return np.concatenate((pos, quat))


def draw_pose(ax, pose, length=0.05, name=None):
    """
    pose: [x, y, z, qx, qy, qz, qw]
    """
    pos = np.array(pose[:3])
    quat = pose[3:]
    rot = R.from_quat(quat).as_matrix()  # (3, 3)

    # 각 축 방향 단위벡터
    x_axis = rot[:, 0] * length
    y_axis = rot[:, 1] * length
    z_axis = rot[:, 2] * length

    # origin -> 각 축 그리기
    ax.plot([pos[0], pos[0] + x_axis[0]],
            [pos[1], pos[1] + x_axis[1]],
            [pos[2], pos[2] + x_axis[2]], color='r')

    ax.plot([pos[0], pos[0] + y_axis[0]],
            [pos[1], pos[1] + y_axis[1]],
            [pos[2], pos[2] + y_axis[2]], color='g')

    ax.plot([pos[0], pos[0] + z_axis[0]],
            [pos[1], pos[1] + z_axis[1]],
            [pos[2], pos[2] + z_axis[2]], color='b')


    # ax.text(pos[0], pos[1], pos[2], 'None' if name is None else f'{name}', color='black')



def visualize_poses(pose_list, length=0.02):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, pose in enumerate(pose_list):
        draw_pose(ax, pose, length=length, name=f"pose_{i}")

    # draw world coordinate
    # draw_pose(ax, [0, 0, 0, 0, 0, 0, 1], length=length, name="world")

    # 축 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.set_box_aspect([1, 1, 1])  # aspect ratio 1:1:1
    plt.axis('equal')
    ax.grid(True)
    # ax.set_zlim(bottom=0)
    # plt.legend()
    # plt.show()
    return ax


def plot_needle(ax, Tw_needle, radius=0.024, color='c', alpha=0.3):
    """
    Tw_needle: - needle center pose
    radius: 반지름
    """
    # 1. pose 정보 분리
    pos = np.array(Tw_needle[:3, -1])
    rot = Tw_needle[:3, :3]

    x_axis = rot[:, 0]  # head 방향
    y_axis = rot[:, 1]  # 곡률 방향
    z_axis = np.cross(x_axis, y_axis)  # 평면 법선 방향

    # 2. 반원 곡선 좌표 생성 (XY 평면에서 먼저 그린 후, 회전/이동)
    theta = np.linspace(0, np.pi, 100)
    circle_points = np.zeros((3, len(theta)))
    circle_points[0, :] = radius * np.cos(theta)  # x
    circle_points[1, :] = radius * np.sin(theta)  # y
    # z = 0 (원래 평면)

    # 3. local → world 변환
    # local: [x_local; y_local; z_local] → world = R @ local + pos
    needle_points = rot @ circle_points + pos.reshape(3, 1)

    # 3. 그리기
    ax.plot(needle_points[0], needle_points[1], needle_points[2],
            color=color, linewidth=2, alpha=alpha)

    # 5. head point (반원의 시작점: theta = -pi/2)
    head_point = needle_points[:, 0]
    ax.scatter(*head_point, color='r', s=30, label='needle head')
    ax.text(*head_point, 'head', color='r')
    return ax

if __name__ == "__main__":
    # pose = np.array([1, 1, 1, 0, 0, 0, 1])
    # ax = visualize_poses([pose])
    import matplotlib.pyplot as plt
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_needle(ax, pose_to_T([0.0, 0.1, 0.05, 0, 0.707, 0, 0.707]))
    plt.legend()
    plt.axis('equal')
    ax.grid(True)
    plt.show()
    quit()


    Tw_no_init = np.identity(4)
    Tw_no_init[:3, -1] = [0.09858588, 0.04641729, 0]

    Tw_no_new = np.array(
    [[-1.00000000e+00, - 1.22460635e-16 , 0.00000000e+00 , 0.00000000e+00],
     [1.22460635e-16, - 1.00000000e+00 , 0.00000000e+00 , 1.00000000e-01],
    [0.00000000e+00 ,   0.00000000e+00 ,   1.00000000e+00 ,   5.00000000e-02],
    [0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

    Tw_pickkup = np.array(
    [[-5.00000000e-01, - 8.66025404e-01, 6.12303177e-17 , 1.10585878e-01],
       [-8.66025404e-01 , 5.00000000e-01,  1.06054021e-16 , 6.72019029e-02],
    [-1.22460635e-16 ,   0.00000000e+00, - 1.00000000e+00 ,   0.00000000e+00],
    [0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    Tw_pickup_after = np.array(
    [[5.00000000e-01,  8.66025404e-01 ,- 6.12303177e-17, - 1.20000000e-02],
       [8.66025404e-01, - 5.00000000e-01, - 1.06054021e-16,  7.92153903e-02],
    [-1.22460635e-16 ,   0.00000000e+00 ,- 1.00000000e+00  ,  5.00000000e-02],
    [0.00000000e+00  ,0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])



    ax = visualize_poses([T_to_pose(Tw_pickkup), T_to_pose(Tw_pickup_after)])
    plot_needle(ax, Tw_no_new)
    plot_needle(ax, Tw_no_init)
    plt.legend()
    plt.axis('equal')
    ax.grid(True)
    plt.show()



