import numpy as np


def compute_intrinsics(fov_y_deg, resolution):
    """
    fov_y_deg: 세로 방향 FOV (degree 단위)
    resolution: (width, height)
    """
    width, height = resolution
    aspect_ratio = width / height

    # FOV_y를 라디안으로 변환
    fov_y_rad = np.deg2rad(fov_y_deg)

    # FOV_x 계산
    fov_x_rad = 2 * np.arctan(np.tan(fov_y_rad / 2) * aspect_ratio)

    # focal length 계산 (pixel 단위)
    fx = (width / 2) / np.tan(fov_x_rad / 2)
    fy = (height / 2) / np.tan(fov_y_rad / 2)

    # principal point (이미지 중심)
    cx = width / 2
    cy = height / 2

    # Intrinsic matrix 구성
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    np.save('imgs/K.npy', K)
    return K
