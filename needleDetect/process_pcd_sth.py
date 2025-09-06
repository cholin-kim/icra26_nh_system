import cv2
import numpy as np
import open3d as o3d

def triangulate(pt_left, pt_right, P_left, P_right):
    """
    Triangulate corresponding 2D points from left and right cameras.

    Parameters:
        pt_left  (np.ndarray): shape (2, N) or (N, 2), left camera pixel coordinates
        pt_right (np.ndarray): shape (2, N) or (N, 2), right camera pixel coordinates
        P_left   (np.ndarray): shape (3, 4), projection matrix of left camera
        P_right  (np.ndarray): shape (3, 4), projection matrix of right camera

    Returns:
        points_3d (np.ndarray): shape (N, 3), triangulated 3D points
    """

    # Convert (N, 2) to (2, N) if needed
    if pt_left.shape[0] != 2:
        pt_left = pt_left.T
    if pt_right.shape[0] != 2:
        pt_right = pt_right.T

    # Triangulate
    points_4d = cv2.triangulatePoints(P_left, P_right, pt_left, pt_right)  # (4, N)

    # Convert from homogeneous to 3D
    valid = points_4d[3, :] != 0
    points_3d = (points_4d[:3, valid] / points_4d[3, valid]).T  # shape (N, 3)
    valid_mask = np.all(np.isfinite(points_3d), axis=1)
    return points_3d[valid_mask]

def plane_to_param(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # 평면 모델 추정
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.002,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])
    plane_point = np.mean(np.asarray(pcd.points)[inliers], axis=0)  # inlier 점들의 평균 위치를 평면의 한 점으로 사용

    points_3d = np.asarray(pcd.points)[inliers]

    # 2. plane 기준점
    p0 = points_3d[0]

    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # normalize

    # 3. vector on plane (plane_point → centroid_proj)
    tangent = plane_point - p0
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(plane_normal, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)

    return plane_normal, tangent, bitangent, plane_point, points_3d



def compute_needle_frames(start_3d, end_3d, points_3d, center_3d,
                          plane_normal, tangent, rotation_type,
                          neighbor_radius=0.005):

    # 1. x축 = start - center
    x_axis =  start_3d - center_3d
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 2. y축
    tmp_vecs = points_3d - center_3d
    tmp_vec = np.mean(tmp_vecs, axis=0) # points_3d가 있는 방향으로의 벡터.
    tmp_normal = np.cross(x_axis, tmp_vec)

    y_axis = np.cross(tmp_normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 3. z축 = x cross y
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 회전 행렬 구성
    R = np.column_stack([x_axis, y_axis, z_axis])

    # SE(3) 구성
    frame_se3 = np.eye(4)
    frame_se3[:3, :3] = R
    frame_se3[:3, 3] = center_3d

    return frame_se3