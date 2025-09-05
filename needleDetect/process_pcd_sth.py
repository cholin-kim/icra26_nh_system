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
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.003,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])
    # plane_point = np.mean(np.asarray(pcd.points)[inliers], axis=0)  # inlier 점들의 평균 위치를 평면의 한 점으로 사용
    # centroid
    center = np.mean(np.mean(np.asarray(pcd.points)[inliers], axis=0))


    inlier_cloud = pcd.select_by_index(inliers)
    points_3d = np.asarray(inlier_cloud.points)

    # 2. plane 기준점 (여기서는 points_3d[0] 사용)
    p0 = points_3d[0]


    # 3. centroid를 plane에 투영
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # normalize
    projected_centroid = center - np.dot(center - p0, plane_normal) * plane_normal

    # 3. vector on plane (plane_point → centroid_proj)
    tangent = projected_centroid - p0
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(plane_normal, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)

    return plane_normal, tangent, bitangent, projected_centroid, points_3d

def circle_ransac(points_3d, plane_normal, tangent, bitangent, plane_point, target_radius,
                  distance_threshold=0.003, num_iterations=1000, visualize=False):
    """
    points_3d: (N,3) ndarray
    plane_normal: 평면 normal vector
    # plane_point: 평면 위의 한 점
    target_radius: 기준 반지름
    """
    # u, v = plane_basis(plane_normal)
    u, v = tangent, bitangent

    # 3D -> 2D 투영
    def project_to_plane(p):
        vec = p - plane_point
        return np.array([np.dot(vec, u), np.dot(vec, v)])

    points_2d = np.array([project_to_plane(p) for p in points_3d])


    best_inliers = []
    best_center = None
    best_radius = None

    for _ in range(num_iterations):
        # 랜덤 3점 선택
        idx = np.random.choice(len(points_2d), 3, replace=False)
        p1, p2, p3 = points_2d[idx]

        # 2D circle fit: solve linear system
        A = np.array([
            [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
            [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])]
        ])
        b = np.array([
            p2[0] ** 2 + p2[1] ** 2 - p1[0] ** 2 - p1[1] ** 2,
            p3[0] ** 2 + p3[1] ** 2 - p1[0] ** 2 - p1[1] ** 2
        ])
        try:
            center_2d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue

        radius_est = np.linalg.norm(p1 - center_2d)
        dists = np.linalg.norm(points_2d - center_2d, axis=1)

        inliers = np.where(np.abs(dists - target_radius) < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = center_2d
            best_radius = np.mean(dists[inliers])  # 실제 반지름

    # --- Visualization ---
    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        inlier_cloud.paint_uniform_color([0, 1, 0])  # 초록 = inlier
        outlier_cloud.paint_uniform_color([1, 0, 0])  # 빨강 = outlier

        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="RANSAC Circle Result")

    if best_center is not None:
        # 2D -> 3D 변환
        center_3d = plane_point + best_center[0] * u + best_center[1] * v
        return center_3d, best_radius, plane_normal, best_inliers
    else:
        return None, None, plane_normal, []

def compute_needle_frames(start_3d, end_3d, points_3d, center_3d,
                          plane_normal, tangent, rotation_type,
                          neighbor_radius=0.005):

    # 1. x축 = start - center
    x_axis =  center_3d - start_3d
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 2. y축 = plane 내에서 x축을 rotation_type 방향으로 90도 회전
    # plane_normal과 x축의 외적 -> 평면 위 수직 벡터
    if rotation_type == "CCW":
        y_axis = np.cross(plane_normal, x_axis) * (-1)
    elif rotation_type == "CW":
        y_axis = np.cross(plane_normal, x_axis) * (+1)
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