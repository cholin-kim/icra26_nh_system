import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def fit_ellipse(mask: np.ndarray, visualize=False):
    """
    np.uint8 mask (0/255)에서 모든 contour를 합쳐서 ellipse를 fit하는 함수.
    Args:
        mask (np.ndarray): binary mask, dtype=uint8, shape=(H, W)

    Returns:
        (center, axes, angle) 또는 None
        - center: (x, y)
        - axes: (major_axis_length, minor_axis_length)
        - angle: ellipse 회전 각도 (deg)
    """
    # contour 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    if not contours:
        return None

    # 모든 contour 점을 합침
    all_points = np.vstack(contours)

    # ellipse fitting은 최소 5개 점 필요
    if len(all_points) < 5:
        return None

    ellipse = cv2.fitEllipse(all_points)

    if visualize:
        # ellipse_mask = np.zeros_like(mask)
        color_blank = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        color_blank = cv2.ellipse(color_blank, ellipse, color=(0, 255, 0), thickness=1)
        cv2.imshow("ellipse_naive", color_blank)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ellipse  # (center, axes, angle)

def sample_ellipse(ellipse, mask, num_samples=100):
    (cx, cy), (major_axis, minor_axis), theta_deg = ellipse
    theta_rad = np.deg2rad(theta_deg)

    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    h, w = mask_dilated.shape[:2]

    # 0~360도 범위
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)

    pts = []
    angle_lst = []
    for a in angles:
        x = cx + (major_axis/2)*np.cos(a)*np.cos(theta_rad) - (minor_axis/2)*np.sin(a)*np.sin(theta_rad)
        y = cy + (major_axis/2)*np.cos(a)*np.sin(theta_rad) + (minor_axis/2)*np.sin(a)*np.cos(theta_rad)
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < w and 0 <= y_int < h:
            if mask_dilated[y_int, x_int] > 0:
                pts.append((x, y))  # int 아님
                angle_lst.append(a)

    if len(pts) == 0 or len(angle_lst) == 0:
        return None, None
    return np.array(pts), np.array(angle_lst)

def get_intersection_idx(samples, mask_head, mask_mid, mask_tip):
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_head_dilated = cv2.dilate(mask_head, kernel, iterations=1)
    mask_mid_dilated = cv2.dilate(mask_mid, kernel, iterations=1)
    mask_tip_dilated = cv2.dilate(mask_tip, kernel, iterations=1)

    idx_head_lst = []
    idx_mid_lst = []
    idx_tip_lst = []

    for i in range(len(samples)):
        x, y = map(int, np.round(samples[i]))
        if mask_head_dilated[y, x] > 0:
            idx_head_lst.append(i)
        elif mask_mid_dilated[y, x] > 0:
            idx_mid_lst.append(i)
        elif mask_tip_dilated[y, x] > 0:
            idx_tip_lst.append(i)

    return np.array(idx_head_lst), np.array(idx_mid_lst), np.array(idx_tip_lst)



def sample_ellipse_ordered_dual(ellipse_L, ellipse_R, mask_L, mask_R, num_samples=100):
    sample_ellipse_res_L = sample_ellipse(ellipse_L, mask_L, num_samples)
    if sample_ellipse_res_L is None:
        return None, None
    pts_L, angles_L = sample_ellipse_res_L

    sample_ellipse_res_R = sample_ellipse(ellipse_R, mask_R, num_samples)
    if sample_ellipse_res_R is None:
        return None, None
    pts_R, angles_R = sample_ellipse_res_R

    c_pts_L = pts_L - np.asarray(ellipse_L[0])
    c_pts_R = pts_R - np.asarray(ellipse_R[0])

    if pts_L is None or pts_R is None:
        return None, None

    cost_matrix = np.zeros((len(c_pts_L), len(c_pts_R)))
    for i, pt_L in enumerate(c_pts_L):
        for j, pt_R in enumerate(c_pts_R):
            cost_matrix[i, j] = np.linalg.norm(pt_L - pt_R)

    # 헝가리안 알고리즘으로 최적 매칭 찾기
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_L = pts_L[row_ind]
    matched_R = pts_R[col_ind]

    angles_L = angles_L[row_ind]
    angles_R = angles_R[col_ind]

    # return matched_L, matched_R
    return matched_L, matched_R, angles_L, angles_R

def match_pts_again(ellipse_L, ellipse_R, reconstruct_pts_L, reconstruct_pts_R):
    c_pts_L = reconstruct_pts_L - np.asarray(ellipse_L[0])
    c_pts_R = reconstruct_pts_R - np.asarray(ellipse_R[0])

    cost_matrix = np.zeros((len(c_pts_L), len(c_pts_R)))
    for i, pt_L in enumerate(c_pts_L):
        for j, pt_R in enumerate(c_pts_R):
            cost_matrix[i, j] = np.linalg.norm(pt_L - pt_R)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    reconstruct_pts_L = reconstruct_pts_L[row_ind]
    reconstruct_pts_R = reconstruct_pts_R[col_ind]
    return reconstruct_pts_L, reconstruct_pts_R

def label_segments(sorted_indices, idx_head, idx_mid, idx_tip):
    segment_labels = []
    for idx in sorted_indices:
        if idx in idx_head:
            segment_labels.append("head")
        elif idx in idx_mid:
            segment_labels.append("mid")
        elif idx in idx_tip:
            segment_labels.append("tip")

    # 연속 중복 제거
    unique_labels = [segment_labels[0]]
    for lbl in segment_labels[1:]:
        if lbl != unique_labels[-1]:
            unique_labels.append(lbl)
    return segment_labels, unique_labels


def determine_rotation(unique_labels):
    n = len(unique_labels)
    for i in range(n):
        first = unique_labels[i % n]
        second = unique_labels[(i + 1) % n]
        third = unique_labels[(i + 2) % n]

        if first == 'head' and second == 'mid' and third == 'tip':
            return 'CW'
        elif first == 'head' and second == 'tip' and third == 'mid':
            return 'CCW'
    return None


def find_segment_intersections(samples, angles, all_indices, segment_labels, rotation_type,
                               angle_thresh=None):
    """
    samples: 전체 샘플 좌표
    angles: 샘플 각도
    all_indices: 정렬된 sample indices
    segment_labels: 각 sample에 대한 segment label
    rotation_type: 'CW' or 'CCW'
    angle_thresh: gap threshold (optional)

    Returns:
        intersection1, intersection2
    """
    intersection1, intersection2 = None, None

    # angle differences
    angle_diffs = np.abs(np.diff(np.unwrap(angles)))
    for a in range(len(angle_diffs)):
        if 170 < np.rad2deg(angle_diffs[a]) < 190:
            angle_diffs[a] -= np.deg2rad(180)

    gap_idx = np.where(angle_diffs > angle_thresh)[0] if angle_thresh is not None else []

    for i in range(1, len(segment_labels)):
        if len(gap_idx) > 0 and np.any(np.abs(gap_idx - i) <= 2):
            continue  # gap 근처는 무시

        prev_seg = segment_labels[i - 1]
        curr_seg = segment_labels[i]

        if rotation_type == "CCW":
            if prev_seg == "mid" and curr_seg == "head":
                intersection1 = samples[all_indices[i]]  # mid -> head
            if prev_seg == "tip" and curr_seg == "mid":
                intersection2 = samples[all_indices[i]]  # tip -> mid
        elif rotation_type == "CW":
            if prev_seg == "head" and curr_seg == "mid":
                intersection1 = samples[all_indices[i]]  # head -> mid
            if prev_seg == "mid" and curr_seg == "tip":
                intersection2 = samples[all_indices[i]]  # mid -> tip

    return intersection1, intersection2

def point_to_angle_on_ellipse(x, y, ellipse):
    (cx, cy), (major_axis, minor_axis), theta_deg = ellipse
    theta_rad = np.deg2rad(theta_deg)

    # ellipse 좌표계를 정규화 (회전 보정)
    x_rel = x - cx
    y_rel = y - cy

    xr = x_rel * np.cos(theta_rad) + y_rel * np.sin(theta_rad)
    yr = -x_rel * np.sin(theta_rad) + y_rel * np.cos(theta_rad)

    # angle 계산
    angle = np.arctan2(yr / (minor_axis / 2), xr / (major_axis / 2))
    return angle % (2 * np.pi)

def reconstruct_ellipse(ellipse, intersection1, intersection2, rotation_type, step=np.deg2rad(5)):
    (cx, cy), (major_axis, minor_axis), theta_deg = ellipse
    theta_rad = np.deg2rad(theta_deg)

    rev_rotation_type = "CW" if rotation_type == "CCW" else "CCW"
    angles_total = None

    def sample_arc(start_angle, delta_angle, step, direction):
        if direction == "CCW":
            angles = np.arange(start_angle, start_angle + delta_angle + step, step)
        else:  # CW
            angles = np.arange(start_angle, start_angle - delta_angle - step, -step)
        return angles

    if intersection1 is not None and intersection2 is not None:
        # 1. intersection1 기준 반원 재건 (60°)
        a1 = point_to_angle_on_ellipse(*intersection1, ellipse)
        angles1 = sample_arc(a1, np.deg2rad(60), step, rotation_type)
        head_angle = angles1[-1]

        # 2. intersection1 → intersection2 연결
        a2 = point_to_angle_on_ellipse(*intersection2, ellipse)

        arc_angles = sample_arc(a1, np.deg2rad(60), step, rev_rotation_type)

        # 3. intersection2 기준 반원 재건 (60°)
        angles3 = sample_arc(a2, np.deg2rad(60), step, rev_rotation_type)
        tip_angle = angles3[-1]
        angles_total = np.concatenate([angles1, arc_angles, angles3])


    elif intersection1 is not None and intersection2 is None:
        a1 = point_to_angle_on_ellipse(*intersection1, ellipse)
        angles1 = sample_arc(a1, np.deg2rad(60), step, rotation_type)
        angles2 = sample_arc(a1, np.deg2rad(120), step, rev_rotation_type)
        head_angle = angles1[-1]
        tip_angle = angles2[-1]
        angles_total = np.concatenate([angles1, angles2])

    elif intersection2 is not None and intersection1 is None:
        a2 = point_to_angle_on_ellipse(*intersection2, ellipse)
        angles1 = sample_arc(a2, np.deg2rad(120), step, rotation_type)
        angles2 = sample_arc(a2, np.deg2rad(60), step, rev_rotation_type)
        head_angle = angles1[-1]
        tip_angle = angles2[-1]
        angles_total = np.concatenate([angles1, angles2])

    if angles_total is None:
        return None

    x = cx + (major_axis / 2) * np.cos(angles_total) * np.cos(theta_rad) - (minor_axis / 2) * np.sin(
        angles_total) * np.sin(theta_rad)
    y = cy + (major_axis / 2) * np.cos(angles_total) * np.sin(theta_rad) + (minor_axis / 2) * np.sin(
        angles_total) * np.cos(theta_rad)

    head_2d_x = cx + (major_axis / 2) * np.cos(head_angle) * np.cos(theta_rad) - (minor_axis / 2) * np.sin(
        head_angle) * np.sin(theta_rad)
    head_2d_y = cy + (major_axis / 2) * np.cos(head_angle) * np.sin(theta_rad) + (minor_axis / 2) * np.sin(
        head_angle) * np.cos(theta_rad)

    tip_2d_x = cx + (major_axis / 2) * np.cos(tip_angle) * np.cos(theta_rad) - (minor_axis / 2) * np.sin(
        tip_angle) * np.sin(theta_rad)
    tip_2d_y = cy + (major_axis / 2) * np.cos(tip_angle) * np.sin(theta_rad) + (minor_axis / 2) * np.sin(
        tip_angle) * np.cos(theta_rad)

    return np.stack([x, y], axis=1), np.array([head_2d_x, head_2d_y]), np.array([tip_2d_x, tip_2d_y])