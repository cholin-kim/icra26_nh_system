import cv2
import numpy as np
import matplotlib.pyplot as plt


def project_points(P, points_3d):
    """
    3D points를 카메라 P matrix로 2D image plane에 투영
    """
    n = points_3d.shape[0]
    points_h = np.hstack([points_3d, np.ones((n,1))])
    proj = (P @ points_h.T).T  # shape (n,3)
    uv = proj[:, :2] / proj[:, 2:3]
    return uv  # shape (n,2)

def visualize_mask_segment(mask_head, mask_mid, mask_tip, which="L"):
    cv2.namedWindow("mask_head" + which, cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask_mid" + which, cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask_tip" + which, cv2.WINDOW_NORMAL)
    cv2.imshow("mask_head" + which, mask_head)
    cv2.imshow("mask_mid" + which, mask_mid)
    cv2.imshow("mask_tip" + which, mask_tip)


def visualize_subsamples(mask, samples_head, samples_mid, samples_tip, which="L"):
    vis_subsamples = mask.copy()
    vis_subsamples = cv2.cvtColor(vis_subsamples, cv2.COLOR_GRAY2BGR)
    for pt in samples_head:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis_subsamples, (x, y), 2, (0, 0, 255), 2)  # 빨간색 작은 점
    for pt in samples_mid:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis_subsamples, (x, y), 2, (0, 255, 0), 2)  # 빨간색 작은 점
    for pt in samples_tip:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis_subsamples, (x, y), 2, (255, 0, 0), 2)  # 빨간색 작은 점
    cv2.namedWindow("subsamples" + which, cv2.WINDOW_NORMAL)
    cv2.imshow("subsamples" + which, vis_subsamples)


def visualize_intersections(mask, intersection1=None, intersection2=None, which="L"):
    vis_intersection = mask.copy()
    vis_intersection = cv2.cvtColor(vis_intersection, cv2.COLOR_GRAY2BGR)

    if intersection1 is not None:
        x, y = int(round(intersection1[0])), int(round(intersection1[1]))
        cv2.circle(vis_intersection, (x, y), 10, (255, 100, 100), 5)  # 연보라, head -> mid

    if intersection2 is not None:
        x, y = int(round(intersection2[0])), int(round(intersection2[1]))
        cv2.circle(vis_intersection, (x, y), 10, (255, 0, 255), 5)  # 다홍, mid -> tip

    cv2.namedWindow("intersections_" + which, cv2.WINDOW_NORMAL)
    cv2.imshow("intersections_" + which, vis_intersection)


def visualize_reconstructed_samples(
    mask, ellipse, reconstruct_pts, head_2d, tip_2d,
    intersection1=None, intersection2=None, which="L"
):
    vis_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # ellipse
    cv2.ellipse(vis_img, ellipse, (0, 255, 0), 2)

    # reconstructed pts
    for pt in reconstruct_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis_img, (x, y), 2, (0, 0, 255), 2)  # 빨간 점

    # ellipse center
    cv2.circle(vis_img, (int(ellipse[0][0]), int(ellipse[0][1])), 5, (255, 0, 0), -1)

    # head, tip
    # for pt in [head_2d, tip_2d]:
    #     if pt is not None:
    #         x, y = int(round(pt[0])), int(round(pt[1]))
    #         cv2.circle(vis_img, (x, y), 5, (0, 100, 255), 5)
    x, y = int(round(head_2d[0])), int(round(head_2d[1]))
    cv2.circle(vis_img, (x, y), 5, (0, 0, 255), 5)  # Head red

    x, y = int(round(tip_2d[0])), int(round(tip_2d[1]))
    cv2.circle(vis_img, (x, y), 5, (255, 0, 0), 5)  # Tip blue

    # intersections
    if intersection1 is not None:
        x, y = int(round(intersection1[0])), int(round(intersection1[1]))
        cv2.circle(vis_img, (x, y), 5, (255, 100, 100), 5)  # 연보라
    if intersection2 is not None:
        x, y = int(round(intersection2[0])), int(round(intersection2[1]))
        cv2.circle(vis_img, (x, y), 5, (255, 0, 255), 5)  # 다홍

    cv2.namedWindow("reconstructed_" + which, cv2.WINDOW_NORMAL)
    cv2.imshow("reconstructed_" + which, vis_img)
    return vis_img


def visualize_frame_projection(img_L, img_R, frames_se3, P_left, P_right, points_3d, start_3d, center_3d, end_3d):
    overlay_L, overlay_R = img_L.copy(), img_R.copy()

    if len(frames_se3) > 0:
        x_axis = frames_se3[:3, 0]
        y_axis = frames_se3[:3, 1]
        z_axis = frames_se3[:3, 2]

        # 좌표축 projection 및 시각화
        for img, P in zip([overlay_L, overlay_R], [P_left, P_right]):
            origin = project_points(P, center_3d.reshape(1, 3))
            x_pt = project_points(P, (center_3d + 0.02 * x_axis).reshape(1, 3))
            y_pt = project_points(P, (center_3d + 0.02 * y_axis).reshape(1, 3))
            z_pt = project_points(P, (center_3d + 0.02 * z_axis).reshape(1, 3))

            # NaN 체크 후 draw
            if not (np.isnan(origin).any() or np.isnan(x_pt).any() or np.isnan(y_pt).any() or np.isnan(z_pt).any()):
                cv2.line(img, tuple(origin.astype(int).ravel()), tuple(x_pt.astype(int).ravel()), (0, 0, 255), 2)
                cv2.line(img, tuple(origin.astype(int).ravel()), tuple(y_pt.astype(int).ravel()), (0, 255, 0), 2)
                cv2.line(img, tuple(origin.astype(int).ravel()), tuple(z_pt.astype(int).ravel()), (255, 0, 0), 2)

        # 반원 점 projection
        pj_l = project_points(P_left, points_3d)
        pj_r = project_points(P_right, points_3d)

        for pt in pj_l:
            cv2.circle(overlay_L, tuple(pt.astype(int)), 2, (0, 0, 255), -1)
        for pt in pj_r:
            cv2.circle(overlay_R, tuple(pt.astype(int)), 2, (0, 0, 255), -1)

        # Start / End projection
        start_pj_l = project_points(P_left, start_3d.reshape(1, 3))
        end_pj_l = project_points(P_left, end_3d.reshape(1, 3))
        start_pj_r = project_points(P_right, start_3d.reshape(1, 3))
        end_pj_r = project_points(P_right, end_3d.reshape(1, 3))

        cv2.circle(overlay_L, tuple(start_pj_l.astype(int).ravel()), 10, (0, 0, 255), -1)
        cv2.circle(overlay_L, tuple(end_pj_l.astype(int).ravel()), 10, (255, 0, 0), -1)  # Cyan
        cv2.circle(overlay_R, tuple(start_pj_r.astype(int).ravel()), 10, (0, 0, 255), -1)
        cv2.circle(overlay_R, tuple(end_pj_r.astype(int).ravel()), 10, (255, 0, 0), -1)

        return overlay_L, overlay_R


def visualize_needle_3d_live(ax, frames_se3, start_3d, end_3d, points_3d, center_3d):
    ax.cla()  # 이전 그림 지우기

    # 반원 점
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='silver', s=2, label='needle points')

    # start / end / center
    ax.scatter(*start_3d, c='r', s=20, marker='o', label='Head')
    ax.scatter(*end_3d, c='b', s=20, marker='o', label='Tip')
    ax.scatter(*center_3d, c='m', s=20, marker='^', label='Center')

    # local frame (x=red, y=green, z=blue)
    if frames_se3 is not None:
        origin = frames_se3[:3, 3]
        x_axis = frames_se3[:3, 0]
        y_axis = frames_se3[:3, 1]
        z_axis = frames_se3[:3, 2]

        ax.quiver(*origin, *x_axis*0.024, color='r', linewidth=2)
        ax.quiver(*origin, *y_axis*0.024, color='g', linewidth=2)
        ax.quiver(*origin, *z_axis*0.024, color='b', linewidth=2)

    #     # ===== 좌표축 범위 자동 설정 =====
    # all_points = np.vstack([points_3d, start_3d, end_3d, center_3d])
    # x_min, y_min, z_min = np.min(all_points, axis=0)
    # x_max, y_max, z_max = np.max(all_points, axis=0)
    #
    # # margin 추가 (10%)
    # margin_x = 0.1 * (x_max - x_min)
    # margin_y = 0.1 * (y_max - y_min)
    # margin_z = 0.1 * (z_max - z_min)

    x_max, x_min = -0.07, 0.07
    y_max, y_min = -0.07, 0.07
    z_max, z_min = 0.07, 0.18

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # 비율 동일하게 맞추기
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])

    # 그래프 설정
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Needle Visualization")
    ax.legend()
    # plt.axis('equal')