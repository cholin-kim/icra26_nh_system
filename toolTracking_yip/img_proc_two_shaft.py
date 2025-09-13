import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from rl_sth.Kinematics.dvrkKinematics import dvrkKinematics
kin = dvrkKinematics()

rvec = np.zeros((3, 1), dtype=np.float32)
tvec = np.zeros((3, 1), dtype=np.float32)




K_left = np.load("../needleDetect/camera_calibration/cmtx0.npy")
distort_left = np.load("../needleDetect/camera_calibration/dist0.npy")
# K_left = np.load("needleDetect/camera_calibration/cmtx0.npy")
# distort_left = np.load("needleDetect/camera_calibration/dist0.npy")
# K_right = np.load("needleDetect/camera_calibration/cmtx1.npy")
# distort_right = np.load("needleDetect/camera_calibration/dist1.npy")


def draw_line_rho_phi(img, rho, phi, color=(0, 0, 255), thickness=2):
    """
    img:   그림을 그릴 OpenCV BGR 이미지 (numpy array)
    rho:   직선의 정규형에서의 rho
    phi:   직선의 정규형에서의 phi (라디안)
    color: 직선을 그릴 색상 (기본 빨간색)
    thickness: 직선 두께
    """

    # 1) 법선(ρ,φ)에서 점 + 방향벡터 구하기
    #   점 (x0, y0) = (rho*cos(phi), rho*sin(phi))
    #   방향벡터 dir = (-sin(phi), cos(phi))
    a = np.cos(phi)
    b = np.sin(phi)
    x0 = a * rho
    y0 = b * rho

    # 방향벡터(직선에 평행):
    #   dx = -sin(phi)
    #   dy =  cos(phi)
    #   => 이걸 t에 대해 +/-로 늘릴 것
    dx = -b
    dy =  a

    # 2) 이미지 크기 읽기
    height, width = img.shape[:2]

    # 3) 큰 t 범위를 하나 설정해서, parametric form으로 점을 찍어본 뒤,
    #    화면 바깥에 많이 확장하고, 화면 경계를 넘어갈 때까지 조사.
    #    혹은 두 endpoint만 찾아 clip 해도 됨. 여기서는 간단히 예시:

    # t의 최대 한계를 대략 이미지 대각선 길이 정도로 잡아도 충분
    diag_len = np.hypot(width, height)
    # 두 개 포인트 p1(t=-diag_len), p2(t=+diag_len)
    t1 = -diag_len
    t2 =  diag_len

    x1 = x0 + t1 * dx
    y1 = y0 + t1 * dy
    x2 = x0 + t2 * dx
    y2 = y0 + t2 * dy

    # 4) 실제 픽셀 좌표가 float일 수 있으니, int로 변환
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))

    # 5) OpenCV의 line으로 그리기
    cv2.line(img, p1, p2, color, thickness)

    # # 텍스트 위치가 이미지 바깥일 수 있으므로 중앙 근처에 임시로 배치
    # text_x = max(0, min(width - 100, int(round(x0))))  # x 좌표 조정
    # text_y = max(30, min(height - 10, int(round(y0))))  # y 좌표 조정
    #
    # cv2.putText(img, f'{rho:.1f}', (text_x, text_y),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=2,
    #             lineType=cv2.LINE_AA)

    return img





###############################################################################################################################################
def keypoint_segmentation(img, color):
    img_mask = None
    # utils_rgb에서 나온 결과 뒤집어서 넣어야 함.
    if color == "Blue":
        lower = (0, 0, 92)
        upper = (62, 46, 255)
    elif color == "Yellow":
        lower = (23, 99, 0)
        upper = (114, 255, 21)

    "연두색: 33 54 0 114 255 21"
    # 23 99 0 62 255 0
    "파란색: 0 0 92 62 46 255"


    img_mask = cv2.inRange(img, lower, upper) # for original ros_tracking.py (simulation)

    # img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint(8)))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint(8)))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint(8)))


    return img_mask


from scipy.spatial.distance import cdist
def keypoint_segmentation_centroid(img, color, visualize=False):
    img_mask = keypoint_segmentation(img, color)

    cnts, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    centroids = []
    for c in cnts:
        M = cv2.moments(c)
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        centroids.append(np.array([cX, cY]))
    centroids = np.array(centroids, dtype=float)

    # 가까운 centroid 합치기
    merge_dist = 20
    if len(centroids) > 1:
        merged = []
        used = set()
        for i, c in enumerate(centroids):
            if i in used:
                continue
            close_idx = np.where(cdist([c], centroids)[0] < merge_dist)[0]
            merged_centroid = centroids[close_idx].mean(axis=0)
            merged.append(merged_centroid)
            used.update(close_idx)
        centroids = np.array(merged, dtype=int)

    if visualize:
        try:
            for c in centroids:
                cv2.circle(img, c, radius=10, color=(255, 0, 0), thickness=2)
        except:
            pass
    return centroids, img



def pose2T(pose):
    T = np.eye(4)
    T[:3, -1] = pose[:3]
    T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    return T





def shaft_projection(q, p, T_base2cam ,img = None, visualize=False):
    if img is None:
        visualize = False
    q = q[:10]
    Tbs, Ts = kin.fk(q)

    T_br = Tbs[-4]
    Tmp = np.eye(4)
    Tmp[2, -1] -= 0.59421
    T_br = T_br @ Tmp
    T_cr = np.linalg.inv(T_base2cam) @ pose2T(p) @ T_br

    # Cylinder projection
    radius = 0.004
    dir_vec = T_cr[:3, -2]
    center_position = T_cr[:3, -1]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    a, b, c = dir_vec / np.linalg.norm(dir_vec)
    x0, y0, z0 = center_position
    r = radius

    nu = a * x0 + b * y0 + c * z0
    bigC = (x0**2 + y0**2 + z0**2) - (nu**2) - (r**2)

    if bigC <= 1e-12:
        # 가령 카메라에 의해 제대로 투영 안 되거나, degenerate 한 경우
        return None
    sqrtC = np.sqrt(bigC)

    
    
    alpha = c * y0 - b * z0
    beta = a * z0 - c * x0
    kappa = b * x0 - a * y0

    # rx, ry, rz
    rx = r * (x0 - a * nu) / sqrtC 
    ry = r * (y0 - b * nu) / sqrtC 
    rz = r * (z0 - c * nu) / sqrtC 

    def line_params_in_pixel(alpha_sign, beta_sign, kappa_sign):
        """
        unit camera (X,Y)에서의 (A_,B_,C_)를 구하고,
        이를 픽셀 좌표계(u,v)로 변환한 뒤,
        최종 (rho, phi)를 반환
        """
        # A_ = rx ± alpha, B_ = ry ± beta, C_ = rz ± kappa
        A_ = rx + alpha_sign * alpha
        B_ = ry + beta_sign * beta
        C_ = rz + kappa_sign * kappa

        # 픽셀 좌표에서:
        #   Apx*u + Bpx*v + Cpx = 0  (u,v in pixels)
        #   Apx = A_/fx,  Bpx = B_/fy
        #   Cpx = C_ - A_*cx/fx - B_*cy/fy
        Apx = A_ / fx
        Bpx = B_ / fy
        Cpx = C_ - (A_ * cx) / fx - (B_ * cy) / fy

        denom = np.sqrt(Apx * Apx + Bpx * Bpx)
        if denom < 1e-9:
            return None  # degenerate

        # normal form:  Apx*u + Bpx*v + Cpx=0
        #   => sqrt(Apx^2+Bpx^2)* ( (Apx/denom)*u + (Bpx/denom)*v ) + ...
        # rho = -Cpx/denom, phi = atan2(Bpx, Apx)
        rho = - Cpx / denom
        phi = np.arctan2(Bpx, Apx)

        # rho가 음수면, phi를 π만큼 이동해 양수로 만들 수 있음
        if rho < 0:
            rho = -rho
            phi += np.pi
        phi = phi % (2 * np.pi)  # 0~2π 범위

        return (rho, phi)

    # 두 개 Edge (좌/우)에 대해서 각각 sign을 +/-
    line1 = line_params_in_pixel(-1, -1, -1)
    line2 = line_params_in_pixel(+1, +1, +1)
    line = np.array([line1, line2])
    if visualize:
        img = draw_line_rho_phi(img, line1[0], line1[1], color=(0, 0, 255))
        img = draw_line_rho_phi(img, line2[0], line2[1], color=(0, 0, 255))
        return line, img
    return line, img


def key_point_projection(q, p, Tcam_base, img = None, visualize=False, which='Left', jaw=0):
    if img is None:
        visualize = False
    pts_b = get_pts_b(q, jaw)
    pts_b = np.hstack((pts_b, np.ones((np.shape(pts_b)[0], 1))))

    if which == 'Left':
        K = K_left
        D = distort_left
    elif which == 'Right':
        K = K_right
        D = distort_right

    keypoints = []
    for pt_b in pts_b:
        pt_c = Tcam_base @ pose2T(p) @ pt_b
        points_2d, _ = cv2.projectPoints(pt_c[:3], rvec, tvec, K, D)
        keypoints.append(points_2d.reshape(-1).astype(int))
    if visualize:
        for k in keypoints:
            cv2.circle(img, k, radius=2, color=(0, 255, 0), thickness=5)
        return keypoints, img
    return keypoints, img

# def get_pts_b(q):
def get_pts_b(q, jaw=0.0):
    # assert len(q) == 11
    # q1 = q[:10]
    # q2 = q[:10]
    # q2[-1] = q[-1]  # 혹시 모르니까 q[-2]가 jaw1이고 q[-1]이 jaw2인지 확인

    # Tbs1 = kin.fk(q1)[0]
    # Tbs2 = kin.fk(q2)[0]
    # Tbs = Tbs1
    # q[[0, 1, 3]] = -q[[0, 1, 3]]
    Tbs = np.array(kin.fk(q)[0])
    # T_tmp = np.eye(4)
    # T_tmp[:3, :3] = R.from_euler('Z', np.pi).as_matrix()
    # Tbs = T_tmp @ Tbs

    pts_b = []
    T_tmp_tip = np.eye(4)
    T_tmp_tip[:3, :3] = R.from_euler('z', -np.pi/2 + jaw/2).as_matrix()
    T_tmp_trans = np.eye(4)
    T_tmp_trans[:3, -1] = [-0.0095, 0, 0]
    # pts_b.append(Tbs[-1][:3, -1])      # jaw1
    # pts_b.append(Tbs[-1][:3, -1])      # jaw2
    pts_b.append((Tbs[-3] @ T_tmp_tip @ T_tmp_trans)[:3, -1])
    T_tmp_tip2 = np.eye(4)
    T_tmp_tip2[:3, :3] = R.from_euler('z', -np.pi/2 - jaw/2).as_matrix()
    pts_b.append((Tbs[-3] @ T_tmp_tip2 @ T_tmp_trans)[:3, -1])

    T_tmp = np.eye(4)
    #
    T_tmp[:3, -1] = [0., 0., 0.003]
    pts_b.append((Tbs[-3] @ T_tmp)[:3, -1])
    T_tmp[:3, -1] = [0., 0., -0.003]
    pts_b.append((Tbs[-3] @ T_tmp)[:3, -1])

    T_tmp[:3, -1] = [0., 0., 0.004]
    pts_b.append((Tbs[-4] @ T_tmp)[:3, -1])
    T_tmp[:3, -1] = [0., 0., -0.004]
    pts_b.append((Tbs[-4] @ T_tmp)[:3, -1])
    #
    T_tmp[:3, -1] = [0.004, 0., -0.008]
    pts_b.append((Tbs[-5] @ T_tmp)[:3, -1])     # 맨 위의 점
    T_tmp[:3, -1] = [-0.004, 0., -0.008]
    pts_b.append((Tbs[-5] @ T_tmp)[:3, -1])

    pts_b = np.array(pts_b) #+ noise
    return np.array(pts_b)



def remove_second_largest_black_region(img_mask):
    # 배경은 0, 객체는 255로 가정
    img_inv = cv2.bitwise_not(img_mask)  # 검정(0)을 객체로 바꾸기 위해 반전

    # 연결 요소 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_inv, connectivity=8)

    # 각 영역의 넓이 (0번은 전체 배경이라 제외)
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_idx = np.argsort(areas)[::-1]  # 넓이 큰 순서대로 인덱스 정렬

    if len(sorted_idx) < 2:
        return img_mask  # 두 번째 객체가 없으면 원본 반환

    # 두 번째로 큰 객체의 라벨 번호 (1-based이므로 +1)
    second_largest_label = sorted_idx[1] + 1

    # 해당 영역만 흰색(255)으로 변경
    result = img_mask.copy()
    result[labels == second_largest_label] = 255
    return result

def fill_holes_in_largest_white_region(mask):
    # 1. 외곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    # 2. 가장 큰 흰색 영역 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 3. 해당 영역만 흰색으로 채워진 마스크 생성
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 4. 해당 영역 내부에서 flood fill로 구멍 채우기
    # (검정색은 그대로 두고 내부 구멍만 채우기)
    floodfilled = filled.copy()
    h, w = filled.shape
    mask_ff = np.zeros((h+2, w+2), np.uint8)  # floodFill은 2픽셀 더 큰 마스크 요구

    # Flood fill 검정색 내부에서 실행 (구멍 채우기)
    cv2.floodFill(floodfilled, mask_ff, (0, 0), 255)

    # 5. 채워진 구멍 = 외곽 밖 영역이므로 반전해서 AND 연산
    holes_filled = cv2.bitwise_not(floodfilled)
    result = cv2.bitwise_or(filled, holes_filled)

    return result

def fill_holes_in_two_largest_white_regions(mask):
    # 1. 외곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        return mask

    # 2. 면적 기준으로 정렬하여 가장 큰 두 개 선택
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    top_contours = sorted_contours[:2]  # 가장 큰 2개

    # 결과 마스크 초기화
    result = np.zeros_like(mask)

    for contour in top_contours:
        # 3. 컨투어를 채운 마스크 생성
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

        # 4. flood fill로 외부 영역 채우기
        floodfilled = filled.copy()
        h, w = filled.shape
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(floodfilled, mask_ff, (0, 0), 255)

        # 5. 반전해서 내부 구멍 부분만 얻기
        holes_filled = cv2.bitwise_not(floodfilled)

        # 6. 채운 컨투어와 내부 구멍을 합치기
        combined = cv2.bitwise_or(filled, holes_filled)

        # 7. 전체 결과에 추가
        result = cv2.bitwise_or(result, combined)

    return result

def fill_holes_in_two_largest_white_regions(mask):
    # 1. 외곽선과 계층 정보 얻기
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0 or hierarchy is None:
        return mask

    hierarchy = hierarchy[0]  # 계층 구조를 2D로 변경
    contour_areas = []

    # 2. 외부 컨투어만 선택 (hierarchy[i][3] == -1 → 최상위 contour)
    for i, h in enumerate(hierarchy):
        if h[3] == -1:  # 부모가 없는 contour만
            area = cv2.contourArea(contours[i])
            contour_areas.append((area, i))

    # 3. 면적 기준 정렬 후 상위 2개 선택
    contour_areas.sort(reverse=True)
    top_indices = [idx for _, idx in contour_areas[:2]]

    # 4. 마스크 초기화 후 컨투어 채우기 (구멍 없이)
    result = np.zeros_like(mask)
    for idx in top_indices:
        cv2.drawContours(result, contours, idx, 255, thickness=cv2.FILLED)

    return result
#
def keep_largest_hole_and_invert(mask):
    # 1. 마스크 반전 (0 ↔ 255)
    inverted = cv2.bitwise_not(mask)

    # 2. 외곽선 찾기
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask  # 아무것도 없으면 원본 그대로 반환

    # 3. 가장 큰 컨투어 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 4. 가장 큰 컨투어만 그리기
    filtered = np.zeros_like(mask)
    cv2.drawContours(filtered, [largest_contour], -1, 255, thickness=cv2.FILLED)


    # 5. 다시 반전해서 반환
    result = cv2.bitwise_not(filtered)

    return result


# def detect_shaft_with_hough(image, hsv_min=(0, 0, 0), hsv_max=(255, 255, 182), visualize=False):######################################################
#     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     mask = cv2.inRange(hsv, hsv_min, hsv_max)
#
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
#
#     mask = fill_holes_in_two_largest_white_regions(mask)
#
#     # while True:
#     #     cv2.imshow('asdf', mask)
#     #     cv2.waitKey(1)
#
#     # 2. Edge detection (Canny)
#     edges = cv2.Canny(mask, 50, 150, apertureSize=3)
#
#     # 3. Probabilistic Hough Transform으로 직선 검출
#     lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)
#
#     print(lines)
#
#     shaft_lines = []
#     new_lines = []
#     if lines is not None:
#         # for line in lines[:4]:
#         for line in lines:
#             r, theta = line[0]
#             shaft_lines.append([r, theta])
#         for new in lines[-3:]:
#             r, theta = new[0]
#             new_lines.append([r, theta])
#     # shaft_lines = []
#     #
#     # if lines is not None and len(lines) >= 2:
#     #     # lines → [(rho1, theta1), (rho2, theta2), ...]
#     #     lines_list = [line[0] for line in lines]
#     #
#     #     # 모든 쌍 조합에 대해 theta 차이 계산
#     #     min_diff = float('inf')
#     #     best_pair = None
#     #
#     #     for (l1, l2) in combinations(lines_list, 2):
#     #         theta_diff = abs(l1[1] - l2[1])
#     #
#     #         # 각도는 0 ~ pi 이므로, 반대방향 직선도 고려 (예: pi - theta vs theta)
#     #         theta_diff = min(theta_diff, np.pi - theta_diff)
#     #
#     #         if theta_diff < min_diff:
#     #             min_diff = theta_diff
#     #             best_pair = (l1, l2)
#     #
#     #     if best_pair is not None:
#     #         shaft_lines = [list(best_pair[0]), list(best_pair[1])]
#
#     # 4. 시각화 (옵션)
#     if visualize:
#         img_vis = image.copy()
#         for line in shaft_lines:
#             r, theta = line
#             draw_line_rho_phi(img_vis, r, theta, color=(0, 255, 0))
#             # draw_line_rho_phi(mask, r, theta, color=127)
#         for new in new_lines:
#             r, theta = new
#             draw_line_rho_phi(img_vis, r, theta, color=(255,0,0))
#         return shaft_lines, img_vis
#
#     return shaft_lines, image

def detect_shaft_with_hough(image, visualize=False):
    return [], image
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # sobel = cv2.magnitude(sobelx, sobely)
    # sobel = cv2.convertScaleAbs(sobel)
    # edges = cv2.Canny(sobel, 100, 130, apertureSize=3)
    # cv2.imshow("edgeseee", edges)
    # cv2.waitKey(0)
    #
    #
    # # 3. Shaft line detection (Hough Transform)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
    #                         minLineLength=200, maxLineGap=10)
    #
    #
    #
    # shaft_lines = []
    # if lines is not None:
    #     for line in lines[:]:
    #         x1, y1, x2, y2 = line[0]
    #         length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #         angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    #         # if length > 500:
    #         # if abs(angle) > 40 or abs(angle) < 70:  # shaft 각도 범위 필터:
    #         shaft_lines.append([x1, y1, x2, y2])
    #         # r, theta = line[0]
    #
    #         # if r < 0:
    #         #     r = -r
    #         #     theta += np.pi
    #         # theta = theta % (2 * np.pi)  # 0~2π 범위
    #         # shaft_lines.append([r, theta])
    #
    #
    # def lin_classify_xy(arr_xy):
    #     """
    #     arr_xy : [
    #         [x1, y1],
    #         [x2, y2],
    #         ...
    #     ]
    #     (ndarray)
    #     """
    #     mean = np.mean(arr_xy, axis=0).reshape((-1))
    #     coeff = np.linalg.inv(arr_xy.T @ arr_xy) @ arr_xy.T @ np.ones((len(arr_xy), 1))  # return a, b   ... ax + by = 1
    #     a, b = coeff.reshape((-1))
    #     c = b * mean[0] - a * mean[1]
    #     line_div = lambda x: (b * x - c) / a
    #
    #     L1, L2 = [], []
    #     for r, phi in arr_xy:
    #         if phi > line_div(r):
    #             L1.append([r, phi])
    #         else:
    #             L2.append([r, phi])
    #
    #     return [np.array(L1), np.array(L2)]
    #
    # def get_median_line(arr_rp):
    #     L_line = list(arr_rp)
    #     L_line.sort(key=lambda ele: ele[0])
    #     return L_line[int(len(L_line) / 2)]
    #
    # def get_median_line2(arr_rp, ind):
    #     L_line = list(arr_rp)
    #     L_line.sort(key=lambda ele: ele[0])
    #     return L_line[ind]


    # try:
    #     shaft_lines = np.array(shaft_lines)
    #     arr1, arr2 = lin_classify_xy(shaft_lines)
    #     shaft_lines = []
    #     type_l = 1
    #
    #     t1, t2 = lin_classify_xy(arr1)
    #     if type_l == 1:
    #         # mean
    #         shaft_lines.append(np.mean(t1, axis=0))
    #         shaft_lines.append(np.mean(t2, axis=0))
    #     if type_l == 2:
    #         # median
    #         shaft_lines.append(get_median_line(t1))
    #         shaft_lines.append(get_median_line(t2))
    #     if type_l == 3:
    #         shaft_lines.append(get_median_line2(t1, 0))
    #         shaft_lines.append(get_median_line2(t2, 0))
    #
    #     t1, t2 = lin_classify_xy(arr2)
    #     if type_l == 1:
    #         # mean
    #         shaft_lines.append(np.mean(t1, axis=0))
    #         shaft_lines.append(np.mean(t2, axis=0))
    #     elif type_l == 2:
    #         # median
    #         shaft_lines.append(get_median_line(t1))
    #         shaft_lines.append(get_median_line(t2))
    #     if type_l == 3:
    #         shaft_lines.append(get_median_line2(t1, 0))
    #         shaft_lines.append(get_median_line2(t2, 0))
    # except:
    #     return [], image

    # print("Line selection time:", time.time()-st)
    #
    # # 4. 시각화 (옵션)
    # ########################### 수정 #########################
    # for arr in shaft_lines:
    #     if np.isnan(arr).any():
    #         return [], image
    # ########################### 수정 #########################
    #
    # if visualize:
    #     img_vis = image.copy()
    #     for (x1, y1, x2, y2) in shaft_lines:
    #         cv2.line(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     # for line in shaft_lines:
    #     #     r, theta = line
    #     #     draw_line_rho_phi(img_vis, r, theta, color=(0, 255, 0))
    #     # return shaft_lines, img_vis
    #     cv2.imshow("line", img_vis)
    #     cv2.waitKey(0)
    #
    # return shaft_lines, image



def pose2T_vectorized(particles):
    """
    Convert an array of particles (N,6) [x, y, z, roll, pitch, yaw]
    into a batch of 4x4 transformation matrices (N,4,4).
    """
    N = particles.shape[0]
    Ts = np.zeros((N, 4, 4))

    # Unpack translation and rotation components.
    x, y, z = particles[:, 0], particles[:, 1], particles[:, 2]
    roll, pitch, yaw = particles[:, 3], particles[:, 4], particles[:, 5]

    # Precompute trigonometric functions.
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Compute rotation matrices.
    # Here we assume R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = cos_y * cos_p
    R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    R[:, 1, 0] = sin_y * cos_p
    R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    R[:, 2, 0] = -sin_p
    R[:, 2, 1] = cos_p * sin_r
    R[:, 2, 2] = cos_p * cos_r

    # Fill in the transformation matrices.
    Ts[:, :3, :3] = R
    Ts[:, :3, 3] = np.stack([x, y, z], axis=-1)
    Ts[:, 3, 3] = 1.0
    return Ts





# --- Vectorized Key-Point Projection ---
# def key_point_projection_vectorized(q, particles, Tcam_base, img=None, visualize=False, which='Left'):
def key_point_projection_vectorized(q, particles, Tcam_base, img=None, visualize=False, which='Left', jaw=None):
    """
    Projects key points for all particles at once.
    q: current joint state (constant for all particles)
    particles: array of shape (N,6)
    Returns an array of shape (N, M, 2), where M is the number of key points.
    """
    if img is None:
        visualize = False

    # Get key-point positions in the base frame (M,3) and convert to homogeneous (M,4)
    # pts_b = get_pts_b(q)  # Provided function, returns shape (M, 3)
    pts_b = get_pts_b(q, jaw)  # Provided function, returns shape (M, 3)
    pts_b_h = np.hstack((pts_b, np.ones((pts_b.shape[0], 1))))  # (M,4)

    # Compute transformation for each particle: (N, 4,4)
    Ts = pose2T_vectorized(particles)
    # Compute: pt_c = inv(T_base2cam) @ (T_particle @ pts_b_h.T) for each particle.
    # T_inv = np.linalg.inv(T_base2cam)  # constant (4,4)
    pts_c = np.matmul(Tcam_base, np.matmul(Ts, pts_b_h.T))  # (N, 4, M)
    # Use only first 3 coordinates and rearrange to (N, M, 3)
    pts_c = pts_c[:, :3, :].transpose(0, 2, 1)


    # # 보이는 점에 대해서만 계산시키려고 했는데... 이상하게 잘 안되네...
    # L_p_new = []
    # for ind_n in range(len(pts_c)):
    #     v_mean = np.mean(pts_c[ind_n], axis=0)
    #     L_temp = []
    #     for ind_m in range(len(pts_c[ind_n])):
    #         if v_mean[2] > pts_c[ind_n, ind_m, 2]:
    #             L_temp.append(pts_c[ind_n, ind_m])

    #     L_p_new.append(L_temp)
    # pts_c = np.array(L_p_new)

    # Flatten all points for projection.
    N, M, _ = pts_c.shape
    pts_flat = pts_c.reshape(-1, 3)  # (N*M, 3)

    # Use OpenCV’s vectorized projectPoints.
    if which == 'Left':
        K = K_left
        D = distort_left
    elif which == 'Right':
        K = K_right
        D = distort_right
    projected_flat, _ = cv2.projectPoints(pts_flat, rvec, tvec, K, D)
    projected = projected_flat.reshape(N, M, 2).astype(int)  # (N, M, 2)

    pts_c_ = (Tcam_base @ pts_b_h.T)[:3].T
    projected_observation, _ = cv2.projectPoints(pts_c_, rvec, tvec, K, D)
    projected_observation = projected_observation.reshape(-1, 2).astype(int)
    visualize = True
    if visualize and img is not None:
        # If visualization is needed, draw circles for each key point.
        for pt in projected_observation:
            cv2.circle(img, tuple(pt.astype(int)), radius=4, color=(0, 255, 0), thickness=10)
    return projected, img



# --- Vectorized Shaft Projection ---
def shaft_projection_vectorized(q, particles, T_base2cam, img=None, visualize=False):
    """
    Projects shaft lines for all particles.
    Returns an array of shape (N, 2, 2), where each particle gives two line parameters [rho, phi].
    """
    if img is None:
        visualize = False

    # Compute forward kinematics for the constant joint state.
    q = q[:10]
    Tbs, _ = kin.fk(q)  # Assuming Tbs is a list/array of base-to-tool transforms.
    T_br = Tbs[-4]
    Tmp = np.eye(4)
    Tmp[2, -1] -= 0.59421
    T_br = T_br @ Tmp

    # For each particle, compute T = pose2T(p) and then T_cr = inv(T_base2cam) @ T @ T_br.
    Ts = pose2T_vectorized(particles)  # (N, 4,4)
    T_inv = np.linalg.inv(T_base2cam)
    T_cr = np.matmul(T_inv, np.matmul(Ts, T_br))  # (N, 4,4)

    # Extract direction vector (from the second-to-last column) and center position.
    dir_vec = T_cr[:, :3, -2]  # (N,3)
    center_position = T_cr[:, :3, -1]  # (N,3)

    # Normalize the direction vector.
    norm_dir = np.linalg.norm(dir_vec, axis=1, keepdims=True)
    a = dir_vec[:, 0] / norm_dir[:, 0]
    b = dir_vec[:, 1] / norm_dir[:, 0]
    c = dir_vec[:, 2] / norm_dir[:, 0]

    # Center position components.
    x0 = center_position[:, 0]
    y0 = center_position[:, 1]
    z0 = center_position[:, 2]
    r_val = 0.004

    nu = a * x0 + b * y0 + c * z0
    bigC = (x0 ** 2 + y0 ** 2 + z0 ** 2) - (nu ** 2) - (r_val ** 2)
    valid = bigC > 1e-12
    # Use a safe sqrt (avoiding division by zero).
    sqrtC = np.sqrt(np.where(valid, bigC, 1))

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    alpha = c * y0 - b * z0
    beta = a * z0 - c * x0
    kappa = b * x0 - a * y0
    rx = r_val * (x0 - a * nu) / sqrtC
    ry = r_val * (y0 - b * nu) / sqrtC
    rz = r_val * (z0 - c * nu) / sqrtC

    # Helper to compute line parameters for a given sign.
    def compute_line(sign):
        # For sign==1: use plus; for sign==-1: use minus.
        A = np.where(sign == 1, rx + alpha, rx - alpha)
        B = np.where(sign == 1, ry + beta, ry - beta)
        C = np.where(sign == 1, rz + kappa, rz - kappa)
        Apx = A / fx
        Bpx = B / fy
        Cpx = C - (A * cx) / fx - (B * cy) / fy
        denom = np.sqrt(Apx ** 2 + Bpx ** 2)
        rho = - Cpx / denom
        phi = np.arctan2(Bpx, Apx)
        # Adjust negative rho values.
        neg = rho < 0
        rho[neg] = -rho[neg]
        phi[neg] += np.pi
        phi = np.mod(phi, 2 * np.pi)
        return rho, phi

    rho1, phi1 = compute_line(-1)
    rho2, phi2 = compute_line(1)
    # Stack the two lines so that each particle yields a (2,2) array ([rho, phi] for each line).
    lines = np.stack([np.stack([rho1, phi1], axis=-1),
                      np.stack([rho2, phi2], axis=-1)], axis=1)  # (N,2,2)
    # Mark invalid cases with NaN.
    lines[~valid] = np.nan

    if visualize and img is not None:
        for line in lines:
            if np.isnan(line).any():
                continue
            img = draw_line_rho_phi(img, line[0, 0], line[0, 1], color=(0, 0, 255))
            img = draw_line_rho_phi(img, line[1, 0], line[1, 1], color=(0, 0, 255))
    return lines, img
