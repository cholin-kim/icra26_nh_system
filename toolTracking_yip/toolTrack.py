import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .img_proc_two_shaft import (
    keypoint_segmentation_centroid, key_point_projection_vectorized, key_point_projection, pose2T
)
from .particle_filter import SimpleParticleFilter

class ToolTracker:
    def __init__(self, Tcam_rbBlue: np.ndarray, Tcam_rbYellow: np.ndarray, init_noise_std: float = 0.01):
        self.pf_blue = SimpleParticleFilter(num_particles=200, state_dim=6)
        self.pf_yellow = SimpleParticleFilter(num_particles=200, state_dim=6)

        self.Tcam_rbBlue = Tcam_rbBlue.copy()
        self.Tcam_rbYellow = Tcam_rbYellow.copy()

        # 초기 분산 크게 → 관측이 끌어오도록
        self.pf_blue.predict(noise_std=init_noise_std)
        self.pf_yellow.predict(noise_std=init_noise_std)

        self.lumped_error_blue = np.zeros(6)
        self.lumped_error_yellow = np.zeros(6)

    # --- 안전한 그리기 유틸 ---
    @staticmethod
    def _draw_pts(img, pts, color, r=2, th=1):
        H, W = img.shape[:2]
        for p in pts:
            if not np.isfinite(p).all():
                continue
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(img, (x, y), r, color, th)



    def _compute_likelihood(self, particles, cur_joint, Tcam_rb, obs_pts, jaw_val, img=None, draw=False, which='Left'):
        likelihoods = []
        # 벡터라이즈된 투영 (모든 파티클 한번에)
        projected_pts, _ = key_point_projection_vectorized(cur_joint, particles, Tcam_rb,
                                                      img=img, which=which, jaw=jaw_val)
        # 파티클별 likelihood
        # projected_pts = projected_pts_key
        C_all = np.linalg.norm(projected_pts[:, :, None, :] - obs_pts[None, None, :, :], axis=3)
        likelihoods = []
        threshold = 100
        gamma = 0.1
        for C in C_all:
            row_idx, col_idx = linear_sum_assignment(C)

            # Only keep assignments with a cost below the threshold.
            valid = C[row_idx, col_idx] < threshold
            row_idx = row_idx[valid]
            col_idx = col_idx[valid]

            m = C.shape[0]  # number of projected points for this particle
            likelihood = (5 * np.sum(np.exp(-gamma * C[row_idx, col_idx])) +
                          (m - len(row_idx)) * np.exp(-gamma * threshold))
            likelihoods.append(likelihood)

        return np.array(likelihoods, dtype=float)

    def step(self, img_bgr: np.ndarray,
             joint_blue: np.ndarray, joint_yellow: np.ndarray,
             jaw_blue: float, jaw_yellow: float,
             visualize: bool = True):
        """
        외부에서 매 프레임 호출.
        - img_bgr: 외부 스트리밍 BGR 이미지
        - joint_*, jaw_*: 외부 공급 조인트
        반환: (lumped_error_blue_T(4x4), lumped_error_yellow_T(4x4), debug_img_bgr)
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) 관측 추출
        obs_blue,  img_rgb = keypoint_segmentation_centroid(img_rgb, "Blue",   visualize=visualize)
        obs_yellow, img_rgb = keypoint_segmentation_centroid(img_rgb, "Yellow", visualize=visualize)

        flag_blue   = obs_blue.size   > 0
        flag_yellow = obs_yellow.size > 0

        # (선 검출은 옵션) 현재는 꺼둠
        # shaft_lines, img_rgb = detect_shaft_with_hough(img_rgb, visualize=False)

        noise_std = 0.001

        if flag_blue:
            self.pf_blue.predict(noise_std=noise_std)
        if flag_yellow:
            self.pf_yellow.predict(noise_std=noise_std)

        # 3) 업데이트(관측 중심)
        if flag_blue:
            lh_blue = lambda parts: self._compute_likelihood(
                parts, joint_blue, self.Tcam_rbBlue, obs_blue, jaw_blue,
                img=img_rgb, draw=visualize, which='Left'
            )
            self.pf_blue.update(lh_blue)

        if flag_yellow:
            lh_yel = lambda parts: self._compute_likelihood(
                parts, joint_yellow, self.Tcam_rbYellow, obs_yellow, jaw_yellow,
                img=img_rgb, draw=visualize, which='Left'
            )
            self.pf_yellow.update(lh_yel)

        # 4) 추정치
        lumped_error_blue   = self.pf_blue.estimate()
        lumped_error_yellow = self.pf_yellow.estimate()
        self.lumped_error_blue = pose2T(lumped_error_blue)
        self.lumped_error_yellow = pose2T(lumped_error_yellow)

        # 5) 시각화(예측 키포인트)
        if visualize:
            # Blue
            pts_b, _ = key_point_projection(joint_blue, lumped_error_blue, self.Tcam_rbBlue, jaw=jaw_blue)
            self._draw_pts(img_rgb, pts_b, (255,   0,   0), r=5, th=10)
            # Yellow
            pts_y, _ = key_point_projection(joint_yellow, lumped_error_yellow, self.Tcam_rbYellow, jaw=jaw_yellow)
            self._draw_pts(img_rgb, pts_y, (255, 100, 100), r=5, th=10)

        # 6) 반환 (4x4 변환행렬도 같이)
        return self.lumped_error_blue, self.lumped_error_yellow, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    Tcam_rbBlue = np.array(
        [[0.9525762338306422, -0.1582504221953879, -0.2599140677531933, 0.16217083782196423],
         [-0.19313623593338458, 0.345633233966328, -0.9182788584887827, -0.18783519220885622],
         [0.23515295683982837, 0.9249294413375321, 0.29867811342066364, 0.18348841276653938],
         [0.0, 0.0, 0.0, 1.0]]
    )

    Tcam_rbYellow = np.array(
        [[0.9854955949365622, -0.008485093701868219, -0.16948874754833193, -0.20784290642770953],
         [-0.16022344193643362, 0.28257677370266776, -0.9457689018030958, -0.14153294166489205],
         [0.05591852121322575, 0.95920715705663, 0.2771186547972982, 0.1380827298278568],
         [0.0, 0.0, 0.0, 1.0]]
    )
    import cv2
    import numpy as np
    import time
    from needleDetect.Basler import Basler
    import crtk
    import sys
    sys.path.append("/home/surglab/pycharmprojects")
    from dvrk_surglab.motion.psm import psm

    cam = Basler(serial_number="40262045")
    cam.start()
    time.sleep(0.1)


    ral = crtk.ral(node_name='toolTrack')
    ral.check_connections()
    ral.spin()
    psm1 = psm(ral=ral, arm_name='PSM1')
    psm2 = psm(ral=ral, arm_name='PSM2')

    # 카메라-로봇 변환 행렬
    Tcam_rbBlue = np.array(
        [[0.9525762338306422, -0.1582504221953879, -0.2599140677531933, 0.16217083782196423],
         [-0.19313623593338458, 0.345633233966328, -0.9182788584887827, -0.18783519220885622],
         [0.23515295683982837, 0.9249294413375321, 0.29867811342066364, 0.18348841276653938],
         [0.0, 0.0, 0.0, 1.0]]
    )
    Tcam_rbYellow = np.array(
        [[0.9854955949365622, -0.008485093701868219, -0.16948874754833193, -0.20784290642770953],
         [-0.16022344193643362, 0.28257677370266776, -0.9457689018030958, -0.14153294166489205],
         [0.05591852121322575, 0.95920715705663, 0.2771186547972982, 0.1380827298278568],
         [0.0, 0.0, 0.0, 1.0]]
    )

    tracker = ToolTracker(Tcam_rbBlue, Tcam_rbYellow)

    while True:
        frame = cam.image

        joint_blue = psm2.measured_js()[0]
        joint_yellow = psm1.measured_js()[0]
        jaw_blue = psm2.jaw.measured_js()[0]
        jaw_yellow = psm1.jaw.measured_js()[0]


        TrbBlue_rbBlue, TrbYellow_rbYellow, dbg_img = tracker.step(
            frame, joint_blue, joint_yellow, jaw_blue, jaw_yellow, visualize=True
        )


        # 결과 디스플레이
        cv2.imshow("Tool Tracking", dbg_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

