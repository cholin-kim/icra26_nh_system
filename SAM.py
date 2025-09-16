import cv2
import numpy as np
import time





'''
SAM_L, SAM_R 은 실행시 point selector 자동으로 놓고 계속 돌릴 것. 항상 self.mask로 mask 접근할 수 있도록.
'''
import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 현재 폴더 기준 상위.
sys.path.append("/home/surglab/icra26_nh_system/segment-anything-2-real-time")
from sam2.build_sam import build_sam2_camera_predictor


# sam2_checkpoint = "../checkpoints/sam2.1_hiera_base_plus.pt"
sam2_checkpoint = "/home/surglab/icra26_nh_system/segment-anything-2-real-time/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

class SAM:
    def __init__(self, which='L'):

        # CUDA 설정 (선택)
        import torch
        print(torch.cuda.is_available())
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True



        # SAM2 로딩

        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

        # -------- 포인트 프롬프트 선택 UI --------
        self.preview_frame = None  # 첫 프레임(BGR)
        self.roi_window = which + "| Add points: L=positive, R=negative | TAB: switch obj | n: new obj | u: undo | ENTER/SPACE: start | r: reset | q: quit"

        # 객체별 포인트 저장: {obj_id: {"pts": [(x,y),...], "labels": [1/0,...]}}
        self.obj_points = {}
        self.current_obj = 1
        self.obj_points[self.current_obj] = {"pts": [], "labels": []}

        self.initialized = False

        
    def point_selector(self, image):
        # roi_window가 열려있는지 확인 후 없으면 열기
        if not hasattr(self, 'roi_window'):
            self.roi_window = "ROI Selector"
        cv2.namedWindow(self.roi_window)
        cv2.imshow(self.roi_window, image)
        cv2.setMouseCallback(self.roi_window, self.on_mouse)
        cv2.waitKey(1)
        while True:
            self.h, self.w = image.shape[:2]

            # 초기 포인트 입력 단계
            if not self.initialized:
                if self.preview_frame is None:
                    self.preview_frame = image.copy()
                    cv2.namedWindow(self.roi_window, cv2.WINDOW_AUTOSIZE)
                    cv2.setMouseCallback(self.roi_window, self.on_mouse)

                vis = self.draw_overlays_points(self.preview_frame, self.obj_points, self.current_obj)
                cv2.imshow(self.roi_window, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (13, 32):  # ENTER/SPACE -> 확정
                    # 적어도 하나의 객체에 positive가 있어야 함
                    valid_ids = []
                    for oid, data in self.obj_points.items():
                        if 1 in data["labels"]:
                            valid_ids.append(oid)
                    if len(valid_ids) == 0:
                        continue  # 아무 것도 유효하지 않음

                    # SAM2 초기화
                    self.predictor.load_first_frame(image)

                    # 각 객체에 대해 포인트 프롬프트 추가
                    for oid in sorted(valid_ids):
                        pts_xy = np.array(self.obj_points[oid]["pts"], dtype=np.float32) if len(
                            self.obj_points[oid]["pts"]) > 0 else None
                        lbls = np.array(self.obj_points[oid]["labels"], dtype=np.int32) if len(
                            self.obj_points[oid]["labels"]) > 0 else None
                        if pts_xy is None or lbls is None or len(pts_xy) == 0:
                            continue
                        # SAM2: point prompt
                        # add_new_prompt(frame_idx=0, obj_id=oid, points=pts_xy, labels=lbls)
                        _ = self.predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=oid,
                            points=pts_xy,
                            labels=lbls
                        )

                    self.initialized = True
                    cv2.destroyWindow(self.roi_window)
                    break

                elif key == ord('\t'):  # TAB: 객체 전환
                    # 다음 객체 ID로 전환 (없으면 1부터 순환)
                    ids = sorted(self.obj_points.keys())
                    if self.current_obj in ids:
                        idx = ids.index(self.current_obj)
                        self.current_obj = ids[(idx + 1) % len(ids)]
                    else:
                        self.current_obj = ids[0]
                elif key == ord('n'):  # 새 객체 추가
                    new_id = max(self.obj_points.keys()) + 1 if len(self.obj_points) > 0 else 1
                    self.obj_points[new_id] = {"pts": [], "labels": []}
                    self.current_obj = new_id
                elif key == ord('u'):  # 현재 객체 undo
                    if len(self.obj_points[self.current_obj]["pts"]) > 0:
                        self.obj_points[self.current_obj]["pts"].pop()
                        self.obj_points[self.current_obj]["labels"].pop()
                elif key == ord('r'):  # 전체 리셋
                    self.obj_points = {}
                    self.current_obj = 1
                    self.obj_points[self.current_obj] = {"pts": [], "labels": []}
                    self.preview_frame = image.copy()
                elif key == ord('q'):
                    break

        return None

    def segment(self, image):   # rgb
        # -------- 추적 단계 --------
        st = time.time()
        out_obj_ids, out_mask_logits = self.predictor.track(image)
        dt = time.time() - st
        print("SAM Inference:", round(dt, 5))

        # 마스크 합성
        all_mask = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        all_mask[..., 1] = 255
        nobj = len(out_obj_ids)

        # mask_L, 초기화
        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        for i in range(nobj):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255

            # 좌/우 마스크 추출
            mask = cv2.bitwise_or(mask, out_mask)

            # hue = int((i + 3) / (nobj + 3) * 179)
            # all_mask[out_mask[..., 0] == 255, 0] = hue
            # all_mask[out_mask[..., 0] == 255, 2] = 255
        return mask

    def ensure_obj(self, obj_id, obj_points):
        if obj_id not in obj_points:
            obj_points[obj_id] = {"pts": [], "labels": []}

    def on_mouse(self, event, x, y, flags, param):
        if self.preview_frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:  # positive
            self.ensure_obj(self.current_obj, self.obj_points)
            self.obj_points[self.current_obj]["pts"].append((x, y))
            self.obj_points[self.current_obj]["labels"].append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:  # negative
            self.ensure_obj(self.current_obj, self.obj_points)
            self.obj_points[self.current_obj]["pts"].append((x, y))
            self.obj_points[self.current_obj]["labels"].append(0)



    def draw_overlays_points(self, img_rgb, obj_points, current_obj):
        vis = img_rgb.copy()
        h, w = vis.shape[:2]

        # 객체별 고유 색상 (HSV 기반)
        def color_for_id(i):
            hue = int((i + 3) / (len(obj_points) + 3) * 179)  # OpenCV HSV hue range: 0-179
            col = np.uint8([[[hue, 255, 255]]])
            rgb = cv2.cvtColor(col, cv2.COLOR_HSV2RGB)[0, 0].tolist()
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        # 점 그리기
        for oid, data in obj_points.items():
            col = color_for_id(oid)
            for (x, y), lab in zip(data["pts"], data["labels"]):
                if lab == 1:
                    cv2.circle(vis, (int(x), int(y)), 5, col, -1)  # positive: filled circle
                else:
                    cv2.circle(vis, (int(x), int(y)), 6, col, 2)  # negative: hollow circle
            # 객체 라벨 표시
            if len(data["pts"]) > 0:
                cx, cy = data["pts"][-1]
                cv2.putText(vis, f"ID {oid}", (int(cx) + 8, int(cy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # 헤더 바
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
        txt = f"[Current Obj: {current_obj}]  L: +  R: -  TAB: switch  n: new  u: undo  ENTER/SPACE: start  r: reset  q: quit"
        cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return vis


