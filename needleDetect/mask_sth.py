import cv2
import numpy as np
from sklearn.cluster import KMeans


def generate_segment_masks(img, mask):
    """
    img: rgb
    segment_ranges: dict { "head": ((low1, low2, low3), (high1, high2, high3)), ... }
    """
    segment_ranges = {
        # "head": ((0, 0, 59), (255, 37, 101)),
        "head": ((0, 174, 98), (38, 255, 255)),
        # "mid": ((0, 0, 124), (255, 80, 255)),
        "mid": ((31, 0, 157), (255, 34, 255)),
        # "tip": ((79, 0, 0), (255, 255, 255))
        "tip": ((28, 26, 147), (134, 140, 242))
    }

    masks = {}
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for key, (low, high) in segment_ranges.items():
        # if key == "head":  # BGR 기준
        #     mask_seg = cv2.inRange(img, low, high)
        # else:  # HSV 기준
        mask_seg = cv2.inRange(img_hsv, low, high)

        mask_seg = cv2.bitwise_and(mask, mask_seg)
        mask_seg = _morph_open(mask_seg)
        mask_seg = _morph_close(mask_seg)
        mask_seg = _sharpen_mask_edges(mask_seg)
        masks[key] = mask_seg
    return masks


def _gaussian_blur(mask: np.ndarray):
    kernel = (5, 5)
    mask_processed = cv2.GaussianBlur(mask, ksize=kernel, sigmaX=0)
    return mask_processed

def _morph_open(mask: np.ndarray, visualize=False):
    kernel = (5, 5)   # how can I find proper kernel size?
    mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if visualize:
        h, w = mask.shape
        merged = np.zeros((h, w, 3), dtype=np.uint8)
        merged[..., 0] = mask  # B
        merged[..., 1] = mask_processed  # G
        img_stacked = np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), merged, cv2.cvtColor(mask_processed, cv2.COLOR_GRAY2BGR)])
        win_width, win_height = int(w/3 * 3), int(h/3)
        img_resized = cv2.resize(img_stacked, (win_width, win_height))
        cv2.imshow("small_noise_removed, overlap", img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_processed

def _morph_close(mask: np.ndarray, visualize=False):
    kernel = (5, 5)
    mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if visualize:
        h, w = mask.shape
        merged = np.zeros((h, w, 3), dtype=np.uint8)
        merged[..., 1] = mask  # G
        merged[..., 2] = mask_processed  # R
        img_stacked = np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), merged, cv2.cvtColor(mask_processed, cv2.COLOR_GRAY2BGR)])
        win_width, win_height = int(w/3 * 3), int(h/3)
        img_resized = cv2.resize(img_stacked, (win_width, win_height))
        cv2.imshow("small_hole_filled, overlap", img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mask_processed



def _sharpen_mask_edges(mask, visualize=False):
    alpha = 0.7 # alpha 값이 클수록 edge 강조
    lap = cv2.Laplacian(mask, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)  # convert dtype float64 -> uint8

    # Sharpening (원본 - alpha*lap)
    sharp = cv2.subtract(mask, cv2.multiply(lap, np.array([alpha], dtype=np.float32)))

    # Threshold 다시 적용해서 binary 보정
    _, sharp_bin = cv2.threshold(sharp, 127, 255, cv2.THRESH_BINARY)

    if visualize:
        img_stacked = np.hstack([mask, lap, sharp_bin])
        cv2.imshow("Original | Laplacian | Sharpened", img_stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sharp_bin



def cluster_mask(mask: np.ndarray, n_clusters=15):
    """
    Binary mask를 clustering해서 중심점들로 이루어진 mask를 반환하는 함수.

    Args:
        mask (np.ndarray): binary mask (0,255 or 0,1)
        n_clusters (int): 클러스터 개수 (KMeans 기준)

    Returns:
        new_mask (np.ndarray): cluster centroid들이 찍힌 binary mask
        centers (np.ndarray): centroid 좌표 (x,y)
    """
    # mask 이진화 (0,1)
    mask_bin = (mask > 0).astype(np.uint8)

    # nonzero pixel 좌표 추출 (y,x 순서)
    points = np.column_stack(np.where(mask_bin > 0))

    if len(points) == 0:
        return None

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    centers = kmeans.cluster_centers_.astype(int)  # (y,x)

    # 새로운 mask 생성
    new_mask = np.zeros_like(mask_bin)
    for y, x in centers:
        cv2.circle(new_mask, (x, y), radius=1, color=255, thickness=-1)  # 중심 찍기

    return new_mask
