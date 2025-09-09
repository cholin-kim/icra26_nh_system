import cv2
import numpy as np

def nothing(x):
    pass

# 트랙바 창 생성
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-R", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-G", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-B", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-R", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-G", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-B", "Trackbars", 255, 255, nothing)

# 이미지 로드
import sys
for i in range(len(sys.path)):
    print(sys.path[i])
frame = cv2.imread("surgical_tool_tracking_surglab_Ver/saved_image.jpg")
frame = cv2.imread("surgical_tool_tracking_surglab_Ver/sss.jpg")

if frame is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

while True:
    # 트랙바에서 RGB 범위 읽기
    l_r = cv2.getTrackbarPos("L-R", "Trackbars")
    l_g = cv2.getTrackbarPos("L-G", "Trackbars")
    l_b = cv2.getTrackbarPos("L-B", "Trackbars")
    u_r = cv2.getTrackbarPos("U-R", "Trackbars")
    u_g = cv2.getTrackbarPos("U-G", "Trackbars")
    u_b = cv2.getTrackbarPos("U-B", "Trackbars")

    lower = np.array([l_b, l_g, l_r])  # OpenCV는 BGR 순서!
    upper = np.array([u_b, u_g, u_r])

    # 마스크 및 결과 이미지 생성
    mask = cv2.inRange(frame, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 시각화
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Segmented Result", result)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키
        break

cv2.destroyAllWindows()
