import cv2
import numpy as np
import os

# 전역 변수 초기화
image_files = ['sample_img2.jpg']
images = [cv2.imread(f) for f in image_files]
image_index = 0

# HSV 초기값
h_min, s_min, v_min = 0, 0, 0
h_max, s_max, v_max = 179, 255, 255

def nothing(x):
    pass

# Trackbar 창 만들기
cv2.namedWindow('Trackbars')
cv2.createTrackbar('H Min', 'Trackbars', h_min, 180, nothing)
cv2.createTrackbar('H Max', 'Trackbars', h_max, 180, nothing)
cv2.createTrackbar('S Min', 'Trackbars', s_min, 255, nothing)
cv2.createTrackbar('S Max', 'Trackbars', s_max, 255, nothing)
cv2.createTrackbar('V Min', 'Trackbars', v_min, 255, nothing)
cv2.createTrackbar('V Max', 'Trackbars', v_max, 255, nothing)

print("[INFO] 's' 키를 눌러 결과 저장, 'n' 키로 다음 이미지, 'q' 키로 종료")

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
while True:
    img = images[image_index].copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Trackbar 값 읽기
    h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V Max', 'Trackbars')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # 결과 출력
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Segmented', result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        save_name = f"seg_result_{image_files[image_index]}"
        cv2.imwrite(save_name, result)
        print(f"[INFO] 결과 저장됨: {save_name}")

    elif key == ord('n'):
        image_index = (image_index + 1) % len(images)
        print(f"[INFO] 다음 이미지: {image_files[image_index]}")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
