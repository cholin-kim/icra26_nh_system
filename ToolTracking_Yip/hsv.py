import cv2
import numpy as np
import os

def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H Lower", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("H Upper", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S Lower", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("S Upper", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V Lower", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V Upper", "Trackbars", 255, 255, nothing)

def get_hsv_bounds():
    h_lower = cv2.getTrackbarPos("H Lower", "Trackbars")
    h_upper = cv2.getTrackbarPos("H Upper", "Trackbars")
    s_lower = cv2.getTrackbarPos("S Lower", "Trackbars")
    s_upper = cv2.getTrackbarPos("S Upper", "Trackbars")
    v_lower = cv2.getTrackbarPos("V Lower", "Trackbars")
    v_upper = cv2.getTrackbarPos("V Upper", "Trackbars")
    lower = np.array([h_lower, s_lower, v_lower])
    upper = np.array([h_upper, s_upper, v_upper])
    return lower, upper

def main():
    index = 1
    create_trackbars()

    while True:
        filename = f"/home/surglab/GotYourBack/Tool_Tracking/SurgRIPE/lnd_train/TRAIN/image/{index}.png"
        filename = f"/home/surglab/GotYourBack/Tool_Tracking/SurgRIPE/lnd_test_occ/TEST_occ/image/{index}.png"
        if not os.path.exists(filename):
            print(f"{filename} 파일이 존재하지 않습니다.")
            break

        image = cv2.imread(filename)
        if image is None:
            print(f"{filename} 이미지를 불러올 수 없습니다.")
            break

        while True:
            lower, upper = get_hsv_bounds()

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            cv2.imshow("Original", image)
            cv2.imshow("HSV Mask", mask)
            cv2.imshow("Segmented Result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                index += 1
                break
            elif key == 27:  # ESC 키
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
