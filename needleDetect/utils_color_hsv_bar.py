import cv2 as cv

max_value = 255
max_value_H = 255
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

max_value = 255
low_R = 0
low_G = 0
low_B = 0
high_R = max_value
high_G = max_value
high_B = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_R_name = 'Low R'
low_G_name = 'Low G'
low_B_name = 'Low B'
high_R_name = 'High R'
high_G_name = 'High G'
high_B_name = 'High B'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


def on_low_R_thresh_trackbar(val):
    global low_R
    global high_R
    low_R = val
    low_R = min(high_R - 1, low_R)
    cv.setTrackbarPos(low_R_name, window_detection_name, low_R)


def on_high_R_thresh_trackbar(val):
    global low_R
    global high_R
    high_R = val
    high_R = max(high_R, low_R + 1)
    cv.setTrackbarPos(high_R_name, window_detection_name, high_R)


def on_low_G_thresh_trackbar(val):
    global low_G
    global high_G
    low_G = val
    low_G = min(high_G - 1, low_G)
    cv.setTrackbarPos(low_G_name, window_detection_name, low_G)


def on_high_G_thresh_trackbar(val):
    global low_G
    global high_G
    high_G = val
    high_G = max(high_G, low_G + 1)
    cv.setTrackbarPos(high_G_name, window_detection_name, high_G)


def on_low_B_thresh_trackbar(val):
    global low_B
    global high_B
    low_B = val
    low_B = min(high_B - 1, low_B)
    cv.setTrackbarPos(low_B_name, window_detection_name, low_B)


def on_high_B_thresh_trackbar(val):
    global low_B
    global high_B
    high_B = val
    high_B = max(high_B, low_B + 1)
    cv.setTrackbarPos(high_B_name, window_detection_name, high_B)

def main(mode, img):
    if mode == "hsv":
        cv.namedWindow(window_capture_name, flags=cv.WINDOW_NORMAL)
        cv.namedWindow(window_detection_name, flags=cv.WINDOW_NORMAL)
        cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
        cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
        cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
        cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
        cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
        cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
    elif mode == "rgb":
        cv.namedWindow(window_capture_name, flags=cv.WINDOW_NORMAL)
        cv.namedWindow(window_detection_name, flags=cv.WINDOW_NORMAL)
        cv.createTrackbar(low_R_name, window_detection_name, low_R, max_value, on_low_R_thresh_trackbar)
        cv.createTrackbar(high_R_name, window_detection_name, high_R, max_value, on_high_R_thresh_trackbar)
        cv.createTrackbar(low_G_name, window_detection_name, low_G, max_value, on_low_G_thresh_trackbar)
        cv.createTrackbar(high_G_name, window_detection_name, high_G, max_value, on_high_G_thresh_trackbar)
        cv.createTrackbar(low_B_name, window_detection_name, low_B, max_value, on_low_B_thresh_trackbar)
        cv.createTrackbar(high_B_name, window_detection_name, high_B, max_value, on_high_B_thresh_trackbar)

    if mode == "hsv":
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    while True:
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if mode == "hsv":

            frame_threshold = cv.inRange(img, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif mode == "rgb":
            frame_threshold = cv.inRange(img, (low_R, low_G, low_B), (high_R, high_G, high_B))

        cv.imshow(window_capture_name, img)
        cv.imshow(window_detection_name, frame_threshold)

        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            if mode == "hsv":
                print((low_H, low_S, low_V), (high_H, high_S, high_V))
            elif mode == "rgb":
                print((low_R, low_G, low_B), (high_R, high_G, high_B))
            break

if __name__ == "__main__":
    import numpy as np
    # img = np.load("../data_raw/no_occlusion/loose_to_loose/rgb_0.npy")
    # img = cv.imread('../Needle/coppeliasim/saved_stereo_images/image_right_20250718_131223_703498.png')
    # mode = "rgb"
    # main(mode, img)

    img_L_name = "../NeedleDetection/segment-anything-2-real-time/demo/img_left.jpg"
    # img_R_name = "../NeedleDetection/segment-anything-2-real-time/demo/raw_img/img_right.jpg"
    img_L = cv.imread(img_L_name)
    # img_R = cv.imread(img_R_name)
    mode = "hsv"
    main(mode, img_L)


    # from Basler import Basler
    # import time
    #
    # cam_L = Basler(serial_number="40262045")
    # cam_R = Basler(serial_number="40268300")
    # # cam_L.start()
    # cam_R.start()
    # time.sleep(0.1)

    # main(mode, cam_R.image)
    # if mode == "hsv":
    #     cv.namedWindow(window_capture_name, flags=cv.WINDOW_NORMAL)
    #     cv.namedWindow(window_detection_name, flags=cv.WINDOW_NORMAL)
    #     cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    #     cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    #     cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    #     cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    #     cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    #     cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
    # elif mode == "rgb":
    #     cv.namedWindow(window_capture_name, flags=cv.WINDOW_NORMAL)
    #     cv.namedWindow(window_detection_name, flags=cv.WINDOW_NORMAL)
    #     cv.createTrackbar(low_R_name, window_detection_name, low_R, max_value, on_low_R_thresh_trackbar)
    #     cv.createTrackbar(high_R_name, window_detection_name, high_R, max_value, on_high_R_thresh_trackbar)
    #     cv.createTrackbar(low_G_name, window_detection_name, low_G, max_value, on_low_G_thresh_trackbar)
    #     cv.createTrackbar(high_G_name, window_detection_name, high_G, max_value, on_high_G_thresh_trackbar)
    #     cv.createTrackbar(low_B_name, window_detection_name, low_B, max_value, on_low_B_thresh_trackbar)
    #     cv.createTrackbar(high_B_name, window_detection_name, high_B, max_value, on_high_B_thresh_trackbar)
    #
    # while True:
    #     # img = cam_R.image
    #     img = img_L
    #     # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    #     if mode == "hsv":
    #         img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    #         frame_threshold = cv.inRange(img, (low_H, low_S, low_V), (high_H, high_S, high_V))
    #     elif mode == "rgb":
    #         frame_threshold = cv.inRange(img, (low_R, low_G, low_B), (high_R, high_G, high_B))
    #
    #     cv.imshow(window_capture_name, img)
    #     cv.imshow(window_detection_name, frame_threshold)
    #
    #     key = cv.waitKey(30)
    #     if key == ord('q') or key == 27:
    #         if mode == "hsv":
    #             print((low_H, low_S, low_V), (high_H, high_S, high_V))
    #         elif mode == "rgb":
    #             print((low_R, low_G, low_B), (high_R, high_G, high_B))
    #         cam_L.stop()
    #         cam_R.stop()
    #         break

"R (0, 0, 59) (255, 37, 101) or (110, 16, 0) (128, 255, 255) hsv"
"W (0, 0, 124) (255, 80, 255) hsv, 종이 부분 잘라야함."
"B(79, 0, 0) (255, 255, 255) hsv"