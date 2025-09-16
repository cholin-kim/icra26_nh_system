'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
import threading
from pypylon import pylon
import cv2


class Basler(threading.Thread):
    def __init__(self, serial_number):
        threading.Thread.__init__(self)
        self.stop_flag = False

        # Get the transport layer factory.
        tlf = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlf.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        for d in devices:
            if d.GetSerialNumber() == serial_number:
                print(f"Model= {d.GetModelName()}, Serial= {d.GetSerialNumber()}")
                self.cam = pylon.InstantCamera(tlf.CreateDevice(d))
                self.cam.Open()

        # Grabing Continusely (video) with minimal delay
        self.cam.StartGrabbing()

        # converting to opencv bgr format
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.image = []

    def __del__(self):
        self.stop()

    def run(self):
        while self.cam.IsGrabbing():
            if self.stop_flag:
                print ("stop flag detected")
                break
            grabResult = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                self.image = image.GetArray()

        # Releasing the resource
        self.cam.StopGrabbing()
        self.cam.Close()
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_flag = True


if __name__ == '__main__':
    import cv2
    import time
    from ImgUtils import ImgUtils
    cam_L = Basler(serial_number="40262045")
    cam_R = Basler(serial_number="40268300")
    cam_L.start()
    cam_R.start()
    time.sleep(0.1)


    import os
    # === 저장 폴더 설정 ===
    os.makedirs("frames_left", exist_ok=True)
    os.makedirs("frames_right", exist_ok=True)

    fps = 20
    frame_interval = 1.0 / fps
    frame_idx = 0

    try:
        while True:
            start_time = time.time()

            # 프레임 읽기
            frame_L = cam_L.image
            frame_R = cam_R.image

            if frame_L is None or frame_R is None:
                print("Frame not captured.")
                continue

            # 이미지 저장
            left_path = f"frames_left/frame_{frame_idx:05d}.jpg"
            right_path = f"frames_right/frame_{frame_idx:05d}.jpg"
            cv2.imwrite(left_path, frame_L)
            cv2.imwrite(right_path, frame_R)
            print(f"Saved {left_path}, {right_path}")

            frame_idx += 1

            # 미리보기 (옵션)
            stacked = ImgUtils.stack_stereo_img(frame_L, frame_R, scale=0.5)
            cv2.imshow('Stereo', stacked)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            # 20Hz 유지
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))

    finally:
        cam_L.stop()
        cam_R.stop()
        cv2.destroyAllWindows()
        print("Image capture finished.")

    # while True:
    #     st = time.time()
        # cv2.namedWindow('img_left', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('img_right', cv2.WINDOW_NORMAL)
        # stacked = ImgUtils.stack_stereo_img(cam_L.image, cam_R.image, scale=0.5)
        # cv2.imshow('stacked', stacked)
        # k = cv2.waitKey(1)
        # if k == ord('s'):
        #     cv2.imwrite(f"img_left_{time.time()}.jpg", cam_L.image)
        #     cv2.imwrite(f"img_right_{time.time()}.jpg", cam_R.image)
        # if k == ord('q'):
        #     cam_L.stop()
        #     cam_R.stop()
        #     break
        # print (time.time() - st)
