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

    img_num = 0

    # import os
    # dir_name = "demo"
    # if os.path.isdir(dir_name):
    #     if not os.listdir(dir_name):
    #         print("Directory is empty")
    #     else:
    #         print("Directory is not empty")
    # else:
    #     print("Given directory doesn't exist")
    # quit()

    while True:
        st = time.time()
        cv2.namedWindow('img_left', cv2.WINDOW_NORMAL)
        cv2.namedWindow('img_right', cv2.WINDOW_NORMAL)
        stacked = ImgUtils.stack_stereo_img(cam_L.image, cam_R.image, scale=0.5)
        cv2.imshow('stacked', stacked)
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite(f"img_left_{time.time()}.jpg", cam_L.image)
            cv2.imwrite(f"img_right_{time.time()}.jpg", cam_R.image)
        if k == ord('q'):
            cam_L.stop()
            cam_R.stop()
            break
        # print (time.time() - st)

        # cv2.imwrite(f"demo/CL/demo3/left_scene_{img_num}.png", cam_L.image)
        img_num += 1