import time
import numpy as np
import cupy as cp
from scipy import ndimage
from pyueye import ueye
from threading import Thread
import cv2


class UeyeCamera(Thread):

    def __init__(self, k, live):
        super().__init__()

        self.hCam3 = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID

        nret = ueye.is_InitCamera(self.hCam3, None)
        if nret != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        nret = ueye.is_SetColorMode(self.hCam3, ueye.IS_CM_BGR8_PACKED)
        if nret != ueye.IS_SUCCESS:
            print("is_SetColorMode ERROR")

        # set AOI
        fr_x1 = 642
        fr_y1 = 920
        self.fr_w = 808
        self.fr_h = 90
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(fr_x1)
        rect_aoi.s32Y = ueye.int(fr_y1)
        rect_aoi.s32Width = ueye.int(self.fr_w)
        rect_aoi.s32Height = ueye.int(self.fr_h)

        nret = ueye.is_AOI(self.hCam3, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        if nret != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        # allocate memory
        self.mem_ptr3 = ueye.c_mem_p()
        self.mem_id3 = ueye.int()

        self.bitspixel = 24  # for colormode = IS_CM_BGR8_PACKED
        self.lineinc = self.fr_w * int((self.bitspixel + 7) / 8)

        nret = ueye.is_AllocImageMem(self.hCam3, self.fr_w, self.fr_h, self.bitspixel, self.mem_ptr3, self.mem_id3)
        if nret != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")

        # set active memory region
        nret = ueye.is_SetImageMem(self.hCam3, self.mem_ptr3, self.mem_id3)
        if nret != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")

        # use trigger from DMD
        nret = ueye.is_SetExternalTrigger(self.hCam3, ueye.IS_SET_TRIGGER_LO_HI)
        if nret != ueye.IS_SUCCESS:
            print("is_SetExternalTrigger ERROR")

        # set framerate to just above trigger framerate
        framerate = 70
        actual_fr = ueye.DOUBLE(0.)
        nret = ueye.is_SetFrameRate(self.hCam3, ueye.DOUBLE(framerate), actual_fr)
        if nret != ueye.IS_SUCCESS:
            print("is_SetFrameRate ERROR")

        # set exposure time
        exposure = 2
        nret = ueye.is_Exposure(self.hCam3, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.DOUBLE(exposure),
                                ueye.sizeof(ueye.DOUBLE(exposure)))
        if nret != ueye.IS_SUCCESS:
            print("is_Exposure ERROR")

        # set gain
        gain = 0
        nret = ueye.is_SetHardwareGain(self.hCam3, ueye.INT(gain), ueye.INT(gain), ueye.INT(gain), ueye.INT(gain))
        if nret != ueye.IS_SUCCESS:
            print("is_GAIN ERROR")

        # set gamma to 1
        nret = ueye.is_Gamma(self.hCam3, ueye.IS_GAMMA_CMD_SET, ueye.INT(100), ueye.sizeof(ueye.INT(100)))
        if nret != ueye.IS_SUCCESS:
            print("is_Gamma ERROR")

        s_info = ueye.SENSORINFO()
        c_info = ueye.CAMINFO()
        nret = ueye.is_GetCameraInfo(self.hCam3, c_info)
        if nret != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
        nret = ueye.is_GetSensorInfo(self.hCam3, s_info)
        if nret != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")
        print("Camera model:\t\t", s_info.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", c_info.SerNo.decode('utf-8'))

        init_event = ueye.IS_INIT_EVENT()
        init_event.nEvent = ueye.IS_SET_EVENT_FRAME
        init_event.bManualReset = ueye.BOOL(False)
        init_event.bInitialState = ueye.BOOL(False)

        nret = ueye.is_Event(self.hCam3, ueye.IS_EVENT_CMD_INIT, init_event,
                             ueye.sizeof(init_event))
        if nret != ueye.IS_SUCCESS:
            print("IS_EVENT_CMD_INIT ERROR")

        nret = ueye.is_Event(self.hCam3, ueye.IS_EVENT_CMD_ENABLE, init_event,
                             ueye.sizeof(init_event))
        if nret != ueye.IS_SUCCESS:
            print("IS_EVENT_CMD_ENABLE ERROR")

        self.new_frame_event = ueye.IS_WAIT_EVENT()
        self.new_frame_event.nEvent = ueye.IS_SET_EVENT_FRAME
        self.new_frame_event.nTimeoutMilliseconds = ueye.UINT(1000)

        # continuous capture to memory
        nret = ueye.is_CaptureVideo(self.hCam3, ueye.IS_DONT_WAIT)
        if nret != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        self.frames = []
        self.ampls = []
        self.daemon = True
        self.t0 = time.perf_counter()
        self.t1 = time.perf_counter()

        spot_indxs = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/ONN/tools/spot_indxs2.npy")
        self.y_center = spot_indxs[-1]
        self.spot_indxs = spot_indxs[:-1]
        self.k = k
        self.area_width = self.spot_indxs.shape[0]//k

        self.live = live


    def run(self):

        self.t0 = time.perf_counter()

        while True:
            nret = ueye.is_Event(self.hCam3, ueye.IS_EVENT_CMD_WAIT, self.new_frame_event,
                                 ueye.sizeof(self.new_frame_event))
            if nret == ueye.IS_SUCCESS:
                ueye_data = ueye.get_data(self.mem_ptr3, self.fr_w, self.fr_h, self.bitspixel, self.lineinc, copy=True)
                self.process(ueye_data)
            else:
                print(nret)

    def process(self, data):

        frame = np.reshape(data, (self.fr_h, self.fr_w, 3))[..., 0].astype(np.uint8)
        frame = ndimage.rotate(frame, -0.2, reshape=False).T
        mask = frame < 4
        frame -= 3
        frame[mask] = 0

        self.frames.append(frame.T)
        self.frames = self.frames[-20:]

        if self.live:
            cv2.imshow('ueye', frame.T)
            cv2.waitKey(5)

        spots = frame[self.spot_indxs, self.y_center-2:self.y_center+3].copy()
        powers = spots.reshape((self.k, self.area_width, 5)).mean(axis=(1, 2))
        ampls = np.sqrt(powers)

        self.ampls.append(ampls)
        self.ampls = self.ampls[-20:]

        # self.t1 = time.perf_counter()
        # print(f'{self.t1 - self.t0:.4f}')
        # self.t0 = self.t1

    def stop(self):
        ueye.is_StopLiveVideo(self.hCam3, ueye.IS_FORCE_VIDEO_STOP)


if __name__ == "__main__":
    cam = UeyeCamera()
    cam.start()

    for i in range(100):
        time.sleep(5/60)
        print('tick')
