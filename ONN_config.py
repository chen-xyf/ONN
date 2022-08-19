import time
import os
import numpy as np
import cupy as cp
from scipy.optimize import curve_fit
from scipy import ndimage
from termcolor import colored
from pypylon import pylon
from pyueye import ueye
from glumpy import app
from glumpy.app import clock
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm_images import make_slm1_rgb, make_slm2_rgb, make_dmd_batch, update_params
from threading import Thread
import ueye_thread
import matplotlib.pyplot as plt
from PIL import Image
import cv2


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class CaptureProcess(pylon.ImageEventHandler):

    def __init__(self, window, live, m, back=False):
        super().__init__()

        self.window = window
        self.live = live
        self.frames = []
        self.ampls = []
        self.m = m
        self.back = back
        if back:
            spot_indxs = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/ONN/tools/"
                                 "spot_indxs_back.npy")
        else:
            spot_indxs = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/ONN/tools/"
                                 "spot_indxs.npy")
        self.y_center = spot_indxs[-1]
        self.spot_indxs = spot_indxs[:-1]
        self.area_width = self.spot_indxs.shape[0]//m

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            if self.live:
                self.window.SetImage(grab_result)
            # self.window.Show()

            image = grab_result.GetArray().T

            mask = image < 4
            image -= 3
            image[mask] = 0

            spots = image[self.spot_indxs, self.y_center-2:self.y_center+3].copy()
            powers = spots.reshape((self.m, self.area_width, 5)).mean(axis=(1, 2))
            ampls = np.sqrt(powers)
            # if self.back:
            #     ampls = np.flip(ampls)

            self.frames.append(image.T)
            self.frames = self.frames[-20:]  # only store the 20 most recent frames

            self.ampls.append(ampls)
            self.ampls = self.ampls[-20:]  # only store the 20 most recent frames


class Controller:

    def __init__(self,  _n, _m, _k, _num_frames=5, _scale_guess=2, _ref_guess=0,
                 use_pylons=True, use_ueye=True, live=False):

        self.n = _n
        self.m = _m
        self.k = _k
        self.num_frames = _num_frames
        self.scale_guess = _scale_guess
        self.ref_guess = _ref_guess
        self.live = live

        #################
        # PYLON CAMERAS #
        #################

        if use_pylons:

            # os.environ["PYLON_CAMEMU"] = "2"
            tlfactory = pylon.TlFactory.GetInstance()
            devices = tlfactory.EnumerateDevices()
            assert len(devices) == 2
            self.cameras = pylon.InstantCameraArray(2)
            for i, camera in enumerate(self.cameras):
                camera.Attach(tlfactory.CreateDevice(devices[i]))
                camera.Open()

            pylon.FeaturePersistence.Load("./tools/MVM v3/pylon_settings_a1.pfs", self.cameras[0].GetNodeMap())
            pylon.FeaturePersistence.Load("./tools/MVM v3/pylon_settings_back.pfs", self.cameras[1].GetNodeMap())

            self.imageWindow1 = pylon.PylonImageWindow()
            self.imageWindow1.Create(1)
            self.imageWindow1.Show()

            self.imageWindow3 = pylon.PylonImageWindow()
            self.imageWindow3.Create(3)
            self.imageWindow3.Show()

            self.capture1 = CaptureProcess(self.imageWindow1, self.live, self.m)
            self.capture3 = CaptureProcess(self.imageWindow3, self.live, self.m, back=True)

            self.cameras[0].RegisterImageEventHandler(self.capture1, pylon.RegistrationMode_ReplaceAll,
                                                      pylon.Cleanup_None)
            self.cameras[1].RegisterImageEventHandler(self.capture3, pylon.RegistrationMode_ReplaceAll, pylon.
                                                      Cleanup_None)

            self.cameras.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
            time.sleep(1)
            print('pylon cameras started')

        ###############
        # UEYE CAMERA #
        ###############

        if use_ueye:

            self.capture2 = ueye_thread.UeyeCamera(self.k, self.live)
            self.capture2.start()

        #######
        # SLM #
        #######
        #
        # self.y_centers = np.load('./tools/y_centers_list.npy')
        # self.x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
        #
        # self.y_centers2 = np.load('./tools/y_centers_list2.npy')
        # self.x_edge_indxs2 = np.load('./tools/x_edge_indxs2.npy')
        #
        # self.y_centers3 = np.load('./tools/y_centers_list_back.npy')
        # self.x_edge_indxs3 = np.load('./tools/x_edge_indxs_back.npy')

        self.dmd_block_w, self.dmd_err_block_w = update_params(self.n, self.m, self.k, self.num_frames)

        ampl_lut_slm1 = np.load("./tools/ampl_lut.npy")
        self.gpu_ampl_lut_slm1 = cp.array(ampl_lut_slm1, dtype=cp.float16)
        ampl_lut_slm2 = np.load("./tools/ampl_lut_2.npy")
        self.gpu_ampl_lut_slm2 = cp.array(ampl_lut_slm2, dtype=cp.float16)

        self.slm = SLMdisplay([-1920, -3*1920],
                              [1920, 1920],
                              [1080, 1152],
                              ['SLM1', 'SLM2'],
                              isImageLock=False)

        #######
        # DMD #
        #######

        self.backend = app.use('glfw')
        self.window = app.Window(1920, 1080, fullscreen=0, decoration=0)
        self.window.set_position(-2*1920, 0)
        self.window.activate()
        self.window.show()

        self.dmd_clock = clock.Clock()

        @self.window.event
        def on_draw(dt):
            window_on_draw(self.window, self.screen, self.cuda_buffer, self.cp_arr)
            self.frame_count += 1
            self.cp_arr = self.target_frames[self.frame_count % len(self.target_frames)]

        self.screen, self.cuda_buffer, self.context = setup(1920, 1080)

        self.null_frames = cp.zeros((self.num_frames + 4, 1080, 1920, 4), dtype=cp.uint8)
        self.null_frames[..., -1] = 255

        print('started slm windows')



        ###############

        self.target_frames = None
        self.fc = None
        self.cp_arr = None
        self.frame_count = None

        self.frames1 = None
        self.ampls1 = None
        self.a1s = None
        self.norm_params1 = None

        self.frames2 = None
        self.ampls2 = None
        self.z2s = None
        self.norm_params2 = None

        self.frames3 = None
        self.ampls3 = None
        self.d2s = None
        self.norm_params3 = None

        self.init_dmd()

        print('setup complete')

    def init_dmd(self):

        self.target_frames = cp.zeros((self.num_frames+3, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[..., -1] = 255
        self.fc = self.target_frames.shape[0] - 1
        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        for _ in range(3):
            app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
            time.sleep(0.1)

    def test(self):

        # cp.random.seed(0)

        # dmd_vecs = cp.random.normal(0.5, 0.4, (self.num_frames, self.n)) *0. + 1.
        # dmd_vecs = cp.clip(dmd_vecs, 0, 1)
        # dmd_vecs = (dmd_vecs * self.dmd_block_w).astype(cp.int)/self.dmd_block_w
        #
        # dmd_errs = cp.random.normal(0., 0.5, (self.num_frames, 2*self.k)) *0. + 1.
        # dmd_errs = cp.clip(dmd_errs, -1, 1)
        # dmd_errs = (dmd_errs*self.dmd_err_block_w).astype(cp.int)/self.dmd_err_block_w
        #
        # slm_arr = cp.random.normal(0, 0.5, (self.n, self.m)) *0. + 1.
        # slm_arr = cp.clip(slm_arr, -1, 1)
        # slm_arr = (slm_arr*64).astype(cp.int)/64
        #
        # self.update_slm1(slm_arr, lut=True)
        #
        # slm2_arr = cp.random.normal(0, 0.5, (self.k, self.m)) *0. + 1.
        # slm2_arr = cp.clip(slm2_arr, -1, 1)
        # slm2_arr = (slm2_arr*64).astype(cp.int)/64
        #
        # self.update_slm2(slm2_arr, lut=False)
        #
        # time.sleep(0.5)

        dmd_vecs = cp.ones((self.num_frames, self.n))
        dmd_vecs[:, 0] = 0.1
        dmd_errs = cp.ones((self.num_frames, 2*self.k))
        dmd_errs[:, 0] = 0.1
        slm_arr = cp.ones((self.n, self.m))
        slm_arr[0, :] = 0.1
        slm_arr[:, 1] = 0.1
        slm2_arr = cp.ones((self.k, self.m))
        slm2_arr[0, :] = 0.1
        slm2_arr[:, 1] = 0.1
        self.update_slm1(slm_arr, lut=True)
        self.update_slm2(slm2_arr, lut=True)

        self.target_frames = cp.zeros((self.num_frames, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[:, :, :, :-1] = make_dmd_batch(dmd_vecs, dmd_errs)
        self.target_frames[..., -1] = 255
        self.fc = self.target_frames.shape[0] - 1
        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(1)

    def run_batch(self, vecs_in, errs_in=None):

        self.target_frames = cp.zeros((self.num_frames+3, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[1:-2, :, :, :-1] = make_dmd_batch(vecs_in, errs_in)
        self.target_frames[..., -1] = 255

        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        self.capture1.frames = []
        self.capture2.frames = []
        self.capture3.frames = []
        self.capture1.ampls = []
        self.capture2.ampls = []
        self.capture3.ampls = []

        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.1)

        self.frames1 = np.array(self.capture1.frames)
        self.frames2 = np.array(self.capture2.frames)
        self.frames3 = np.array(self.capture3.frames)
        self.ampls1 = np.array(self.capture1.ampls)
        self.ampls2 = np.array(self.capture2.ampls)
        self.ampls3 = np.array(self.capture3.ampls)

        return

    # def find_spot_ampls1(self, arrs):
    #
    #     # mask = arrs < 3
    #     # arrs -= 2
    #     # arrs[mask] = 0
    #
    #     def spot_s(i):
    #         return np.s_[:, self.y_centers[i]-2:self.y_centers[i]+3,
    #                      self.x_edge_indxs[2 * i]:self.x_edge_indxs[2 * i + 1]]
    #
    #     spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.m)])
    #     # spot_powers = cp.random.randint(0, 256, (self.num_frames, self.m)).T
    #
    #     spot_ampls = cp.sqrt(spot_powers).T
    #
    #     return spot_ampls
    #
    # def find_spot_ampls2(self, arrs):
    #
    #     # mask = arrs < 3
    #     # arrs -= 2
    #     # arrs[mask] = 0
    #
    #     def spot_s(i):
    #         return np.s_[:, self.y_centers2[i]-2:self.y_centers2[i]+3,
    #                      self.x_edge_indxs2[2 * i]:self.x_edge_indxs2[2 * i + 1]]
    #
    #     spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.k)])
    #     # spot_powers = cp.random.randint(0, 256, (self.num_frames, self.m)).T
    #
    #     spot_ampls = cp.sqrt(spot_powers)
    #     spot_ampls = cp.flip(spot_ampls, axis=0).T
    #
    #     return spot_ampls
    #
    # def find_spot_ampls3(self, arrs):
    #
    #     # mask = arrs < 3
    #     # arrs -= 2
    #     # arrs[mask] = 0
    #
    #     def spot_s(i):
    #         return np.s_[:, self.y_centers3[i]-2:self.y_centers3[i]+3,
    #                      self.x_edge_indxs3[2 * i]:self.x_edge_indxs3[2 * i + 1]]
    #
    #     spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.m)])
    #     # spot_powers = cp.random.randint(0, 256, (self.num_frames, self.m)).T
    #
    #     spot_ampls = cp.sqrt(spot_powers)
    #     spot_ampls = cp.flip(spot_ampls, axis=0).T
    #
    #     return spot_ampls

    def check_ampls(self, a1=True, z2=True, back=True):

        success = True
        processed_ampls = {}
        processed_frames = {}
        checklist_ampls = {}
        checklist_frames = {}

        if a1:
            checklist_ampls['a1'] = self.ampls1
            checklist_frames['a1'] = self.frames1
        if z2:
            checklist_ampls['z2'] = self.ampls2
            checklist_frames['z2'] = self.frames2
        if back:
            checklist_ampls['back'] = self.ampls3
            checklist_frames['back'] = self.frames3

        for key in checklist_ampls.keys():

            ampls = checklist_ampls[key]
            frames = checklist_frames[key]

            maxs = ampls.max(axis=1)
            # print(maxs)

            if maxs[0] > 0.1:
                print(colored('frames out of sync', 'red'))
                success = False
            elif maxs[-1] > 0.1:
                print(colored('frames out of sync', 'red'))
                success = False
            else:
                start = (maxs > 0.1).argmax()
                end = maxs.shape[0] - np.flip((maxs > 0.1)).argmax()
                ampls = ampls[start:end, :]
                frames = frames[start:end, ...]

                if ampls.shape[0] != self.num_frames:
                    print(colored('wrong num frames', 'red'))
                    success = False
                # else:
                #     diffs = np.abs(np.diff(ampls, axis=0))
                #     diffs /= ampls[:-1, :]+1e-5
                #     diffs = np.sort(diffs, axis=1)[:, 2:-2]
                #     diffs = diffs.mean(axis=1)
                #     repeats = (diffs < 0.015).sum() > 0
                #     if repeats:
                #         print(colored('repeated frames', 'red'))
                #         success = False
                else:
                    processed_ampls[key] = ampls
                    processed_frames[key] = frames

        if success:
            if a1:
                self.ampls1 = processed_ampls['a1']
                self.frames1 = processed_frames['a1']
            if z2:
                self.ampls2 = processed_ampls['z2']
                self.frames2 = processed_frames['z2']
            if back:
                self.ampls3 = processed_ampls['back']
                self.frames3 = processed_frames['back']
            return True
        else:
            return False

    def update_slm1(self, arr, phi=None, lut=True):

        gpu_a = cp.abs(arr)
        if phi is None:
            gpu_phi = cp.angle(arr)
        else:
            gpu_phi = phi
        # gpu_a = cp.flip(gpu_a, axis=0)
        # gpu_phi = cp.flip(gpu_phi, axis=0)

        if lut:
            map_indx = cp.argmin(cp.abs(self.gpu_ampl_lut_slm1 - gpu_a), axis=0)
            gpu_a = cp.linspace(0., 1., 64)[map_indx]

        img = make_slm1_rgb(gpu_a, gpu_phi)
        self.slm.updateArray('SLM1', img)

    def update_slm2(self, arr, lut):

        gpu_a = cp.abs(arr)
        gpu_phi = cp.angle(arr)

        # gpu_a = cp.flip(gpu_a, axis=1)
        # gpu_phi = cp.flip(gpu_phi, axis=1)

        if lut:
            map_indx = cp.argmin(cp.abs(self.gpu_ampl_lut_slm2 - gpu_a), axis=0)
            gpu_a = cp.linspace(0., 1., 32)[map_indx]

        img = make_slm2_rgb(gpu_a, gpu_phi)
        self.slm.updateArray('SLM2', img)

    @staticmethod
    def find_norm_params(theory, measured):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory.shape[1] == measured.shape[1]

        norm_params = np.array([curve_fit(line, np.abs(theory[:, j]), measured[:, j])[0]
                               for j in range(theory.shape[1])])
        return norm_params

    @staticmethod
    def update_norm_params(theory, measured, norm_params):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory.shape[1] == measured.shape[1]

        norm_params_adjust = np.array([curve_fit(line, np.abs(theory[:, j]), measured[:, j])[0]
                                       for j in range(theory.shape[1])])
        norm_params[:, 1] += norm_params[:, 0].copy() * norm_params_adjust[:, 1].copy()
        norm_params[:, 0] *= norm_params_adjust[:, 0].copy()

        return norm_params

    @staticmethod
    def normalise_ampls(ampls, norm_params):
        ampls = (ampls - norm_params[:, 1].copy()) / norm_params[:, 0].copy()
        return ampls

    def close(self):
        self.cameras.Close()
        # imageWindow.Close()
        self.context.pop()
        app.quit()
        print('\nfinished\n')
        os._exit(0)


if __name__ == "__main__":

    n, m, k = 3, 10, 3
    num_frames = 2

    control = Controller(n, m, k, num_frames, use_pylons=False, use_ueye=False)

    for _ in range(500):

        t0 = time.perf_counter()

        control.test()

        t1 = time.perf_counter()

        print(t1-t0)

        # time.sleep(1)

    print('done')
