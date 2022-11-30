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
from collections import deque
from threading import Thread
import ueye_thread
import matplotlib.pyplot as plt
from PIL import Image
import cv2


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class CaptureProcess(pylon.ImageEventHandler):

    def __init__(self, window, live):
        super().__init__()

        buffer_size = 300

        self.window = window
        self.live = live
        self.raw_markers = deque(maxlen=buffer_size)
        self.raw_frames = deque(maxlen=buffer_size)
        self.frames = None
        self.ampls = None
        self.markers = None
        self.values = None
        self.norm_params = None

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            # print('got one')

            # if self.live:
            #     self.window.SetImage(grab_result)
            # self.window.Show()

            image = grab_result.GetArray()
            self.raw_frames.append(image)

            marker = image[40-2:40+3, 595:614].max()
            self.raw_markers.append(marker)


class Controller:

    def __init__(self,  _n, _m, _k, _num_frames,
                 use_pylons=True, use_ueye=True, live=True):

        self.n = _n
        self.m = _m
        self.k = _k
        self.num_frames = _num_frames
        self.live = live

        self.captures = {}

        #################
        # PYLON CAMERAS #
        #################

        if use_pylons:

            # os.environ["PYLON_CAMEMU"] = "2"
            tlfactory = pylon.TlFactory.GetInstance()
            time.sleep(0.5)
            devices = tlfactory.EnumerateDevices()
            time.sleep(0.5)
            assert len(devices) == 2
            self.cameras = pylon.InstantCameraArray(2)
            time.sleep(0.5)
            for i, camera in enumerate(self.cameras):
                print(devices[i].GetFriendlyName())
                camera.Attach(tlfactory.CreateDevice(devices[i]))
                camera.Open()

            pylon.FeaturePersistence.Load("./tools/MVM v3/pylon_settings_a1.pfs", self.cameras[0].GetNodeMap())
            pylon.FeaturePersistence.Load("./tools/MVM v3/pylon_settings_back.pfs", self.cameras[1].GetNodeMap())

            # self.imageWindow1 = pylon.PylonImageWindow()
            # self.imageWindow1.Create(1)
            # self.imageWindow1.Show()
            #
            # self.imageWindow3 = pylon.PylonImageWindow()
            # self.imageWindow3.Create(3)
            # self.imageWindow3.Show()

            self.imageWindow1 = None
            self.imageWindow3 = None

            self.captures['a1'] = CaptureProcess(self.imageWindow1, self.live)
            # self.captures['back'] = CaptureProcess(self.imageWindow3, self.live)

            self.cameras[0].RegisterImageEventHandler(self.captures['a1'], pylon.RegistrationMode_ReplaceAll,
                                                      pylon.Cleanup_None)
            # self.cameras[1].RegisterImageEventHandler(self.captures['back'], pylon.RegistrationMode_ReplaceAll,
            #                                           pylon.Cleanup_None)

            self.cameras.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
            time.sleep(1)
            print('pylon cameras started')

        ###############
        # UEYE CAMERA #
        ###############

        if use_ueye:
            self.captures['z2'] = ueye_thread.UeyeCamera(self.k, self.live)
            self.captures['z2'].start()

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

        self.slm = SLMdisplay([-0*1920, -2*1920],
                              [1920, 1920],
                              [1080, 1152],
                              ['SLM1', 'SLM2'],
                              isImageLock=True)

        #######
        # DMD #
        #######

        self.backend = app.use('glfw')
        self.window = app.Window(1920, 1080, fullscreen=0, decoration=0)
        self.window.set_position(-1*1920, 0)
        self.window.activate()
        self.window.show()

        self.dmd_clock = clock.Clock()

        @self.window.event
        def on_draw(dt):
            window_on_draw(self.window, self.screen, self.cuda_buffer, self.cp_arr)
            self.frame_count += 1
            self.cp_arr = self.target_frames[self.frame_count % len(self.target_frames)]

        self.screen, self.cuda_buffer, self.context = setup(1920, 1080)

        print('started slm windows')

        ###############

        # self.target_frames = None
        # self.fc = self.num_frames+2
        # self.cp_arr = None
        # self.frame_count = None

        # self.frames1 = None
        # self.ampls1 = None
        # self.a1s = None
        # self.norm_params1 = None
        #
        # self.frames2 = None
        # self.ampls2 = None
        # self.z2s = None
        # self.norm_params2 = None
        #
        # self.frames3 = None
        # self.ampls3 = None
        # self.d2s = None
        # self.norm_params3 = None

        self.marker_frame = make_dmd_batch(vecs=None, marker=True)

        spot_indxs = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/ONN/tools/"
                             "spot_indxs.npy")

        self.y_center = spot_indxs[-1]
        self.spot_indxs = spot_indxs[:-1]
        self.area_width = self.spot_indxs.shape[0]//self.m

        spot_indxs2 = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/ONN/tools/"
                              "spot_indxs2.npy")
        self.y_center2 = spot_indxs2[-1]
        self.spot_indxs2 = spot_indxs2[:-1]
        self.area_width2 = self.spot_indxs2.shape[0]//self.k

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

    def dmd_run_blanks(self, on=False):

        self.target_frames = cp.zeros((10, 1080, 1920, 4), dtype=cp.uint8)
        if on:
            self.target_frames += 255
        self.cp_arr = self.target_frames[0]
        self.frame_count = 0
        self.fc = self.target_frames.shape[0]+1

        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.2)

        return

    def run_batch(self, vecs_in, errs_in=None):

        self.target_frames = cp.zeros((self.num_frames+10, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[1, :, :, :-1] = self.marker_frame.copy()
        self.target_frames[2:2+self.num_frames, :, :, :-1] = make_dmd_batch(vecs_in, errs_in, marker=False)
        self.target_frames[2+self.num_frames, :, :, :-1] = self.marker_frame.copy()
        self.target_frames[..., -1] = 255

        self.cp_arr = self.target_frames[0]
        self.frame_count = 0
        self.fc = self.target_frames.shape[0]

        for key, camera in self.captures.items():
            camera.raw_frames.clear()
            camera.raw_markers.clear()

        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.1)

        for key, camera in self.captures.items():
            camera.frames = np.array(camera.raw_frames)
            camera.markers = np.array(camera.raw_markers)

            camera.success, camera.error = self.process_markers(key)

            if camera.success:
                camera.ampls = self.find_spot_ampls(camera.frames, key)

        return

    def process_markers(self, key):

        camera = self.captures[key]

        markers = camera.markers
        markers = markers/markers.max()

        # print(markers)
        # print(camera.frames.max(axis=(1, 2)))

        marker_indxs = np.where(markers > 0.8)[0]

        if marker_indxs.shape[0] == 2:
            start = marker_indxs[0]
            end = marker_indxs[1]
            # print('start: ', start)
            # print('end: ', end)
            if (end - start) != self.num_frames+1:
                print(colored(f'wrong num frames ^', 'red'))
                return False, 'count'
            else:
                camera.frames = camera.frames[start+1:end]
                return True, None
        else:
            print(colored(f'no clear markers ^', 'red'))
            return False, 'markers'

    def find_spot_ampls(self, arrs, cam):
        if cam == 'a1':

            mask = arrs < 2
            arrs -= 1
            arrs[mask] = 0

            arrs = np.transpose(arrs, (0, 2, 1))
            spots = arrs[:, self.spot_indxs, self.y_center-1:self.y_center+2].copy()
            powers = spots.reshape((arrs.shape[0], self.m, self.area_width, 3)).mean(axis=(2, 3))
            spot_ampls = np.sqrt(powers)

            # print('found ampls 1, shape: ', spot_ampls.shape)

        if cam == 'z2':

            # mask = arrs < 2
            # arrs -= 1
            # arrs[mask] = 0

            arrs = np.transpose(arrs, (0, 2, 1))
            spots = arrs[:, self.spot_indxs2, self.y_center2-2:self.y_center2+2].copy()
            spots = spots.reshape((arrs.shape[0], self.k, self.area_width2, 4)).mean(axis=(2, 3))

            spot_ampls = np.sqrt(spots)

            # print('found ampls 2, shape: ', spot_ampls.shape)

        if cam == 'back':
            pass

            # mask = arrs < 3
            # arrs -= 2
            # arrs[mask] = 0
            #
            # def spot_s(i):
            #     return np.s_[:, self.y_centers3[i]-2:self.y_centers3[i]+3,
            #                  self.x_edge_indxs3[2 * i]:self.x_edge_indxs3[2 * i + 1]]
            #
            # spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.m)])
            # # spot_powers = cp.random.randint(0, 256, (self.num_frames, self.m)).T
            #
            # spot_ampls = cp.sqrt(spot_powers)
            # spot_ampls = cp.flip(spot_ampls, axis=0).T

        return spot_ampls

    def check_ampls(self, which_to_check=('a1', 'z2', 'back')):

        success = True
        errors = []

        lim = {'a1': 1., 'z2': 1., 'back': 1.}

        for key in which_to_check:

            camera = self.captures[key]

            maxs = camera.ampls.max(axis=1)
            print(key, ":")
            print(maxs)

            if maxs[0] > lim[key]:
                print(colored('frames out of sync ^', 'red'))
                errors.append(f'{key} sync')
                success = False
                continue
            elif maxs[-1] > lim[key]:
                print(colored('frames out of sync ^', 'red'))
                errors.append(f'{key} sync')
                success = False
                continue
            else:
                camera.ampls = camera.ampls[start + 1:end, :]
                camera.frames = camera.frames[start + 1:end, :]


                # start = (maxs > lim[key]).argmax()
                # print(start)

                # maxs = maxs[start:]
                # diffs = np.abs(np.diff(maxs))
                #
                # start1 = (diffs > 0.1).argmax()

                # end = maxs.shape[0] - np.flip((maxs > lim[key])).argmax()
                # start += 1
                # end -= 1

                # if key == 'a1':
                #     a1_start = start
                #     a1_end = end
                # if key == 'back':
                #     if end - start != self.num_frames:
                #         print(colored(f'wrong num frames, using 10 ^', 'red'))
                #         # start = a1_start
                #         end = start+10 #a1_end

                # end = start + 1 + self.num_frames

                # else:
                #     diffs = np.abs(np.diff(ampls, axis=0))
                #     diffs /= ampls[:-1, :]+1e-5
                #     diffs = np.sort(diffs, axis=1)[:, 2:-2]
                #     diffs = diffs.mean(axis=1)
                #     repeats = (diffs < 0.015).sum() > 0
                #     if repeats:
                #         print(colored('repeated frames', 'red'))
                #         success = False

        if success:
            return True, None
        else:
            return False, errors

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
            gpu_a = cp.linspace(0., 1., 256)[map_indx]

        img = make_slm1_rgb(gpu_a, gpu_phi)

        self.slm.updateArray('SLM1', img)

    def update_slm2(self, arr, lut):

        gpu_a = cp.abs(arr)
        gpu_phi = cp.angle(arr)

        # gpu_a = cp.flip(gpu_a, axis=1)
        # gpu_phi = cp.flip(gpu_phi, axis=1)

        if lut:
            map_indx = cp.argmin(cp.abs(self.gpu_ampl_lut_slm2 - gpu_a), axis=0)
            gpu_a = cp.linspace(0., 1., 64)[map_indx]

        img = make_slm2_rgb(gpu_a, gpu_phi)
        self.slm.updateArray('SLM2', img)

    @staticmethod
    def find_norm_params(theory, measured):

        def line(x, grad, c):
            return (grad * x) + c

        def line_no_c(x, grad):
            return grad*x

        assert theory.shape[1] == measured.shape[1]

        norm_params = np.empty((theory.shape[1], 2))
        for j in range(theory.shape[1]):

            # print(theory[:, j].std())

            # if theory[:, j].std() < 0.1:
            #     norm_params[j, :] = np.array([curve_fit(line_no_c, np.abs(theory[:, j]), measured[:, j])[0], 0.])
            # else:
            norm_params[j, :] = curve_fit(line, theory[:, j], measured[:, j])[0]

        return norm_params

    @staticmethod
    def update_norm_params(theory, measured, norm_params):

        def line(x, grad, c):
            return (grad * x) + c

        def line_no_c(x, grad):
            return grad*x

        assert theory.shape[1] == measured.shape[1]

        norm_params_adjust = np.empty((theory.shape[1], 2))
        for j in range(theory.shape[1]):

            # mask = measured[:, j] > 0.05

            # if theory[:, j].std() < 0.5:
            #     norm_params_adjust[j, :] = np.array([curve_fit(line_no_c, theory[:, j], measured[:, j])[0], 0.])
            # else:

            # try:
            norm_params_adjust[j, :] = curve_fit(line, theory[:, j], measured[:, j])[0]
            # except ValueError:
            #     norm_params_adjust[j, 0] = 1.
            #     norm_params_adjust[j, 1] = 0.

        # print(norm_params.shape, norm_params.dtype)
        # print(norm_params_adjust.shape,  norm_params_adjust.dtype)

        norm_params[:, 1] += norm_params[:, 0].copy() * norm_params_adjust[:, 1].copy()
        norm_params[:, 0] *= norm_params_adjust[:, 0].copy()

        return norm_params

    def close(self):
        self.captures.Close()
        # imageWindow.Close()
        self.context.pop()
        app.quit()
        print('\nfinished\n')
        os._exit(0)


if __name__ == "__main__":

    n, m, k = 3, 10, 5
    num_frames = 2

    control = Controller(n, m, k, num_frames, use_pylons=False, use_ueye=False)

    for _ in range(500):

        t0 = time.perf_counter()

        control.test()

        t1 = time.perf_counter()

        print(t1-t0)

        # time.sleep(1)

    print('done')
