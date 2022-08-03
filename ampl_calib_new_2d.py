import time
import numpy as np
import cupy as cp
from pypylon import pylon
from glumpy import app
from glumpy.app import clock
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm_images import make_slm1_rgb, make_dmd_image, update_params
import matplotlib.pyplot as plt

class CaptureProcess(pylon.ImageEventHandler):

    def __init__(self):
        super().__init__()

        self.frames = []

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            image = grab_result.GetArray()

            if image.max() > 30:  # 8
                self.frames.append(image)

            # self.frames = self.frames[-3001:]

class Controller:

    def __init__(self,  n, m):

        self.n = n
        self.m = m

        self.batch_size = 240
        self.num_frames = 42


        #######
        # SLM #
        #######

        self.dmd_block_w = update_params(self.n, self.m, self.batch_size, self.num_frames)

        self.slm = SLMdisplay(-1920-2560, 1920, 1080, 'SLM 1',
                              2560, 1920, 1152, 'SLM 2',
                              True)

        #######
        # DMD #
        #######

        self.backend = app.use('glfw')
        self.window = app.Window(1920, 1080, fullscreen=0, decoration=0)
        self.window.set_position(-1920-1920-2560, 0)
        self.window.activate()
        self.window.show()

        self.dmd_clock = clock.Clock()

        @self.window.event
        def on_draw(dt):
            window_on_draw(self.window, self.screen, self.cuda_buffer, self.cp_arr)
            self.frame_count += 1
            self.cp_arr = self.target_frames[self.frame_count % len(self.target_frames)]

        self.screen, self.cuda_buffer, self.context = setup(1920, 1080)

        self.null_frame, img = make_dmd_image(np.zeros((self.n, self.m)))
        self.null_frames = [self.null_frame for _ in range(10)]

        ###########
        # CAMERAS #
        ###########

        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        # Print the model name of the camera.
        print("Using device ", self.camera.GetDeviceInfo().GetModelName())
        print()

        pylon.FeaturePersistence.Load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/ONN/tools/pylon_settings_full_area.pfs", self.camera.GetNodeMap())

        # register the background handler and start grabbing using background pylon thread
        self.capture = CaptureProcess()
        self.camera.RegisterImageEventHandler(self.capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)

        self.target_frames = None
        self.fc = None
        self.cp_arr = None
        self.frame_count = None

        self.frames = None

        self.fig, self.axs = plt.subplots(1, 1)

        self.dmd_imgs = np.load('./tools/dmd_calib_squares_n48_m21.npy')
        self.dmd_imgs = cp.array(self.dmd_imgs, dtype=cp.uint8)

        self.init_dmd()

        print('setup complete')

    def init_dmd(self):

            self.target_frames = cp.zeros((10, 1080, 1920, 4), dtype=cp.uint8)
            self.target_frames[..., -1] = 255
            self.fc = self.target_frames.shape[0] - 1
            self.cp_arr = self.target_frames[0]
            self.frame_count = 0

            for _ in range(5):
                self.capture.frames = []
                app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
                time.sleep(0.1)

    def update_slm1(self, arr):

        arr_out = cp.asarray(arr.copy())
        arr_out = np.flip(arr_out.get(), axis=1)

        img = make_slm1_rgb(arr_out)

        self.slm.updateArray_slm1(img)

        time.sleep(0.7)

    def run_batch(self, ref_phase):

        t0 = time.perf_counter()

        phi_arr = np.zeros((self.n, self.m))
        phi_arr[self.n//2, self.m//2] = ref_phase

        arr = np.ones((self.n, self.m))
        self.update_slm1(arr * np.exp(1j * phi_arr))


        # shape = (123, 1080, 1920, 4)


        # all_frames = []

        self.target_frames = cp.zeros((42+4, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[2:-2, ...] = self.dmd_imgs
        self.target_frames[..., -1] = 255
        self.fc = self.target_frames.shape[0] - 1
        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

            # self.axs.imshow(img.get())
            # plt.show()

        self.capture.frames = []
        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.1)

        frames = np.array(self.capture.frames)
        print(frames.shape)
            # all_frames.append(frames)

        # np.save('./raw/frames_temp.npy', np.array(frames))

        t1 = time.perf_counter()

        print('batch time: {:.3f}'.format(t1 - t0))

        return frames

if __name__ == "__main__":

    n, m = 48, 21

    control = Controller(n, m)

    res = 32

    all_frames = np.empty((res, n*m, 91, 128))
    phases = np.linspace(0, 2*np.pi, res)

    for k, phase in enumerate(phases):

        print(k)

        rep_frames = []
        num_reps = 5
        for rep in range(num_reps):

            frs = control.run_batch(phase)

            np.save(f'./raw/frames_temp.npy', frs)

            rep_frames.append(frs)

        frames = np.array(rep_frames)
        print(frames.shape)

        frames = np.mean(frames, axis=0)

        all_frames[k, ...] = frames

        np.save(f'./raw/frames_batch_{k}_4.npy', frames)

        time.sleep(0.1)

    print('finished, saving')

    np.save('./raw/phase_calib_raw_frames_4.npy', all_frames)

    print('done')
