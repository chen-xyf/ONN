from pypylon import pylon
from pypylon import genicam
import time
import sys
import threading
import time
import sys
import signal
import numpy as np
from scipy.optimize import curve_fit
import cupy as cp
from glumpy import app
# from make_dmd_image import make_dmd_rgb, make_dmd_image
# from MNIST import map_vec_to_arr
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm1_image import make_slm_rgb, make_dmd_image, make_dmd_batch, update_params
from multiprocessing import Process, Pipe
from ANN import DNN, DNN_1d, DNN_complex, accuracy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
from termcolor import colored
import queue
from collections import deque
import ticking
from glumpy.app import clock
from pylon import view_camera

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])

ref_spot = m//2

ref_block_val = 0.3
batch_size = 240
num_batches = 5
num_frames = 10

dmd_block_w = update_params(ref_block_val, batch_size, num_frames, is_complex=True)


inputs = loadmat('C:/Users/spall/OneDrive - Nexus365/Code/JS/controller/onn_test/MNIST digit - subsampled - 100.mat')

num_train = 60000
num_test = 10000

trainY_raw = inputs['trainY']
trainY = np.zeros((num_train, 10))
for i in range(num_train):
    trainY[i, trainY_raw[0, i]] = 1

testY_raw = inputs['testY']
testY = np.zeros((num_test, 10))
for i in range(num_test):
    testY[i, testY_raw[0, i]] = 1

trainX_raw = inputs['trainX']
trainX = np.empty((num_train, 100))
for i in range(num_train):
    trainX_k = trainX_raw[i, :] - trainX_raw[i, :].min()
    trainX_k = trainX_k / trainX_k.max()
    trainX[i, :] = trainX_k

testX_raw = inputs['testX']
testX = np.empty((num_test, 100))
for i in range(num_test):
    testX_k = testX_raw[i, :] - testX_raw[i, :].min()
    testX_k = testX_k / testX_k.max()
    testX[i, :] = testX_k

np.random.seed(0)
np.random.shuffle(trainX)
np.random.seed(0)
np.random.shuffle(trainY)
np.random.seed(0)
np.random.shuffle(testX)
np.random.seed(0)
np.random.shuffle(testY)

valX = testX[:5000, :].copy()
testX = testX[5000:, :].copy()

valY = testY[:5000, :].copy()
testY = testY[5000:, :].copy()

trainX -= 0.1
trainX = np.clip(trainX, 0, 1)
trainX /= trainX.max()

valX -= 0.1
valX = np.clip(valX, 0, 1)
valX /= valX.max()

testX -= 0.1
testX = np.clip(testX, 0, 1)
testX /= testX.max()

trainX = (trainX*dmd_block_w).astype(int)/dmd_block_w
valX = (valX*dmd_block_w).astype(int)/dmd_block_w
testX = (testX*dmd_block_w).astype(int)/dmd_block_w

trainX_cp = cp.array(trainX, dtype=cp.float32)
valX_cp = cp.array(valX, dtype=cp.float32)
testX_cp = cp.array(testX, dtype=cp.float32)

################
# SLM display #
################

slm = SLMdisplay()

################
# DMD display #
################

backend = app.use('glfw')

window = app.Window(1920, 1080, fullscreen=0, decoration=0)
window.set_position(-1920*2, 0)
window.activate()
window.show()

dmd_clock = clock.Clock()

@window.event
def on_draw(dt):
    global cp_arr, frame_count, target_frames
    window_on_draw(window, screen, cuda_buffer, cp_arr)
    frame_count += 1
    cp_arr = target_frames[frame_count % len(target_frames)]

screen, cuda_buffer, context = setup(1920, 1080)


def dmd_one_frame(arr, ref):
    img = make_dmd_image(arr, ref=ref, ref_block_val=ref_block_val)
    return [img]


null_frame = dmd_one_frame(np.zeros((n, m)), ref=0)[0]
null_frames = [null_frame for _ in range(10)]

full_frame = dmd_one_frame(np.ones((n, m)), ref=0)[0]
full_frames = [full_frame for _ in range(10)]


################
# Pylon camera #
################

# imageWindow = pylon.PylonImageWindow()
# imageWindow.Create(1)

# Create an instant camera object with the camera device found first.
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Print the model name of the camera.
print("Using device ", camera.GetDeviceInfo().GetModelName())
print()

pylon.FeaturePersistence.Load("./tools/pylon_settings.pfs", camera.GetNodeMap())

class CaptureProcess(pylon.ImageEventHandler):
    def __init__(self):
        super().__init__()

        self.frames = []

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            image = grab_result.GetArray()

            if image.max() > 10:
                self.frames.append(image)

            self.frames = self.frames[-1001:]

            # imageWindow.SetImage(grab_result)
            # imageWindow.Show()

# register the background handler and start grabbing using background pylon thread
capture = CaptureProcess()
camera.RegisterImageEventHandler(capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
time.sleep(1)

ampl_norm_val = 0.1

scale_guess = 10

batch_size = 240
num_frames = 10

target_frames = cp.zeros((num_frames+4, 1080, 1920, 4), dtype=cp.uint8)
target_frames[..., -1] = 255
fc = target_frames.shape[0] - 1
cp_arr = target_frames[0]
frame_count = 0
for _ in range(5):
    capture.frames = []
    app.run(clock=dmd_clock, framerate=0, framecount=fc)
    time.sleep(0.1)

print('finished config')
print('############')
print()

complex_output_ratios = np.load('./tools/complex_output_ratios.npy')

# actual_uppers_arr_256 = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_256.npy")
#
# actual_uppers_arr_256[:, :, ref_spot] = actual_uppers_arr_256[:, :, ref_spot+1]
#
# uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
# uppers1_ann = np.delete(uppers1_nm, ref_spot, 1)
#
# k = np.abs(np.linspace(-1, 1, 256) - 0.1).argmin()
# z0 = actual_uppers_arr_256[k, ...].sum(axis=0)
#
# z0_norm = z0.copy()/z0.max()
# z0_norm = np.delete(z0_norm, ref_spot)
#
# gpu_actual_uppers_arr_256 = cp.asarray(actual_uppers_arr_256)


actual_uppers_arr_128_flat = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_128_flat.npy")

actual_uppers_arr_128_flat /= actual_uppers_arr_128_flat.max()

uppers1_nm_flat = actual_uppers_arr_128_flat[-1, ...].copy()

uppers1_ann = uppers1_nm_flat.copy()[:, ::4]

gpu_actual_uppers_arr_128_flat = cp.asarray(actual_uppers_arr_128_flat)


x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
y_centers = np.load('./tools/y_centers_list.npy')


def find_spot_ampls(arrs):

    # arrs = np.array([recombine(arr.T) for arr in arrs_in])

    mask = arrs < 3
    arrs -= 2
    arrs[mask] = 0

    def spot_s(i):
        return np.s_[:, y_centers[i] - 2:y_centers[i] + 3, x_edge_indxs[2 * i]:x_edge_indxs[2 * i + 1]]

    spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(m + 1)])

    spot_ampls = cp.sqrt(spot_powers)

    spot_ampls = cp.flip(spot_ampls, axis=0)

    ratio = spot_ampls[ref_spot, :] / spot_ampls[ref_spot + 1, :]

    spot_ampls[ref_spot + 1:, :] *= ratio[None, :]

    spot_ampls = np.delete(spot_ampls.get(), ref_spot+1, 0)

    return spot_ampls.T


def update_slm(arr, lut=False, ref=False, noise_arr_A=None, noise_arr_phi=None):

    global ampl_norm_val, ref_spot

    if arr.shape[1] == m-1:
        arr = np.insert(arr, ref_spot, np.zeros(n), 1)

    if arr.shape[1] == 10:
        arr = np.repeat(arr.copy(), 4, axis=1) * complex_output_ratios.copy()[None, :]

    if lut:
        # gpu_arr = cp.asarray(arr)
        # map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_256 - gpu_arr), axis=0)
        # arr_A = cp.linspace(-1., 1., 256)[map_indx].get()

        gpu_arr = cp.abs(cp.asarray(arr.copy()))
        map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_128_flat - gpu_arr), axis=0)
        arr_A = cp.linspace(0, 1, 128)[map_indx]

    else:
        arr_A = cp.abs(cp.asarray(arr.copy()))

    if ref:
        arr_A[:, ref_spot] = ampl_norm_val

    arr_phi = cp.angle(cp.array(arr.copy()))

    if noise_arr_A is not None:
        arr_A += cp.array(noise_arr_A)

    if noise_arr_phi is not None:
        arr_phi += cp.array(noise_arr_phi)

    arr_out = arr_A * cp.exp(1j*arr_phi)

    arr_out = np.flip(arr_out.get(), axis=1)
    img = make_slm_rgb(arr_out, ref_block_val)
    slm.updateArray(img)
    # time.sleep(0.7)


def init_dmd():

    global target_frames, ref_block_val, batch_size, num_frames
    global cp_arr, frame_count, capture, dmd_clock

    target_frames = cp.zeros((num_frames + 4, 1080, 1920, 4), dtype=cp.uint8)
    target_frames[..., -1] = 255
    fc = target_frames.shape[0] - 1
    cp_arr = target_frames[0]
    frame_count = 0
    for _ in range(5):
        capture.frames = []
        app.run(clock=dmd_clock, framerate=0, framecount=fc)
        time.sleep(0.1)


def run_frames(vecs_in, ref):

    global target_frames, ref_block_val, batch_size, num_frames
    global cp_arr, frame_count, capture, dmd_clock

    target_frames[2:-2, :, :, :-1] = make_dmd_batch(vecs_in, ref=ref, ref_block_val=ref_block_val,
                                                    batch_size=batch_size, num_frames=num_frames)

    cp_arr = target_frames[0]
    frame_count = 0

    capture.frames = []
    app.run(clock=dmd_clock, framerate=0, framecount=fc)
    time.sleep(0.1)

    frames = np.array(capture.frames.copy())

    return frames


def check_num_frames(ampls_in, true_size):

    global target_frames, num_frames

    if ampls_in.shape[0] == true_size:

        meas = ampls_in.copy().reshape((num_frames, true_size // num_frames, m))
        diffs = np.abs(np.array([meas[k + 1, :, m // 3] - meas[k, :, m // 3]
                                 for k in range(num_frames - 1)])).mean(axis=1)
        diffs /= diffs.max()
        repeats = (diffs < 0.25).sum() > 0

        if repeats:
            print(colored('repeated frames', 'red'))
            return False

        else:
            return True

    else:
        print(colored('wrong num frames: {}'.format(ampls_in.shape[0]), 'red'))
        return False


def process_ampls(ampls_in, meas_type):

    if meas_type == 'real':
        z1s = ampls_in - Aref
        z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
        z1s = np.delete(z1s, ref_spot, axis=1)

    elif meas_type == 'complex':

        Iall = ampls_in.copy() ** 2
        I0 = Iall[:, 0::4].copy()
        I1 = Iall[:, 1::4].copy()
        I2 = Iall[:, 2::4].copy()
        I3 = Iall[:, 3::4].copy()
        Xmeas = (I0 - I2) / scale_guess
        Ymeas = (I1 - I3) / scale_guess
        z1s = Xmeas + (1j * Ymeas)

    elif meas_type == 'complex_power':
        Imeas = ampls_in.copy() ** 2
        Imeas = Imeas.reshape(Imeas.shape[0], 10, 4).mean(axis=-1)
        Imeas *= scale_guess
        z1s = Imeas.copy()

    else:
        raise ValueError

    return z1s


def line(x, grad, c):
        return (grad * x) + c


def find_norm_params(theory, measured, meas_type):

    assert theory.shape[1] == measured.shape[1]

    if meas_type == 'complex':

        real_norm_params = np.array([curve_fit(line, np.real(theory[:, j]), np.real(measured[:, j]))[0]
                                     for j in range(theory.shape[1])])
        imag_norm_params = np.array([curve_fit(line, np.imag(theory[:, j]), np.imag(measured[:, j]))[0]
                                     for j in range(theory.shape[1])])

        norm_params = real_norm_params + (1j * imag_norm_params)

    else:
        norm_params = np.array([curve_fit(line, theory[:, j], measured[:, j])[0]
                                for j in range(theory.shape[1])])

    return norm_params


def normalise(z1s_in, norm_params, meas_type):

    if meas_type == 'complex':

        Zreals = (np.real(z1s_in).copy() - np.real(norm_params)[:, 1]) / np.real(norm_params)[:, 0]
        Zimags = (np.imag(z1s_in).copy() - np.imag(norm_params)[:, 1]) / np.imag(norm_params)[:, 0]
        z1s = Zreals + (1j * Zimags)

    else:
        z1s = (z1s_in - norm_params[:, 1])/norm_params[:, 0]

    return z1s


def close():
    global camera, context
    camera.Close()
    # imageWindow.Close()
    context.pop()
    app.quit()
    print('\nfinished\n')