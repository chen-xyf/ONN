from ANN import DNN, DNN_1d, accuracy, softmax, sigmoid, bce, error
from ANN import DNN_backprop, satab, satab_d
import matplotlib.pyplot as plt
import random
from termcolor import colored
import time
import numpy as np
import cupy as cp
from scipy.io import loadmat
from termcolor import colored
from glumpy.app import clock
from ONN_config import Controller
from scipy.optimize import curve_fit
import pyautogui
from datetime import datetime


class MyONN:

    def __init__(self, batch_size, num_batches, num_epochs, w1_0, w2_0, b1_0, lr, scaling,
                 dimensions, save_folder,
                 trainx=None, trainy=None, testx=None, testy=None,
                 forward='digital', backward='digital'):

        if forward == 'optical':
            self.ctrl = Controller(*dimensions, _num_frames=batch_size, use_pylons=True, use_ueye=True)

        self.n = dimensions[0]
        self.m = dimensions[1]
        self.k = dimensions[2]
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.loop_clock = clock.Clock()

        self.accs = []
        self.loss = []
        self.min_loss = 10

        self.errors1 = []
        self.errors2 = []
        self.errors3 = []

        self.best_w1 = w1_0
        self.best_w2 = w2_0
        self.best_b1 = b1_0

        self.trainx = trainx
        self.testx = testx
        self.trainy = trainy
        self.testy = testy

        self.res = 20
        low = 0.
        high = 1.
        gridx1 = np.linspace(low, high, self.res).repeat(self.res)
        gridx2 = np.tile(np.linspace(low, high, self.res), self.res)
        self.gridx = np.empty((self.res**2, 2))*0.
        self.gridx[:, 0] = gridx1
        self.gridx[:, 1] = gridx2

        self.batch_indxs_list = None
        self.save_folder = save_folder

        self.forward = forward
        self.backward = backward

        self.sm_scaling = scaling

        ##### initialise dnn parameters #####

        m_dw1 = np.zeros((self.n - 1, self.m))
        v_dw1 = np.zeros((self.n - 1, self.m))

        m_db1 = np.zeros(self.m)
        v_db1 = np.zeros(self.m)

        m_dw2 = np.zeros((self.m, 2))
        v_dw2 = np.zeros((self.m, 2))

        beta1 = 0.9
        beta2 = 0.999

        adam_params = (m_dw1, v_dw1, m_db1, v_db1, m_dw2, v_dw2, beta1, beta2)

        self.dnn = DNN_backprop(w1_0, w2_0, b1_0, lr, scaling, *adam_params)

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2, self.axsa, self.axsd],
                    [self.axs3, self.axs4, self.axs5, self.axsb, self.axse],
                    [self.axs6, self.axs7, self.axs8, self.axsc, self.axsf]] = plt.subplots(3, 5, figsize=(24, 12))

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1920, 0, 1920, 1080)
        mngr.window.showMaximized()

        self.axsa.set_title('Loss')
        self.axsa.set_xlim(0, self.num_epochs*self.num_batches)
        self.axsa.set_ylim(0, 0.5)

        self.axsb.set_title('Accuracy')
        self.axsb.set_xlim(0, self.num_epochs)
        self.axsb.set_ylim(0, 100)

        self.axsc.set_title('Errors')
        self.axsc.set_xlim(0, self.num_epochs*self.num_batches)
        self.axsc.set_ylim(0, 0.1)

        self.loss_plot = self.axsa.plot(self.loss, linestyle='-', marker='', c='b')
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='x', c='b')
        self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
        self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')
        self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')

        # self.axsd.set_title('Test actual classes')
        # self.label_scatter = self.axsd.scatter(valx[:, 0], valx[:, 1], c=valy.argmax(axis=1))
        # self.axse.set_title('Test prediction classes')
        # self.pred_scatter = self.axse.scatter(valx[:, 0], valx[:, 1])
        # self.axsf.set_title('Correct predictions')
        # self.correct_scatter = self.axsf.scatter(valx[:, 0], valx[:, 1])

        self.axs3.set_ylim(-3, 3)
        self.axs4.set_ylim(-3, 3)
        self.axs5.set_ylim(-3, 3)

        self.th_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        # self.th_line2 = self.axs4.plot(np.zeros(5), linestyle='', marker='o', c='b')[0]
        # self.meas_line2 = self.axs4.plot(np.zeros(5), linestyle='', marker='x', c='r')[0]
        # self.label_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='o', c='g')[0]
        # self.softmax_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='x', c='orange')[0]

        self.th_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        self.img1 = self.axs6.imshow(np.zeros((80, 672)), aspect='auto', vmin=0, vmax=255)
        self.img2 = self.axs7.imshow(np.zeros((86, 720)), aspect='auto', vmin=0, vmax=255)
        self.img3 = self.axs8.imshow(np.zeros((80, 672)), aspect='auto', vmin=0, vmax=255)

        self.od, self.thresh = np.load("./tools/satab_params.npy")
        self.norm_params1 = np.load("./tools/calibration/norm_params1.npy ")
        self.norm_params2 = np.load("./tools/calibration/norm_params2.npy")[2:4, :]

        self.theory_a1 = np.load("./tools/calibration/theory_a1.npy")
        self.meas_a1 = np.load("./tools/calibration/meas_a1.npy")
        self.theory_z2 = np.load("./tools/calibration/theory_z2.npy")[:, 2:4]
        self.meas_z2 = np.load("./tools/calibration/meas_z2.npy")[:, 2:4]

        self.axs0.cla()
        low = -3
        high = 3
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.a1_scatter = self.axs0.plot(self.theory_a1, self.meas_a1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

        xs = np.linspace(low, high, 100)
        gs = satab(xs, self.od, self.thresh)
        self.axs0.plot(xs, gs, c='black', linewidth=1)

        self.axs1.cla()
        low = -2
        high = 2
        self.axs1.set_ylim(low, high)
        self.axs1.set_xlim(low, high)
        # self.z2_scatter = self.axs1.plot(self.theory_z2, self.meas_z2, linestyle='', marker='x')
        self.axs1.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs1.set_title('layer 2')
        self.axs1.set_xlabel('theory')
        self.axs1.set_ylabel('measured')

        plt.tight_layout()

        plt.draw()
        # plt.pause(2)
        # plt.show()

        self.loop_clock.tick()

    def case10_dmd(self, seed):
        cp.random.seed(seed)

        # stds = cp.random.uniform(0., 1., self.ctrl.num_frames)
        # means = cp.random.uniform(0.3, 0.7, self.ctrl.num_frames)
        # dmd_vec = cp.zeros((self.ctrl.num_frames, self.n), dtype=cp.uint8)
        # for j in range(self.ctrl.num_frames):
        #     dmd_vec[j, :] = cp.random.normal(means[j], stds[j], self.n)

        dmd_vec = cp.zeros((self.ctrl.num_frames, self.n), dtype=cp.float16)
        for j in range(self.ctrl.num_frames):
            dmd_vec[j, :] = cp.random.normal(0.5, 0.4, self.n)

        dmd_vec = cp.clip(dmd_vec, 0., 1.)
        dmd_vec = (dmd_vec*self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

        # dmd_vec *= 0.
        # dmd_vec += 1.

        # dmd_vec = cp.linspace(0., 1., self.ctrl.num_frames)[:, None].repeat(self.n, axis=-1)

        return dmd_vec

    def case10_slm(self, seed):
        cp.random.seed(seed*100)
        slm_arr = cp.zeros((self.n, self.m), dtype=cp.float32)
        stds = cp.random.uniform(0., 1., self.m)
        means = cp.random.uniform(-1., 1., self.m)
        for j in range(self.m):
            slm_arr[:, j] = cp.random.normal(means[j], stds[j], self.n)

        # slm_arr = cp.random.normal(0., 0.5, (self.n, self.m))

        slm_arr = cp.clip(slm_arr, -1., 1.)
        slm_arr = (slm_arr*256).astype(cp.int)/256

        # slm_arr *= 0.
        # slm_arr += 1.

        return slm_arr

    def case10_slm2(self, seed):

        cp.random.seed(seed*1000)
        slm2_arr = cp.random.normal(0., 0.2, (k, m))
        slm2_arr = cp.clip(slm2_arr, -1., 1.)
        slm2_arr = (slm2_arr*64).astype(cp.int)/64

        # slm2_arr *= 0.
        # slm2_arr += 1.

        return slm2_arr

    def wait_for_user(self, message, pos=19):
        print(colored(message, 'yellow'))
        input("Press enter to continue...")
        pyautogui.click(4500+(pos*37), 1425)
        time.sleep(0.1)
        pyautogui.click(4500+(pos*37)+94, 1424-79)
        time.sleep(0.1)
        self.ctrl.dmd_run_blanks()

    def run_calibration_linear(self):

        num = 30

        raw_z1s = []
        theory_z1s = []
        raw_z2s = []
        theory_z2s = []

        slm_arr = cp.ones((self.n, self.m))
        self.ctrl.update_slm1(slm_arr, lut=True)
        slm2_arr = cp.ones((self.k, self.m))
        self.ctrl.update_slm2(slm2_arr, lut=True)
        self.ctrl.dmd_run_blanks(on=True)

        self.wait_for_user("Tune off resonance")

        successful = 0

        for seed in range(num):

            print(seed)

            slm_arr = self.case10_slm(seed)
            self.ctrl.update_slm1(slm_arr, lut=True)

            slm2_arr = self.case10_slm2(seed)
            self.ctrl.update_slm2(slm2_arr, lut=True)

            time.sleep(0.5)

            dmd_vecs = self.case10_dmd(seed)
            self.ctrl.run_batch(dmd_vecs)

            success = self.ctrl.captures['a1'].success and self.ctrl.captures['z2'].success

            if success:
                print('batch successful')
                successful += 1

                ampls = self.ctrl.captures['a1'].ampls
                self.axs2.cla()
                for j in range(10):
                    self.axs2.plot(ampls[:, j], linestyle='', marker='x')
                plt.pause(0.1)

                raw_z1 = self.ctrl.captures['a1'].ampls.copy()
                theory_z1 = cp.dot(dmd_vecs, slm_arr)

                raw_z2 = self.ctrl.captures['z2'].ampls.copy()
                theory_z2 = cp.dot(theory_z1, slm2_arr.T)

                theory_z1 = theory_z1.get()
                theory_z2 = theory_z2.get()

                raw_z1s.append(raw_z1)
                theory_z1s.append(theory_z1)
                raw_z2s.append(raw_z2)
                theory_z2s.append(theory_z2)

                self.img1.set_array(self.ctrl.captures['a1'].frames[0])
                self.img2.set_array(self.ctrl.captures['z2'].frames[0])

                theory_z1_plot = np.abs(theory_z1[0, :].copy())
                theory_z1_plot -= theory_z1_plot.mean()
                theory_z1_plot /= theory_z1_plot.std()
                raw_z1_plot = raw_z1[0, :].copy()
                raw_z1_plot -= raw_z1_plot.mean()
                raw_z1_plot /= raw_z1_plot.std()

                self.th_line1.set_ydata(theory_z1_plot)
                self.meas_line1.set_ydata(raw_z1_plot)

                theory_z2_plot = np.abs(theory_z2[0, :].copy())
                theory_z2_plot -= theory_z2_plot.mean()
                theory_z2_plot /= theory_z2_plot.std()
                raw_z2_plot = raw_z2[0, :].copy()
                raw_z2_plot -= raw_z2_plot.mean()
                raw_z2_plot /= raw_z2_plot.std()

                # self.th_line2.set_ydata(theory_z2_plot)
                # self.meas_line2.set_ydata(raw_z2_plot)

                plt.pause(0.1)

            print()

        raw_z1 = np.array(raw_z1s).reshape(successful * self.ctrl.num_frames, self.m)
        theory_z1 = np.array(theory_z1s).reshape(successful * self.ctrl.num_frames, self.m)
        meas_z1 = raw_z1 * np.sign(theory_z1)
        norm_params1 = self.ctrl.find_norm_params(theory_z1, meas_z1)
        meas_z1 = (meas_z1 - norm_params1[:, 1].copy()) / norm_params1[:, 0].copy()

        raw_z2 = np.array(raw_z2s).reshape(successful * self.ctrl.num_frames, self.k)
        theory_z2 = np.array(theory_z2s).reshape(successful * self.ctrl.num_frames, self.k)
        meas_z2 = raw_z2 * np.sign(theory_z2)
        norm_params2 = self.ctrl.find_norm_params(theory_z2, meas_z2)
        meas_z2 = (meas_z2 - norm_params2[:, 1].copy()) / norm_params2[:, 0].copy()

        error1 = (meas_z1 - theory_z1).std()
        error2 = (meas_z2 - theory_z2).std()

        print(colored(f'error1 : {error1:.3f}, signal1 : {theory_z1.std():.3f}, '
                      f'ratio1 : {theory_z1.std()/error1:.3f}', 'blue'))
        print(colored(f'error2 : {error2:.3f}, signal2 : {theory_z2.std():.3f}, '
                      f'ratio2 : {theory_z2.std()/error2:.3f}', 'blue'))

        self.axs0.cla()
        low = -3
        high = 3
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.axs0.plot(theory_z1, meas_z1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

        self.axs1.cla()
        low = -10
        high = 10
        self.axs1.set_ylim(low, high)
        self.axs1.set_xlim(low, high)
        self.axs1.plot(theory_z2, meas_z2, linestyle='', marker='x')
        self.axs1.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs1.set_title('layer 2')
        self.axs1.set_xlabel('theory')
        self.axs1.set_ylabel('measured')

        plt.pause(0.1)
        plt.show()

        np.save("./tools/calibration/raw_z1.npy", raw_z1)
        np.save("./tools/calibration/meas_z1.npy", meas_z1)
        np.save("./tools/calibration/theory_z1.npy", theory_z1)
        np.save("./tools/calibration/norm_params1.npy", norm_params1)

    def run_calibration_nonlinear_layer1(self):

        num = 30

        slm_arr = cp.ones((self.n, self.m))
        self.ctrl.update_slm1(slm_arr, lut=True)
        self.ctrl.dmd_run_blanks(on=True)

        self.wait_for_user("Tune on resonance")

        raw_a1s = []

        for seed in range(num):

            print(seed)

            slm_arr = self.case10_slm(seed)
            self.ctrl.update_slm1(slm_arr, lut=True)

            time.sleep(0.5)

            dmd_vecs = self.case10_dmd(seed)
            self.ctrl.run_batch(dmd_vecs)

            success = self.ctrl.captures['a1'].success

            if success:
                print('batch successful')

                ampls = self.ctrl.captures['a1'].ampls
                self.axs2.cla()
                for j in range(10):
                    self.axs2.plot(ampls[:, j], linestyle='', marker='x')
                plt.pause(0.1)

                raw_a1 = self.ctrl.captures['a1'].ampls.copy()
                raw_a1s.append(raw_a1)

                plt.pause(0.01)

        raw_a1 = np.array(raw_a1s).reshape(num * self.ctrl.num_frames, self.m)

        meas_z1 = np.load("./tools/calibration/meas_z1.npy")
        theory_z1 = np.load("./tools/calibration/theory_z1.npy")
        norm_params1 = np.load("./tools/calibration/norm_params1.npy")

        meas_a1 = (raw_a1 - norm_params1[:, 1].copy()) / norm_params1[:, 0].copy()
        meas_a1 *= np.sign(theory_z1)

        od_fit, thresh_fit = curve_fit(satab, theory_z1.flatten(), meas_a1.flatten(), p0=[1,1], sigma=None,
                                       absolute_sigma=False, check_finite=True, bounds=([0,0],[1000,10]),
                                       method=None, jac=None)[0]

        np.save("./tools/satab_params.npy", np.array([od_fit, thresh_fit]))

        print(colored(f"optical depth: {od_fit:.2f}, threshold: {thresh_fit:.2f}"), 'green')

        self.axs0.cla()
        low = -3
        high = 3
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.axs0.plot(theory_z1, meas_z1, linestyle='', marker='x')
        self.axs0.plot(theory_z1, meas_a1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

        xs = np.linspace(low, high, 100)
        gs = satab(xs, od_fit, thresh_fit)
        self.axs0.plot(xs, gs, c='black', linewidth=1)

        theory_a1 = satab(theory_z1, od_fit, thresh_fit)

        error1 = (meas_a1 - theory_a1).std()

        print(colored(f'error1 : {error1:.3f}, signal1 : {theory_a1.std():.3f}, '
                      f'ratio1 : {theory_a1.std()/error1:.3f}', 'blue'))

        plt.pause(0.1)
        plt.show()

        input('Press enter to continue...')

    def run_calibration_nonlinear_layer2(self):

        num = 30

        raw_a1s = []
        theory_a1s = []
        raw_z2s = []
        theory_z2s = []

        od_fit, thresh_fit = np.load("./tools/satab_params.npy")

        slm_arr = cp.ones((self.n, self.m))
        self.ctrl.update_slm1(slm_arr, lut=True)
        slm2_arr = cp.ones((self.k, self.m))
        self.ctrl.update_slm2(slm2_arr, lut=True)
        self.ctrl.dmd_run_blanks(on=True)

        self.wait_for_user("Tune on resonance")

        successful = 0

        for seed in range(num):

            print(seed)

            slm_arr = self.case10_slm(seed)
            self.ctrl.update_slm1(slm_arr, lut=True)

            slm2_arr = self.case10_slm2(seed)
            self.ctrl.update_slm2(slm2_arr, lut=True)

            time.sleep(0.5)

            dmd_vecs = self.case10_dmd(seed)
            self.ctrl.run_batch(dmd_vecs)

            success = self.ctrl.captures['a1'].success and self.ctrl.captures['z2'].success

            if success:
                print('batch successful')
                successful += 1

                ampls = self.ctrl.captures['a1'].ampls
                self.axs2.cla()
                for j in range(10):
                    self.axs2.plot(ampls[:, j], linestyle='', marker='x')
                plt.pause(0.1)

                raw_a1 = self.ctrl.captures['a1'].ampls.copy()
                theory_z1 = cp.dot(dmd_vecs, slm_arr)
                theory_a1 = satab(theory_z1, od_fit, thresh_fit)

                raw_z2 = self.ctrl.captures['z2'].ampls.copy()
                theory_z2 = cp.dot(theory_a1, slm2_arr.T)

                theory_z1 = theory_z1.get()
                theory_a1 = theory_a1.get()
                theory_z2 = theory_z2.get()

                raw_a1s.append(raw_a1)
                theory_a1s.append(theory_z1)
                raw_z2s.append(raw_z2)
                theory_z2s.append(theory_z2)

                self.img1.set_array(self.ctrl.captures['a1'].frames[0])
                self.img2.set_array(self.ctrl.captures['z2'].frames[0])

                theory_a1_plot = np.abs(theory_a1[0, :].copy())
                theory_a1_plot -= theory_a1_plot.mean()
                theory_a1_plot /= theory_a1_plot.std()
                raw_a1_plot = raw_a1[0, :].copy()
                raw_a1_plot -= raw_a1_plot.mean()
                raw_a1_plot /= raw_a1_plot.std()

                self.th_line1.set_ydata(theory_a1_plot)
                self.meas_line1.set_ydata(raw_a1_plot)

                theory_z2_plot = np.abs(theory_z2[0, :].copy())
                theory_z2_plot -= theory_z2_plot.mean()
                theory_z2_plot /= theory_z2_plot.std()
                raw_z2_plot = raw_z2[0, :].copy()
                raw_z2_plot -= raw_z2_plot.mean()
                raw_z2_plot /= raw_z2_plot.std()

                self.th_line2.set_ydata(theory_z2_plot)
                self.meas_line2.set_ydata(raw_z2_plot)

                plt.pause(0.1)

            print()

        meas_z1 = np.load("./tools/calibration/meas_z1.npy")
        theory_z1 = np.load("./tools/calibration/theory_z1.npy")
        norm_params1 = np.load("./tools/calibration/norm_params1.npy")

        raw_a1 = np.array(raw_a1s).reshape(successful * self.ctrl.num_frames, self.m)
        theory_a1 = np.array(theory_a1s).reshape(successful * self.ctrl.num_frames, self.m)
        meas_a1 = raw_a1 * np.sign(theory_a1)
        meas_a1 = (meas_a1 - norm_params1[:, 1].copy()) / norm_params1[:, 0].copy()

        raw_z2 = np.array(raw_z2s).reshape(successful * self.ctrl.num_frames, self.k)
        theory_z2 = np.array(theory_z2s).reshape(successful * self.ctrl.num_frames, self.k)
        meas_z2 = raw_z2 * np.sign(theory_z2)
        norm_params2 = self.ctrl.find_norm_params(theory_z2, meas_z2)
        meas_z2 = (meas_z2 - norm_params2[:, 1].copy()) / norm_params2[:, 0].copy()

        error1 = (meas_a1 - theory_a1).std()
        error2 = (meas_z2 - theory_z2).std()

        print(colored(f'error1 : {error1:.3f}, signal1 : {theory_a1.std():.3f}, '
                      f'ratio1 : {theory_a1.std()/error1:.3f}', 'blue'))
        print(colored(f'error2 : {error2:.3f}, signal2 : {theory_z2.std():.3f}, '
                      f'ratio2 : {theory_z2.std()/error2:.3f}', 'blue'))

        self.axs0.cla()
        low = -3
        high = 3
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.axs0.plot(theory_z1, meas_z1, linestyle='', marker='x')
        self.axs0.plot(theory_z1, meas_a1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

        xs = np.linspace(low, high, 100)
        gs = satab(xs, od_fit, thresh_fit)
        self.axs0.plot(xs, gs, c='black', linewidth=1)

        self.axs1.cla()
        low = -10
        high = 10
        self.axs1.set_ylim(low, high)
        self.axs1.set_xlim(low, high)
        self.axs1.plot(theory_z2, meas_z2, linestyle='', marker='x')
        self.axs1.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs1.set_title('layer 2')
        self.axs1.set_xlabel('theory')
        self.axs1.set_ylabel('measured')

        plt.pause(0.1)
        plt.show()

        np.save("./tools/calibration/raw_a1.npy", raw_a1)
        np.save("./tools/calibration/meas_a1.npy", meas_a1)
        np.save("./tools/calibration/theory_a1.npy", theory_a1)

        np.save("./tools/calibration/raw_z2.npy", raw_z2)
        np.save("./tools/calibration/meas_z2.npy", meas_z2)
        np.save("./tools/calibration/theory_z2.npy", theory_z2)
        np.save("./tools/calibration/norm_params2.npy", norm_params2)

        plt.draw()
        plt.pause(1)
        plt.show()

        input('Press enter to continue...')

    def init_weights(self):

        slm_arr = cp.empty((self.n, self.m))
        slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
        slm_arr[0, :] = cp.array(self.dnn.b1.copy())

        slm2_arr = cp.zeros((self.k, self.m))*0.
        slm2_arr[2:4, :] = cp.array(self.dnn.w2.copy())

        self.ctrl.update_slm1(slm_arr, lut=True)
        self.ctrl.update_slm2(slm2_arr, lut=True)
        time.sleep(1)

    def run_batch(self, batch_num):

        #####################################
        # Start by running only forward MVM #
        #####################################

        xs = self.trainx[self.batch_indxs_list[batch_num], :].copy()
        ys = self.trainy[self.batch_indxs_list[batch_num], :].copy()

        self.dnn.xs = xs
        self.dnn.ys = ys

        self.theory_z1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
        self.theory_a1 = satab(self.theory_z1, self.od, self.thresh)
        self.theory_z2 = np.dot(self.theory_a1, self.dnn.w2.T)

        if self.forward == 'digital':
            self.dnn.z1 = self.theory_z1.copy()
            self.dnn.a1 = self.theory_a1.copy()
            self.dnn.z2 = self.theory_z2.copy()

        elif self.forward == 'optical':

            dmd_vecs = cp.ones((self.batch_size, self.n))*1.
            dmd_vecs[:, 1:] = cp.array(xs)
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            success = False
            fails = 0
            while not success:

                self.ctrl.run_batch(dmd_vecs)
                success = self.ctrl.captures['a1'].success and self.ctrl.captures['z2'].success

                if not success:
                    fails += 1
                    if fails == 5:
                        print(colored('FAILED BATCH', 'red'))
                        return False

            # print('batch successful')
            self.raw_a1 = self.ctrl.captures['a1'].ampls.copy()
            self.meas_a1 = self.raw_a1 * np.sign(self.theory_a1)
            self.meas_a1 = (self.meas_a1 - self.norm_params1[:, 1].copy()) / self.norm_params1[:, 0].copy()

            self.dnn.z1 = self.theory_z1.copy()
            self.dnn.a1 = self.meas_a1.copy()

            self.raw_z2 = self.ctrl.captures['z2'].ampls.copy()[:, 2:4]
            self.meas_z2 = self.raw_z2 * np.sign(self.theory_z2)
            self.meas_z2 = (self.meas_z2 - self.norm_params2[:, 1].copy()) / self.norm_params2[:, 0].copy()

            self.dnn.z2 = self.meas_z2.copy()

            error1 = (self.meas_a1 - self.theory_a1).std()
            error2 = (self.meas_z2 - self.theory_z2).std()
            self.errors1.append(error1)
            self.errors2.append(error2)
            # print(colored(f'error1 : {error1:.3f}', 'blue'))
            # print(colored(f'error2 : {error2:.3f}', 'blue'))

        ########################################################
        # Now calculate error vector and perform backward MVM  #
        ########################################################

        self.dnn.a2 = softmax(self.dnn.z2*self.sm_scaling)

        self.dnn.loss = error(self.dnn.a2, self.dnn.ys)

        self.loss.append(self.dnn.loss)
        # print(colored('loss : {:.2f}'.format(self.dnn.loss), 'green'))
        self.loss_plot.pop(0).remove()
        self.loss_plot = self.axsa.plot(self.loss, linestyle='-', marker='', c='b')

        self.dnn.pred = self.dnn.a2.copy().argmax(axis=1)
        self.dnn.label = self.dnn.ys.copy().argmax(axis=1)
        self.accs.append(accuracy(self.dnn.pred, self.dnn.label))

        self.dnn.a2_delta = (self.dnn.a2 - self.dnn.ys)/self.batch_size
        # self.dnn.a2_delta = self.dnn.a2_delta/np.abs(self.dnn.a2_delta).max()

        self.theory_back = np.dot(self.dnn.a2_delta, self.dnn.w2)
        self.theory_back *= satab_d(self.dnn.z1.copy(), self.od, self.thresh)

        if self.backward == 'digital':
            self.dnn.a1_delta = self.theory_back.copy()

        elif self.backward == 'optical':

            dmd_vecs = cp.ones((self.batch_size, self.n))
            dmd_vecs[:, 1:] = cp.array(self.dnn.xs)
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            dmd_errs = cp.array(self.dnn.a2_delta)
            # dmd_errs = dmd_errs/cp.abs(dmd_errs).max()
            dmd_errs = cp.clip(dmd_errs, -1, 1)
            dmd_errs = (dmd_errs*self.ctrl.dmd_err_block_w).astype(cp.int)/self.ctrl.dmd_err_block_w

            # dmd_errs = cp.ones((self.batch_size, self.k))

            self.meas_back = np.empty((self.batch_size, self.m))

            # We allow 5 attempts at running the batch. After that, skip the batch.
            ii = 0
            fails = 0
            while ii < num_repeats:

                self.ctrl.run_batch(dmd_vecs[ii*self.ctrl.num_frames:(ii+1)*self.ctrl.num_frames, :],
                                    dmd_errs[ii*self.ctrl.num_frames:(ii+1)*self.ctrl.num_frames, :])
                success, err = self.ctrl.check_ampls()
                if not success:
                    fails += 1
                    print(fails)
                    if fails == 5:
                        print(colored('FAILED BATCH', 'red'))
                        print()
                        return False, err
                    else:
                        continue
                else:
                    self.meas_back[ii*self.ctrl.num_frames:(ii+1)*self.ctrl.num_frames, :] = self.ctrl.ampls3.copy()
                    ii += 1

            self.meas_back = (self.meas_back - self.ctrl.norm_params3[:, 1].copy()) / self.ctrl.norm_params3[:, 0].copy()
            self.meas_back *= np.sign(self.theory_back)
            self.dnn.a1_delta = self.meas_back.copy()

            error3 = (self.meas_back - self.theory_back).std() #/np.abs(self.meas_back - self.theory_back).mean()
            self.errors3.append(error3)
            # print(colored(f'error3 : {error3:.3f}', 'blue'))

        ########################################
        # Calculate and perform weight update  #
        ########################################

        self.dnn.update_weights()

        self.dnn.w1 = np.clip(self.dnn.w1.copy(), -1, 1)
        self.dnn.w2 = np.clip(self.dnn.w2.copy(), -1, 1)
        self.dnn.b1 = np.clip(self.dnn.b1.copy(), -1, 1)

        # self.dnn.w1 /= self.dnn.w1.max()
        # self.dnn.w2 /= self.dnn.w2.max()
        # self.dnn.b1 /= self.dnn.b1.max()

        if self.forward == 'optical':

            slm_arr = cp.empty((self.n, self.m))
            slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
            slm_arr[0, :] = cp.array(self.dnn.b1.copy())

            slm2_arr = cp.zeros((self.k, self.m))*0.
            slm2_arr[2:4, :] = cp.array(self.dnn.w2.copy())

            self.ctrl.update_slm1(slm_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)

        return True

    def run_validation(self, epoch, grid=False):

        if grid:
            xs = self.gridx.copy()
            num_val_batches = 10
            folder = 'boundary'
        else:
            xs = self.testx.copy()
            num_val_batches = 2
            folder = 'validation'

        if self.forward == 'digital':
            z1 = np.dot(xs.copy(), self.dnn.w1) + self.dnn.b1
            a1 = satab(z1, self.od, self.thresh)
            z2 = np.dot(a1, self.dnn.w2.T)
            a2 = softmax(z2*self.sm_scaling)
            pred = a2.argmax(axis=1)

        elif self.forward == 'optical':

            xs_arr = xs.copy().reshape((num_val_batches, self.ctrl.num_frames, xs.shape[1]))
            self.raw_a1 = np.empty((num_val_batches, self.ctrl.num_frames, self.m))*0.
            self.raw_z2 = np.empty((num_val_batches, self.ctrl.num_frames, self.k))*0.

            for batch in range(num_val_batches):

                dmd_vecs = cp.ones((self.batch_size, self.n))*1.
                dmd_vecs[:, 1:] = cp.array(xs_arr[batch, ...])
                dmd_vecs = cp.clip(dmd_vecs, 0, 1)
                dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

                success = False
                fails = 0
                while not success:

                    self.ctrl.run_batch(dmd_vecs)
                    success = self.ctrl.captures['a1'].success and self.ctrl.captures['z2'].success

                    if not success:
                        fails += 1
                        if fails == 5:
                            print(colored('FAILED BATCH', 'red'))
                            return False

                # print('batch successful')
                self.raw_a1[batch, ...] = self.ctrl.captures['a1'].ampls.copy()
                self.raw_z2[batch, ...] = self.ctrl.captures['z2'].ampls.copy()

            self.theory_z1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
            self.theory_a1 = satab(self.theory_z1, self.od, self.thresh)
            self.meas_a1 = np.reshape(self.raw_a1, (num_val_batches*self.ctrl.num_frames, self.m))

            self.meas_a1 *= np.sign(self.theory_a1)
            self.meas_a1 = (self.meas_a1 - self.norm_params1[:, 1].copy()) / self.norm_params1[:, 0].copy()

            self.theory_z2 = np.dot(self.theory_a1, self.dnn.w2.T)
            self.raw_z2 = np.reshape(self.raw_z2, (num_val_batches*self.ctrl.num_frames, self.k))[:, 2:4]
            self.meas_z2 = self.raw_z2 * np.sign(self.theory_z2)
            self.meas_z2 = (self.meas_z2 - self.norm_params2[:, 1].copy()) / self.norm_params2[:, 0].copy()

            self.dnn.a1 = self.meas_a1.copy()
            self.dnn.z2 = self.meas_z2.copy()

            pred = softmax(self.meas_z2*self.sm_scaling).argmax(axis=1)

            np.save(self.save_folder+f'{folder}/raw_ampls1/raw_ampls_1_epoch{epoch}.npy', self.raw_a1)
            np.save(self.save_folder+f'{folder}/raw_ampls2/raw_ampls_2_epoch{epoch}.npy', self.raw_z2)
            np.save(self.save_folder+f'{folder}/meas_a1/meas_a1_epoch{epoch}.npy', self.meas_a1)
            np.save(self.save_folder+f'{folder}/meas_z2/meas_z2_epoch{epoch}.npy', self.meas_z2)
            np.save(self.save_folder+f'{folder}/theory_a1/theory_a1_epoch{epoch}.npy', self.theory_a1)
            np.save(self.save_folder+f'{folder}/theory_z2/theory_z2_epoch{epoch}.npy', self.theory_z2)

            [self.a1_scatter.pop(0).remove() for _ in range(self.m)]
            self.a1_scatter = self.axs0.plot(self.theory_z1, self.meas_a1, linestyle='', marker='x')
            [self.z2_scatter.pop(0).remove() for _ in range(2)]
            self.z2_scatter = self.axs1.plot(self.theory_z2, self.meas_z2, linestyle='', marker='x')

        if grid:

            np.save(self.save_folder+f'{folder}/predictions/pred_epoch{epoch}.npy', pred)

            xs0, ys0 = np.meshgrid(np.linspace(0, 1, self.res), np.linspace(0, 1, self.res))
            self.axse.contourf(xs0, ys0, pred.reshape((self.res, self.res)).T)
            self.axse.scatter(self.trainx[:, 0], self.trainx[:, 1], c=self.trainy[:, 0], cmap='winter')

        else:

            label = self.testy.copy().argmax(axis=1)

            np.save(self.save_folder+f'{folder}/predictions/pred_epoch{epoch}.npy', pred)
            np.save(self.save_folder+f'{folder}/labels/labels_epoch{epoch}.npy', label)

            acc = (pred == label).sum()*100/80
            self.accs.append(acc)

            np.save(self.save_folder+f'accuracies.npy', np.array(self.accs))

            self.accs_plot.pop(0).remove()
            self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='x', c='b')
            plt.draw()
            plt.pause(0.1)



        # if test_or_val == 'val':
        #     acc = accuracy(pred, label)
        #     self.accs.append(acc)
        #     np.save(self.save_folder+f'accuracies.npy', np.array(self.accs))
        #     np.save(self.save_folder+f'validation/predictions/predictions_epoch{epoch}.npy', pred)
        #     np.save(self.save_folder+f'validation/labels/labels_epoch{epoch}.npy', label)
        #
        # self.label_scatter.remove()
        # self.label_scatter = self.axsd.scatter(xs[:, 0], xs[:, 1], c=label)
        #
        # self.pred_scatter.remove()
        # self.pred_scatter = self.axse.scatter(xs[:, 0], xs[:, 1], c=pred)
        #
        # correct = pred == label
        # self.correct_scatter.remove()
        # self.correct_scatter = self.axsf.scatter(xs[:, 0], xs[:, 1], c=correct, cmap='RdYlGn')


    def save_batch(self, epoch, batch):

        if self.forward == 'optical':
            np.save(self.save_folder+f'training/frames1/frames1_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.captures['a1'].frames)
            np.save(self.save_folder+f'training/frames2/frames2_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.captures['z2'].frames)

            np.save(self.save_folder+f'training/raw_ampls1/raw_ampls1_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.captures['a1'].ampls)
            np.save(self.save_folder+f'training/raw_ampls2/raw_ampls2_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.captures['z2'].ampls)

            np.save(self.save_folder+f'training/meas_a1/meas_a1_epoch{epoch}_batch{batch}.npy',
                    self.meas_a1)
            np.save(self.save_folder+f'training/meas_z2/meas_z2_epoch{epoch}_batch{batch}.npy',
                    self.meas_z2)

            np.save(self.save_folder+f'training/theory_a1/theory_a1_epoch{epoch}_batch{batch}.npy',
                    self.theory_a1)
            np.save(self.save_folder+f'training/theory_z2/theory_z2_epoch{epoch}_batch{batch}.npy',
                    self.theory_z2)

        if self.backward == 'optical':

            np.save(self.save_folder+f'training/theory_back/theory_back_epoch{epoch}_batch{batch}.npy',
                    self.theory_back)
            np.save(self.save_folder+f'training/meas_back/meas_back_epoch{epoch}_batch{batch}.npy',
                    self.meas_back)
            np.save(self.save_folder+f'training/raw_ampls3/raw_ampls3_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.captures['back'].ampls)
            np.save(self.save_folder+f'training/frames3/frames3_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.captures['back'].frames)

        np.save(self.save_folder+f'training/xs/xs_epoch{epoch}_batch{batch}.npy',
                self.dnn.xs)
        np.save(self.save_folder+f'training/ys/ys_epoch{epoch}_batch{batch}.npy',
                self.dnn.ys)
        np.save(self.save_folder+f'training/w1s/w1_epoch{epoch}_batch{batch}.npy',
                self.dnn.w1)
        np.save(self.save_folder+f'training/b1s/b1_epoch{epoch}_batch{batch}.npy',
                self.dnn.b1)
        np.save(self.save_folder+f'training/w2s/w2_epoch{epoch}_batch{batch}.npy',
                self.dnn.w2)

        np.save(self.save_folder+f'loss.npy', np.array(self.loss))

    def graph_batch(self):

        frame = -1

        if self.forward == 'optical':

            self.th_line1.set_ydata(self.theory_a1[frame, :])
            self.meas_line1.set_ydata(self.meas_a1[frame, :])

            # self.th_line2.set_ydata(self.theory_z2[frame, :])
            # self.meas_line2.set_ydata(self.meas_z2[frame, :])
            # self.softmax_line2.set_ydata(self.dnn.a2[frame, :])
            # self.label_line2.set_ydata(self.dnn.ys[frame, :])

            [self.a1_scatter.pop(0).remove() for _ in range(self.m)]
            self.a1_scatter = self.axs0.plot(self.theory_z1, self.meas_a1, linestyle='', marker='x')
            try:
                [self.z2_scatter.pop(0).remove() for _ in range(2)]
            except AttributeError:
                pass
            self.z2_scatter = self.axs1.plot(self.theory_z2, self.meas_z2, linestyle='', marker='x')

            self.img1.set_array(self.ctrl.captures['a1'].frames[frame])
            self.img2.set_array(self.ctrl.captures['z2'].frames[frame])

            self.err1_plot.pop(0).remove()
            self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
            self.err2_plot.pop(0).remove()
            self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')

        self.accs_plot.pop(0).remove()
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='', c='b')

        if self.backward == 'optical':

            self.th_line3.set_ydata(self.theory_back[frame, :])
            self.meas_line3.set_ydata(self.meas_back[frame, :])

            [self.back_scatter.pop(0).remove() for _ in range(self.m)]
            self.back_scatter = self.axs2.plot(self.theory_back, self.meas_back, linestyle='', marker='x')

            self.img3.set_array(self.ctrl.frames3[frame])

            self.err3_plot.pop(0).remove()
            self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')


if __name__ == "__main__":

    n, m, k = 3, 10, 5

    batch_size = 5
    num_batches = 10
    num_epochs = 20
    lr = 0.05
    scaling = 15

    slm_arr = np.random.normal(0, 0.5, (n, m))
    slm_arr = np.clip(slm_arr, -1, 1)
    slm_arr = (slm_arr*64).astype(np.int)/64

    slm2_arr = np.random.normal(0, 0.5, (k, m))
    slm2_arr = np.clip(slm2_arr, -1, 1)
    slm2_arr = (slm2_arr*64).astype(np.int)/64

    onn = MyONN(batch_size=batch_size, num_batches=num_batches, num_epochs=num_epochs,
                w1_0=slm_arr[1:, :], w2_0=slm2_arr, b1_0=slm_arr[0, :],
                lr=lr, scaling=scaling, dimensions=(n, m, k),
                save_folder=None,
                trainx=None, testx=None, trainy=None,
                forward='optical', backward='digital')

    onn.run_calibration_linear()
    onn.run_calibration_nonlinear_layer1()
    onn.run_calibration_nonlinear_layer2()

    plt.show(block=True)

    print('done')
