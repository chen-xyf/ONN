from ANN import DNN, DNN_1d, accuracy, softmax, error, DNN_backprop
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


class MyONN:

    def __init__(self, lr, num_batches, num_epochs, ctrl_vars):

        self.ctrl = Controller(*ctrl_vars)

        self.n = self.ctrl.n
        self.m = self.ctrl.m
        self.k = self.ctrl.k
        self.batch_size = self.ctrl.num_frames
        self.num_frames = self.ctrl.num_frames

        self.lr = lr
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.loop_clock = clock.Clock()

        self.accs = []
        self.loss = [0.5]
        self.errors1 = []
        self.errors2 = []
        self.errors3 = []
        self.best_w1 = None
        self.best_w2 = None
        self.best_b1 = None

        self.dnn = None

        self.w1 = None
        self.w2 = None
        self.b1 = None

        self.trainx = None
        # self.valx = None
        self.testx = None
        self.trainy = None
        # self.valy = None
        self.testy = None

        self.batch_indxs_list = None

    def graphs(self):

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2, self.axsa],
                    [self.axs3, self.axs4, self.axs5, self.axsb],
                    [self.axs6, self.axs7, self.axs8, self.axsc]] = plt.subplots(3, 4, figsize=(24, 12))

        self.axs3.set_ylim(-5, 5)
        self.axs4.set_ylim(-5, 5)
        self.axs5.set_ylim(-5, 5)
        self.axsa.set_ylim(0, 0.5)
        self.axsb.set_ylim(0, 100)

        self.th_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]
        self.th_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='o', c='b')[0]
        self.meas_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='x', c='r')[0]
        self.th_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        self.img1 = self.axs6.imshow(np.zeros((120, 672)), aspect='auto', vmin=0, vmax=255)
        self.img2 = self.axs7.imshow(np.zeros((90, 808)), aspect='auto', vmin=0, vmax=255)
        self.img3 = self.axs8.imshow(np.zeros((120, 672)), aspect='auto', vmin=0, vmax=255)

        plt.draw()
        plt.pause(2)

    def run_calibration(self, initial=True):

        if not initial:
            slm_arr = self.dnn.w1.copy()
            slm2_arr = self.dnn.w2.copy()
            self.ctrl.update_slm1(slm_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)
            time.sleep(0.5)

        meass_a1 = []
        theorys_a1 = []
        meass_z2 = []
        theorys_z2 = []
        meass_back = []
        theorys_back = []

        succeeded = 0
        while succeeded < 5:

            print(succeeded)
            cp.random.seed(succeeded)

            dmd_vecs = cp.random.normal(0.5, 0.4, (self.num_frames, self.n))
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            dmd_errs = cp.random.normal(0., 0.5, (self.num_frames, self.k))
            dmd_errs = cp.clip(dmd_errs, -1, 1)
            dmd_errs = (dmd_errs*self.ctrl.dmd_err_block_w).astype(cp.int)/self.ctrl.dmd_err_block_w

            if initial:
                slm_arr = cp.random.normal(0, 0.5, (self.n, self.m))
                slm_arr = cp.clip(slm_arr, -1, 1)
                slm_arr = (slm_arr*64).astype(cp.int)/64

                slm2_arr = cp.random.normal(0, 0.5, (self.k, self.m))
                slm2_arr = cp.clip(slm2_arr, -1, 1)
                slm2_arr = (slm2_arr*64).astype(cp.int)/64

                self.ctrl.update_slm1(slm_arr, lut=True)
                self.ctrl.update_slm2(slm2_arr, lut=True)

            time.sleep(0.5)

            self.ctrl.run_batch(dmd_vecs, dmd_errs)
            success = self.ctrl.check_ampls()
            time.sleep(0.1)

            if success:
                meas_a1 = self.ctrl.ampls1.copy()
                theory_a1 = cp.dot(dmd_vecs, slm_arr)

                meas_z2 = self.ctrl.ampls2.copy()
                theory_z2 = cp.dot(theory_a1, slm2_arr.T)

                meas_back = self.ctrl.ampls3.copy()
                theory_back = cp.dot(dmd_errs, slm2_arr)

                theory_a1 = theory_a1.get()
                theory_z2 = theory_z2.get()
                theory_back = theory_back.get()

                meass_a1.append(meas_a1)
                theorys_a1.append(theory_a1)
                meass_z2.append(meas_z2)
                theorys_z2.append(theory_z2)
                meass_back.append(meas_back)
                theorys_back.append(theory_back)

                self.img1.set_array(self.ctrl.frames1[0])
                self.img2.set_array(self.ctrl.frames2[0])
                self.img3.set_array(self.ctrl.frames3[0])

                theory_a1_plot = np.abs(theory_a1[0, :].copy())
                # theory_a1_plot /= theory_a1_plot.max()
                theory_a1_plot -= theory_a1_plot.mean()
                theory_a1_plot /= theory_a1_plot.std()
                meas_a1_plot = meas_a1[0, :].copy()
                # meas_a1_plot /= meas_a1_plot.max()
                meas_a1_plot -= meas_a1_plot.mean()
                meas_a1_plot /= meas_a1_plot.std()
                self.th_line1.set_ydata(theory_a1_plot)
                self.meas_line1.set_ydata(meas_a1_plot)

                theory_z2_plot = np.abs(theory_z2[0, :].copy())
                theory_z2_plot -= theory_z2_plot.mean()
                theory_z2_plot /= theory_z2_plot.std()
                meas_z2_plot = meas_z2[0, :].copy()
                meas_z2_plot -= meas_z2_plot.mean()
                meas_z2_plot /= meas_z2_plot.std()
                self.th_line2.set_ydata(theory_z2_plot)
                self.meas_line2.set_ydata(meas_z2_plot)

                theory_back_plot = np.abs(theory_back[0, :].copy())
                theory_back_plot -= theory_back_plot.mean()
                theory_back_plot /= theory_back_plot.std()
                meas_back_plot = meas_back[0, :].copy()
                meas_back_plot -= meas_back_plot.mean()
                meas_back_plot /= meas_back_plot.std()
                self.th_line3.set_ydata(theory_back_plot)
                self.meas_line3.set_ydata(meas_back_plot)

                plt.pause(0.01)

                succeeded += 1

        meas_a1 = np.array(meass_a1).reshape(5 * self.num_frames, self.m)
        theory_a1 = np.array(theorys_a1).reshape(5 * self.num_frames, self.m)
        meas_z2 = np.array(meass_z2).reshape(5 * self.num_frames, self.k)
        theory_z2 = np.array(theorys_z2).reshape(5 * self.num_frames, self.k)
        meas_back = np.array(meass_back).reshape(5 * self.num_frames, self.m)
        theory_back = np.array(theorys_back).reshape(5 * self.num_frames, self.m)

        if initial:
            self.ctrl.norm_params1 = self.ctrl.find_norm_params(theory_a1, meas_a1)
            self.ctrl.norm_params2 = self.ctrl.find_norm_params(theory_z2, meas_z2)
            self.ctrl.norm_params3 = self.ctrl.find_norm_params(theory_back, meas_back)
        else:
            self.ctrl.norm_params1 = self.ctrl.update_norm_params(theory_a1, meas_a1, self.ctrl.norm_params1)
            self.ctrl.norm_params2 = self.ctrl.update_norm_params(theory_z2, meas_z2, self.ctrl.norm_params2)
            self.ctrl.norm_params3 = self.ctrl.update_norm_params(theory_back, meas_back, self.ctrl.norm_params2)

        meas_a1 = self.ctrl.normalise_ampls(ampls=meas_a1, norm_params=self.ctrl.norm_params1)
        meas_z2 = self.ctrl.normalise_ampls(ampls=meas_z2, norm_params=self.ctrl.norm_params2)
        meas_back = self.ctrl.normalise_ampls(ampls=meas_back, norm_params=self.ctrl.norm_params3)

        meas_a1 *= np.sign(theory_a1)
        meas_z2 *= np.sign(theory_z2)
        meas_back *= np.sign(theory_back)

        error1 = (meas_a1 - theory_a1).std()
        error2 = (meas_z2 - theory_z2).std()
        error3 = (meas_back - theory_back).std()

        print(colored(f'error1 : {error1:.3f}, signal1 : {theory_a1.std():.3f}, '
                      f'ratio1 : {theory_a1.std()/error1:.3f}', 'blue'))
        print(colored(f'error2 : {error2:.3f}, signal2 : {theory_z2.std():.3f}, '
                      f'ratio2 : {theory_z2.std()/error2:.3f}', 'blue'))
        print(colored(f'error3 : {error3:.3f}, signal3 : {theory_back.std():.3f}, '
                      f'ratio3 : {theory_back.std()/error3:.3f}', 'blue'))
        print()

        self.axs0.cla()
        low = -2
        high = 2
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.axs0.plot(theory_a1, meas_a1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

        self.axs1.cla()
        low = -3
        high = 3
        self.axs1.set_ylim(low, high)
        self.axs1.set_xlim(low, high)
        self.axs1.plot(theory_z2, meas_z2, linestyle='', marker='x')
        self.axs1.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs1.set_title('layer 2')
        self.axs1.set_xlabel('theory')
        self.axs1.set_ylabel('measured')

        self.axs2.cla()
        low = -2
        high = 2
        self.axs2.set_ylim(low, high)
        self.axs2.set_xlim(low, high)
        self.axs2.plot(theory_back, meas_back, linestyle='', marker='x')
        self.axs2.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs2.set_title('backprop')
        self.axs2.set_xlabel('theory')
        self.axs2.set_ylabel('measured')

        plt.draw()
        # plt.pause(1)
        plt.show()

    def init_onn(self):

        m_dw1 = np.zeros((self.ctrl.n - 1, self.ctrl.m))
        v_dw1 = np.zeros((self.ctrl.n - 1, self.ctrl.m))

        m_db1 = np.zeros(self.ctrl.m)
        v_db1 = np.zeros(self.ctrl.m)

        m_dw2 = np.zeros((self.ctrl.m, 1))
        v_dw2 = np.zeros((self.ctrl.m, 1))

        beta1 = 0.9
        beta2 = 0.999

        adam_params = (m_dw1, v_dw1, m_db1, v_db1, m_dw2, v_dw2, beta1, beta2)

        self.dnn = DNN_backprop(*adam_params, w1_0=self.w1, w2_0=self.w2, b1_0=self.b1,
                                batch_size=self.batch_size, num_batches=self.num_batches,
                                lr=self.lr)

        self.axs5.plot(self.accs, linestyle='-', marker='x', c='g')
        self.axs2.plot(self.loss, linestyle='-', marker='', c='r')
        self.axs4.plot(self.errors1, linestyle='-', marker='', c='b')
        self.axs4.plot(self.errors2, linestyle='-', marker='', c='g')

        self.loop_clock.tick()

    # def init_epoch(self, epoch_num):
    #
    #     self.ctrl.update_slm1(self.w1.copy(), lut=True)
    #     self.ctrl.update_slm2(self.w2.copy(), lut=True)
    #     time.sleep(1)
    #
    #     rng = np.random.default_rng(epoch_num)
    #     epoch_rand_indxs = np.arange(self.trainx.shape[0])
    #     rng.shuffle(epoch_rand_indxs)
    #     self.batch_indxs_list = [epoch_rand_indxs[i * self.batch_size: (i + 1) * self.batch_size]
    #                              for i in range(self.num_batches)]
    #
    #     self.ctrl.init_dmd()
    #
    #     all_z1s = []
    #     all_theories1 = []
    #     all_z2s = []
    #     all_theories2 = []
    #
    #     passed = 0
    #     failed = 0
    #     while passed < 3:
    #
    #         batch_indxs = np.random.randint(0, self.trainx.shape[0], self.batch_size)
    #
    #         xs = self.trainx[batch_indxs, :].copy()
    #         vecs = cp.array(xs)
    #
    #         success = self.ctrl.run_batch_both(vecs, normalisation=True)
    #
    #         if success:
    #             theories1 = np.dot(xs, self.dnn.w1.copy())
    #             theories2 = np.dot(theories1, self.dnn.w2.copy())
    #             all_z1s.append(self.ctrl.z1s)
    #             all_theories1.append(theories1)
    #             all_z2s.append(self.ctrl.z2s)
    #             all_theories2.append(theories2)
    #             passed += 1
    #         else:
    #             failed += 1
    #
    #         if failed > 3:
    #             raise TimeoutError
    #
    #     all_z1s = np.array(all_z1s).reshape(3 * self.batch_size, self.m)
    #     all_theories1 = np.array(all_theories1).reshape(3 * self.batch_size, self.m)
    #     all_z2s = np.array(all_z2s).reshape(3 * self.batch_size, 1)
    #     all_theories2 = np.array(all_theories2).reshape(3 * self.batch_size, 1)
    #
    #     self.ctrl.update_norm_params(all_theories1, all_z1s, all_theories2, all_z2s)
    #
    #     self.ctrl.update_slm1(self.dnn.w1.copy(), lut=True)
    #     self.ctrl.update_slm2(self.dnn.w2.copy(), lut=True)
    #
    #     time.sleep(1)

    def run_batch(self, batch_num):

        print('Batch ', batch_num)
        print()
        t0 = time.perf_counter()

        ##### Start by running only forward MVM #####

        xs = self.trainx[self.batch_indxs_list[batch_num], :].copy()
        ys = self.trainy[self.batch_indxs_list[batch_num], :].copy()

        # print(xs)
        print(self.dnn.w1.min(), self.dnn.w1.max())

        self.dnn.xs = xs
        self.dnn.ys = ys

        dmd_vecs = cp.ones((self.num_frames, self.n))
        dmd_vecs[:, 1:] = cp.array(xs)
        dmd_vecs = cp.clip(dmd_vecs, 0, 1)
        dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

        self.ctrl.run_batch(dmd_vecs)
        success = self.ctrl.check_ampls(a1=True, z2=True, back=False)
        if not success:
            print('oh no')
            return False

        meas_a1 = self.ctrl.ampls1.copy()
        meas_a1 = self.ctrl.normalise_ampls(ampls=meas_a1, norm_params=self.ctrl.norm_params1)
        theory_a1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
        meas_a1 *= np.sign(theory_a1)
        self.dnn.a1 = meas_a1.copy()

        meas_z2 = self.ctrl.ampls2.copy()
        meas_z2 = self.ctrl.normalise_ampls(ampls=meas_z2, norm_params=self.ctrl.norm_params2)
        theory_z2 = np.dot(theory_a1, self.dnn.w2.T)
        meas_z2 *= np.sign(theory_z2)
        self.dnn.z2 = meas_z2.copy()

        error1 = (meas_a1 - theory_a1).std()
        error2 = (meas_z2 - theory_z2).std()
        self.errors1.append(error1)
        self.errors2.append(error2)
        print(colored(f'error1 : {error1:.3f}, error2 : {error2:.3f}', 'blue'))

        self.th_line1.set_ydata(theory_a1[0, :])
        self.meas_line1.set_ydata(meas_a1[0, :])
        self.th_line2.set_ydata(theory_z2[0, :])
        self.meas_line2.set_ydata(meas_z2[0, :])

        self.dnn.a2 = softmax(self.dnn.z2)

        ##### Now calculate error vector and perform backward MVM  #####

        self.dnn.loss = error(self.dnn.a2, self.dnn.ys)
        self.dnn.a2_delta = (self.dnn.a2 - self.dnn.ys) #/self.batch_size
        self.dnn.a2_delta = self.dnn.a2_delta/np.abs(self.dnn.a2_delta).max()

        dmd_vecs = cp.ones((self.num_frames, self.n))
        dmd_vecs[:, 1:] = cp.array(self.dnn.xs)
        dmd_vecs = cp.clip(dmd_vecs, 0, 1)
        dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

        dmd_errs = cp.array(self.dnn.a2_delta)
        # dmd_errs = dmd_errs/cp.abs(dmd_errs).max()
        dmd_errs = cp.clip(dmd_errs, -1, 1)
        dmd_errs = (dmd_errs*self.ctrl.dmd_err_block_w).astype(cp.int)/self.ctrl.dmd_err_block_w

        self.ctrl.run_batch(dmd_vecs, dmd_errs)
        success = self.ctrl.check_ampls()
        if not success:
            print('oh no')
            return False

        meas_a1 = self.ctrl.ampls1.copy()
        meas_a1 = self.ctrl.normalise_ampls(ampls=meas_a1, norm_params=self.ctrl.norm_params1)
        theory_a1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
        meas_a1 *= np.sign(theory_a1)
        self.dnn.a1 = meas_a1.copy()

        meas_z2 = self.ctrl.ampls2.copy()
        meas_z2 = self.ctrl.normalise_ampls(ampls=meas_z2, norm_params=self.ctrl.norm_params2)
        theory_z2 = np.dot(theory_a1, self.dnn.w2.T)
        meas_z2 *= np.sign(theory_z2)
        self.dnn.z2 = meas_z2.copy()

        meas_back = self.ctrl.ampls3.copy()
        meas_back = self.ctrl.normalise_ampls(ampls=meas_back, norm_params=self.ctrl.norm_params3)
        theory_back = np.dot(self.dnn.a2_delta, self.dnn.w2)
        meas_back *= np.sign(theory_back)
        self.dnn.a1_delta = meas_back.copy()

        error1 = (meas_a1 - theory_a1).std()
        error2 = (meas_z2 - theory_z2).std()
        error3 = (meas_back - theory_back).std()
        self.errors1.append(error1)
        self.errors2.append(error2)
        self.errors3.append(error3)
        print(colored(f'error1 : {error1:.3f}, error2 : {error2:.3f},  error3 : {error3:.3f}', 'blue'))
        print()

        self.th_line1.set_ydata(theory_a1[0, :])
        self.meas_line1.set_ydata(meas_a1[0, :])
        self.th_line2.set_ydata(theory_z2[0, :])
        self.meas_line2.set_ydata(meas_z2[0, :])
        self.th_line3.set_ydata(theory_back[0, :])
        self.meas_line3.set_ydata(meas_back[0, :])

        self.img1.set_array(self.ctrl.frames1[0])
        self.img2.set_array(self.ctrl.frames2[0])
        self.img3.set_array(self.ctrl.frames3[0])

        self.axsa.cla()
        self.axsa.set_ylim(0, 0.5)
        self.axsa.plot(self.loss)

        ##### Calculate and perform weight update  #####

        self.dnn.update_weights()

        self.dnn.w1 = np.clip(self.dnn.w1.copy(), -1, 1)
        self.dnn.w2 = np.clip(self.dnn.w2.copy(), -1, 1)
        self.dnn.b1 = np.clip(self.dnn.b1.copy(), -1, 1)

        slm_arr = cp.empty((self.n, self.m))
        slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
        slm_arr[0, :] = cp.array(self.dnn.b1.copy())

        slm2_arr = cp.array(self.dnn.w2.copy())

        self.ctrl.update_slm1(slm_arr, lut=True)
        self.ctrl.update_slm2(slm2_arr, lut=True)
        #
        # time.sleep(1)

        if self.dnn.loss < self.loss[-1]:
            self.best_w1 = self.dnn.w1.copy()
            self.best_w2 = self.dnn.w2.copy()
            self.best_b1 = self.dnn.b1.copy()

        self.loss.append(self.dnn.loss)
        print(colored('loss : {:.2f}'.format(self.dnn.loss), 'green'))

        t1 = time.perf_counter()
        print('batch time: {:.2f}'.format(t1 - t0))

        dt = self.loop_clock.tick()
        print(colored(dt, 'yellow'))
        print()

    def run_validation(self, epoch_num):

        self.dnn.w1 = self.best_w1.copy()
        self.dnn.w2 = self.best_w2.copy()
        self.dnn.b1 = self.best_b1.copy()

        slm_arr = cp.empty((self.n, self.m))
        slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
        slm_arr[0, :] = cp.array(self.dnn.b1.copy())

        slm2_arr = cp.array(self.dnn.w2.copy())

        self.ctrl.update_slm1(slm_arr, lut=True)
        self.ctrl.update_slm2(slm2_arr, lut=True)
        time.sleep(1)

        print()
        t0 = time.perf_counter()

        xs = self.testx.copy().reshape((10, 10, 2))
        ys = self.testy.copy().argmax(axis=1).reshape((10, 10))

        all_meas = np.empty((10, 10))

        for batch in range(10):

            print(batch)

            dmd_vecs = cp.ones((self.num_frames, self.n))
            dmd_vecs[:, 1:] = cp.array(xs[batch])
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            self.ctrl.run_batch(dmd_vecs)
            success = self.ctrl.check_ampls(a1=True, z2=True, back=False)
            if not success:
                print('oh no')
                return False

            meas_a1 = self.ctrl.ampls1.copy()
            meas_a1 = self.ctrl.normalise_ampls(ampls=meas_a1, norm_params=self.ctrl.norm_params1)
            theory_a1 = np.dot(xs[batch], self.dnn.w1) + self.dnn.b1
            meas_a1 *= np.sign(theory_a1)

            meas_z2 = self.ctrl.ampls2.copy()
            meas_z2 = self.ctrl.normalise_ampls(ampls=meas_z2, norm_params=self.ctrl.norm_params2)
            theory_z2 = np.dot(theory_a1, self.dnn.w2.T)
            meas_z2 *= np.sign(theory_z2)

            all_meas[batch, ...] = softmax(meas_z2).argmax(axis=1)

            self.th_line1.set_ydata(theory_a1[0, :])
            self.meas_line1.set_ydata(meas_a1[0, :])
            self.th_line2.set_ydata(theory_z2[0, :])
            self.meas_line2.set_ydata(meas_z2[0, :])

        pred = all_meas.reshape(100)
        label = ys.copy().reshape(100)

        acc = accuracy(pred, label)
        self.accs.append(acc)
        self.axsb.cla()
        self.axsb.set_ylim(0, 100)
        self.axsb.plot(self.accs)
        print(colored(acc, 'green'))

    def save_data_batch(self, epoch_num, batch_num):

        np.save('D:/MNIST/raw_images/training/images/images_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.ctrl.frames)
        np.save('D:/MNIST/data/training/ampls/ampls_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.ctrl.ampls)

        np.save('D:/MNIST/data/loss.npy', np.array(self.loss))

        np.save('D:/MNIST/data/training/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.measured)
        np.save('D:/MNIST/data/training/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.theory)

        np.save('D:/MNIST/data/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(self.dnn.w1))
        np.save('D:/MNIST/data/w2/w2_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(self.dnn.w2))

    def save_data_epoch(self):

        np.save('D:/MNIST/data/accuracy.npy', np.array(self.accs))
        np.save('D:/MNIST/data/best_w1.npy', self.best_w1)
        np.save('D:/MNIST/data/best_w2.npy', self.best_w2)


if __name__ == "__main__":

    n, m, k = 3, 10, 3
    num_frames = 10
    num_batches = 20
    num_epochs = 20
    ctrl_vars = (n, m, k, num_frames)

    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0,
                               scale=1.0, shuffle=True, random_state=6)

    y_onehot = np.zeros((500, 3))

    for i in range(500):
        y_onehot[i, int(y[i])] = 1

    trainy = y_onehot[:400, :].copy()
    testy = y_onehot[400:, :].copy()

    xnorm = x.copy()
    xnorm[:, 0] -= xnorm[:, 0].min()
    xnorm[:, 0] *= 1./xnorm[:, 0].max()
    xnorm[:, 1] -= xnorm[:, 1].min()
    xnorm[:, 1] *= 1./xnorm[:, 1].max()

    trainx = xnorm[:400, :].copy()
    testx = xnorm[400:, :].copy()

    slm_arr = np.random.normal(0, 0.5, (n, m))
    slm_arr = np.clip(slm_arr, -1, 1)
    slm_arr = (slm_arr*64).astype(np.int)/64

    slm2_arr = np.random.normal(0, 0.5, (k, m))
    slm2_arr = np.clip(slm2_arr, -1, 1)
    slm2_arr = (slm2_arr*64).astype(np.int)/64

    onn = MyONN(lr=0.01, num_batches=num_batches, num_epochs=num_epochs, ctrl_vars=ctrl_vars)

    onn.w1 = slm_arr[1:, :]
    onn.w2 = slm2_arr
    onn.b1 = slm_arr[0, :]

    onn.graphs()
    onn.init_onn()
    onn.run_calibration(initial=True)

    onn.trainx = trainx
    onn.trainy = trainy
    onn.testx = testx
    onn.testy = testy

    for epoch_num in range(num_epochs):

        rng = np.random.default_rng(epoch_num)
        epoch_rand_indxs = np.arange(onn.trainx.shape[0])
        rng.shuffle(epoch_rand_indxs)
        onn.batch_indxs_list = [epoch_rand_indxs[i * onn.batch_size: (i + 1) * onn.batch_size]
                                for i in range(onn.num_batches)]

        for batch in range(num_batches):
            onn.run_batch(batch)

        onn.run_validation(epoch_num)

    plt.show(block=True)

    print('done')
