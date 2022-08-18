from ANN import DNN, DNN_1d, accuracy
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


class MyMVM:

    def __init__(self, num_batches, ctrl_vars):

        self.ctrl = Controller(*ctrl_vars)

        self.n = self.ctrl.n
        self.m = self.ctrl.m
        self.k = self.ctrl.k
        self.num_frames = self.ctrl.num_frames

        self.num_batches = num_batches
        self.loop_clock = clock.Clock()

    def run_mvm(self):

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2],
                    [self.axs3, self.axs4, self.axs5],
                    [self.axs6, self.axs7, self.axs8]] = plt.subplots(3, 3, figsize=(24, 12))

        self.axs3.set_ylim(-5, 5)
        self.axs4.set_ylim(-5, 5)
        self.axs5.set_ylim(-5, 5)

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

        meass_a1 = []
        theorys_a1 = []
        meass_z2 = []
        theorys_z2 = []
        meass_back = []
        theorys_back = []
        frames1 = []
        frames2 = []
        frames3 = []

        passed = 0
        # failed = 0
        # while passed < self.num_batches:
        for loop in range(self.num_batches):

            print()
            print(loop)
            print()

            cp.random.seed(loop)

            dmd_vecs = cp.random.normal(0.5, 0.4, (self.num_frames, self.n)) #*0. + 1.
            # dmd_vecs = cp.random.normal(0.5, 0.4, (1, self.n)).repeat(self.num_frames, axis=0)
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            dmd_errs = cp.random.normal(0., 0.5, (self.num_frames, self.k))
            # dmd_errs = cp.random.normal(0., 0.5, (1, self.k)).repeat(self.num_frames, axis=0)
            dmd_errs = cp.clip(dmd_errs, -1, 1)
            dmd_errs = (dmd_errs*self.ctrl.dmd_err_block_w).astype(cp.int)/self.ctrl.dmd_err_block_w

            slm_arr = cp.random.normal(0, 0.5, (self.n, self.m)) #*0. + 1.
            slm_arr = cp.clip(slm_arr, -1, 1)
            slm_arr = (slm_arr*64).astype(cp.int)/64

            self.ctrl.update_slm1(slm_arr, lut=True)

            slm2_arr = cp.random.normal(0, 0.5, (self.k, self.m)) #*0. + 0.2
            slm2_arr = cp.clip(slm2_arr, -1, 1)
            slm2_arr = (slm2_arr*64).astype(cp.int)/64
            # slm2_arr[-1, :] = 0

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

                frames1.append(self.ctrl.frames1)
                frames2.append(self.ctrl.frames2)
                frames3.append(self.ctrl.frames3)

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

                passed += 1

            #     passed += 1
            # else:
            #     failed += 1
            #
            # if failed > 10:
            #     raise TimeoutError

        meas_a1 = np.array(meass_a1).reshape(passed * self.num_frames, self.m)
        theory_a1 = np.array(theorys_a1).reshape(passed * self.num_frames, self.m)
        meas_z2 = np.array(meass_z2).reshape(passed * self.num_frames, self.k)
        theory_z2 = np.array(theorys_z2).reshape(passed * self.num_frames, self.k)
        meas_back = np.array(meass_back).reshape(passed * self.num_frames, self.m)
        theory_back = np.array(theorys_back).reshape(passed * self.num_frames, self.m)

        # np.save('./tools/temp_z1s.npy', meas_z1)
        # np.save('./tools/temp_theories1.npy', theory_z1)
        # np.save('./tools/temp_z2s.npy', meas_z2)
        # np.save('./tools/temp_theories2.npy', theory_z2)
        # np.save('./tools/temp_backs.npy', meas_back)
        # np.save('./tools/temp_theories_back.npy', theory_back)

        def line(x, grad, c):
            return (grad * x) + c

        params_a1 = np.array([curve_fit(line, np.abs(theory_a1[:, j]), meas_a1[:, j])[0]
                              for j in range(theory_a1.shape[1])])
        params_z2 = np.array([curve_fit(line, np.abs(theory_z2[:, j]), meas_z2[:, j])[0]
                              for j in range(theory_z2.shape[1])])
        params_back = np.array([curve_fit(line, np.abs(theory_back[:, j]), meas_back[:, j])[0]
                                for j in range(theory_back.shape[1])])

        meas_a1 = (meas_a1 - params_a1[:, 1].copy()) / params_a1[:, 0].copy()
        meas_z2 = (meas_z2 - params_z2[:, 1].copy()) / params_z2[:, 0].copy()
        meas_back = (meas_back - params_back[:, 1].copy()) / params_back[:, 0].copy()

        meas_a1 *= np.sign(theory_a1)
        meas_z2 = meas_z2 * np.sign(theory_z2)
        meas_back *= np.sign(theory_back)
        #
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


        err_z1 = theory_a1-meas_a1
        print(theory_a1.std(), meas_a1.std(), err_z1.std(), theory_a1.std()/err_z1.std())

        err_z2 = theory_z2-meas_z2
        print(theory_z2.std(), meas_z2.std(), err_z2.std(), meas_z2.std()/err_z2.std())

        err_back = theory_back-meas_back
        print(theory_back.std(), meas_back.std(), err_back.std(), meas_back.std()/err_back.std())

        plt.draw()
        # plt.pause(1)
        plt.show(block=True)

    def run_ampl_calib_w1(self):

        res = 64
        ampls = np.linspace(0.1, 1, res)

        meas = np.empty((res, self.n, self.m))

        dmd_vecs = cp.zeros((self.num_frames, self.n))
        for ii in range(self.n):
            dmd_vecs[ii, ii] = 1

        for indx, ampl in enumerate(ampls):

            print(indx)

            slm_arr = cp.ones((self.n, self.m))*ampl
            self.ctrl.update_slm1(slm_arr, lut=True)
            time.sleep(0.8)

            print(dmd_vecs.shape)
            self.ctrl.run_batch(dmd_vecs)
            time.sleep(0.1)

            success = self.ctrl.check_ampls(a1=True, z2=False, back=False)

            if success:
                meas[indx, :, :] = self.ctrl.ampls1.copy()
            else:
                raise RuntimeError

        np.save('./tools/w1_ampl_calib.npy', meas)

    def run_phase_calib_w1(self):

        res = 32
        phis = np.linspace(0, 2, res)

        meas = np.empty((res, self.n, self.m))

        dmd_vecs = cp.zeros((self.num_frames, self.n))
        dmd_vecs[:, 0] = 1
        for ii in range(self.n):
            dmd_vecs[ii, ii] = 1

        for indx, phi in enumerate(phis):

            print(indx)

            slm_arr = cp.ones((self.n, self.m))*0.6
            slm_phi = cp.zeros((self.n, self.m))*0.
            slm_phi[1:, :] = phi*np.pi
            self.ctrl.update_slm1(slm_arr, slm_phi, lut=True)
            time.sleep(0.8)

            print(dmd_vecs.shape)
            success = self.ctrl.run_batch(dmd_vecs)
            time.sleep(0.1)

            success = self.ctrl.check_ampls(a1=True, z2=False, back=False)

            if success:
                meas[indx, :, :] = self.ctrl.ampls1.copy()
            else:
                raise RuntimeError

        np.save('./tools/w1_phase_calib.npy', meas)

if __name__ == "__main__":

    n, m, k = 3, 10, 3
    num_frames = 10
    ctrl_vars = (n, m, k, num_frames)

    t0 = time.time()

    mvm = MyMVM(num_batches=5, ctrl_vars=ctrl_vars)
    mvm.run_mvm()
    # mvm.run_phase_calib_w1()
    # mvm.run_ampl_calib_w1()
    # time.sleep(1)
    plt.show(block=True)

    t1 = time.time()
    print(t1-t0)

    print('done')
