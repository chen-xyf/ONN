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

        self.w1 = np.empty((self.n, self.m), dtype=np.uint32)
        self.w2 = np.empty((self.k, self.m), dtype=np.uint32)

        self.num_batches = num_batches
        self.loop_clock = clock.Clock()

        self.measured = None
        self.theory = None

    def run_mvm(self):

        meass_z1 = []
        theorys_z1 = []
        meass_z2 = []
        theorys_z2 = []
        meass_back = []
        theorys_back = []
        frames1 = []
        frames2 = []
        frames3 = []

        passed = 0
        failed = 0
        while passed < self.num_batches:

            cp.random.seed(passed)

            dmd_vecs = cp.random.normal(0.5, 0.4, (self.num_frames, self.n))
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            dmd_errs = cp.random.normal(0., 0.5, (self.num_frames, self.k))
            dmd_errs = cp.clip(dmd_errs, -1, 1)
            dmd_errs = (dmd_errs*self.ctrl.dmd_err_block_w).astype(cp.int)/self.ctrl.dmd_err_block_w

            slm_arr = cp.random.normal(0, 0.5, (self.n, self.m))
            slm_arr = cp.clip(slm_arr, -1, 1)
            slm_arr = (slm_arr*64).astype(cp.int)/64

            self.ctrl.update_slm1(slm_arr, lut=True)

            slm2_arr = cp.random.normal(0, 0.5, (self.k, self.m))
            slm2_arr = cp.clip(slm2_arr, -1, 1)
            slm2_arr = (slm2_arr*64).astype(cp.int)/64

            self.ctrl.update_slm2(slm2_arr, lut=True)

            time.sleep(0.5)

            self.ctrl.run_batch_forward(dmd_vecs, dmd_errs, normalisation=False)
            time.sleep(0.1)

            meas_z1 = self.ctrl.ampls1.copy()
            theory_z1 = cp.dot(dmd_vecs, slm_arr)

            meas_z2 = self.ctrl.ampls2.copy()
            theory_z2 = (cp.repeat(theory_z1[:, None, :], k, axis=1) * cp.flip(slm2_arr, axis=1)).sum(axis=-1)

            meas_back = self.ctrl.ampls3.copy()
            theory_back = cp.dot(dmd_errs, slm2_arr)

            if (meas_z1.shape[0] == self.num_frames) and (meas_back.shape[0] == self.num_frames):

                meass_z1.append(meas_z1.get())
                theorys_z1.append(theory_z1.get())
                meass_z2.append(meas_z2.get())
                theorys_z2.append(theory_z2.get())
                meass_back.append(meas_back.get())
                theorys_back.append(theory_back.get())

                frames1.append(self.ctrl.frames1)
                frames2.append(self.ctrl.frames2)
                frames3.append(self.ctrl.frames3)

                passed += 1
            else:
                failed += 1

            if failed > 10:
                raise TimeoutError

        meas_z1 = np.array(meass_z1).reshape(self.num_batches * self.num_frames, self.m)
        theory_z1 = np.array(theorys_z1).reshape(self.num_batches * self.num_frames, self.m)
        meas_z2 = np.array(meass_z2).reshape(self.num_batches * self.num_frames, self.k)
        theory_z2 = np.array(theorys_z2).reshape(self.num_batches * self.num_frames, self.k)
        meas_back = np.array(meass_back).reshape(self.num_batches * self.num_frames, self.m)
        theory_back = np.array(theorys_back).reshape(self.num_batches * self.num_frames, self.m)

        # np.save('./tools/temp_z1s.npy', meas_z1)
        # np.save('./tools/temp_theories1.npy', theory_z1)
        # np.save('./tools/temp_z2s.npy', meas_z2)
        # np.save('./tools/temp_theories2.npy', theory_z2)
        # np.save('./tools/temp_backs.npy', meas_back)
        # np.save('./tools/temp_theories_back.npy', theory_back)

        def line(x, grad, c):
            return (grad * x) + c

        params_z1 = np.array([curve_fit(line, np.abs(theory_z1[:, j]), meas_z1[:, j])[0]
                              for j in range(theory_z1.shape[1])])
        params_z2 = np.array([curve_fit(line, np.abs(theory_z2[:, j]), meas_z2[:, j])[0]
                              for j in range(theory_z2.shape[1])])
        params_back = np.array([curve_fit(line, np.abs(theory_back[:, j]), meas_back[:, j])[0]
                                for j in range(theory_back.shape[1])])

        meas_z1 = (meas_z1 - params_z1[:, 1].copy()) / params_z1[:, 0].copy()
        meas_z2 = (meas_z2 - params_z2[:, 1].copy()) / params_z2[:, 0].copy()
        meas_back = (meas_back - params_back[:, 1].copy()) / params_back[:, 0].copy()

        meas_z1 *= np.sign(theory_z1)
        meas_z2 = meas_z2 * np.sign(theory_z2)
        meas_back *= np.sign(theory_back)

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2],
                    [self.axs3, self.axs4, self.axs5]] = plt.subplots(2, 3, figsize=(24, 12))

        plt.draw()
        plt.pause(0.1)

        self.axs3.imshow(frames1[0][0], aspect='auto', vmin=0, vmax=255)
        self.axs4.imshow(frames2[0][0], aspect='auto', vmin=0, vmax=255)
        self.axs5.imshow(frames3[0][0], aspect='auto', vmin=0, vmax=255)

        self.axs0.cla()
        low = -2
        high = 2
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.axs0.plot(theory_z1, meas_z1, linestyle='', marker='x')
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


        err_z1 = theory_z1-meas_z1
        print(theory_z1.std(), meas_z1.std(), err_z1.std(), theory_z1.std()/err_z1.std())

        err_z2 = theory_z2-meas_z2
        print(theory_z2.std(), meas_z2.std(), err_z2.std(), meas_z2.std()/err_z2.std())

        err_back = theory_back-meas_back
        print(theory_back.std(), meas_back.std(), err_back.std(), meas_back.std()/err_back.std())


        plt.show()
        plt.draw()
        plt.pause(1)

        time.sleep(60)


if __name__ == "__main__":

    n, m, k = 3, 10, 3
    num_frames = 10
    ctrl_vars = (n, m, k, num_frames)

    mvm = MyMVM(num_batches=10, ctrl_vars=ctrl_vars)

    mvm.run_mvm()

    print('done')
