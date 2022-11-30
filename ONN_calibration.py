from ANN import satab
import matplotlib.pyplot as plt
import time
import numpy as np
import cupy as cp
from termcolor import colored
from glumpy.app import clock
from ONN_config import Controller
from scipy.optimize import curve_fit
import pyautogui


class MyONN:

    def __init__(self, bsize, dimensions):

        self.ctrl = Controller(*dimensions, _num_frames=batch_size, use_pylons=True, use_ueye=True)

        self.n = dimensions[0]
        self.m = dimensions[1]
        self.k = dimensions[2]
        self.batch_size = bsize
        self.loop_clock = clock.Clock()

        self.errors1 = []
        self.errors2 = []
        self.errors3 = []

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2, self.axsa, self.axsd],
                    [self.axs3, self.axs4, self.axs5, self.axsb, self.axse],
                    [self.axs6, self.axs7, self.axs8, self.axsc, self.axsf]] = plt.subplots(3, 5, figsize=(24, 12))

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1920, 0, 1920, 1080)
        mngr.window.showMaximized()

        self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
        self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')
        self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')

        self.axs3.set_ylim(-3, 3)
        self.axs4.set_ylim(-3, 3)
        self.axs5.set_ylim(-3, 3)

        self.th_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        self.th_line2 = self.axs4.plot(np.zeros(5), linestyle='', marker='o', c='b')[0]
        self.meas_line2 = self.axs4.plot(np.zeros(5), linestyle='', marker='x', c='r')[0]

        self.th_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        self.img1 = self.axs6.imshow(np.zeros((80, 672)), aspect='auto', vmin=0, vmax=255)
        self.img2 = self.axs7.imshow(np.zeros((86, 720)), aspect='auto', vmin=0, vmax=255)
        self.img3 = self.axs8.imshow(np.zeros((80, 672)), aspect='auto', vmin=0, vmax=255)

        self.axs0.cla()
        low = -3
        high = 3
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        # self.a1_scatter = self.axs0.plot(self.theory_a1, self.meas_a1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

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
        slm2_arr = cp.random.normal(0., 0.2, (self.k, self.m))
        slm2_arr = cp.clip(slm2_arr, -1., 1.)
        slm2_arr = (slm2_arr*64).astype(cp.int)/64

        # slm2_arr *= 0.
        # slm2_arr += 1.

        return slm2_arr

    def wait_for_user(self, message, pos=18):
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

                self.th_line2.set_ydata(theory_z2_plot)
                self.meas_line2.set_ydata(raw_z2_plot)

                plt.pause(0.1)

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

        od_fit, thresh_fit = curve_fit(satab, theory_z1.flatten(), meas_a1.flatten(), p0=[1, 1], sigma=None,
                                       absolute_sigma=False, check_finite=True, bounds=([0, 0], [1000, 10]),
                                       method=None, jac=None)[0]

        np.save("./tools/satab_params.npy", np.array([od_fit, thresh_fit]))

        print(colored(f"optical depth: {od_fit:.2f}, threshold: {thresh_fit:.2f}", 'green'))

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


if __name__ == "__main__":

    n, m, k = 3, 10, 5

    batch_size = 5

    onn = MyONN(bsize=batch_size, dimensions=(n, m, k))

    onn.run_calibration_linear()
    onn.run_calibration_nonlinear_layer1()
    onn.run_calibration_nonlinear_layer2()

    plt.show(block=True)

    print('done')
