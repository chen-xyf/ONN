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
from tqdm import trange


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
        self.epoch_accs = []
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
        self.axsb.set_xlim(0, self.num_epochs*self.num_batches)
        self.axsb.set_ylim(0, 100)

        self.axsc.set_title('Errors')
        self.axsc.set_xlim(0, self.num_epochs*self.num_batches)
        self.axsc.set_ylim(0, 1.)

        self.loss_plot = self.axsa.plot(self.loss, linestyle='-', marker='', c='b')
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='x', c='b')
        self.accs_plot_epoch = self.axsb.plot(self.epoch_accs, linestyle='-', marker='x', c='g')
        self.err1_plot = self.axsc.plot(np.array(self.errors1), linestyle='-', marker='')
        self.err2_plot = self.axsf.plot(np.array(self.errors2), linestyle='-', marker='')
        # self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')

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
        low = -0.5
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
        low = -0.5
        high = 4.5
        self.axs1.set_ylim(low, high)
        self.axs1.set_xlim(low, high)
        self.z2_scatter = self.axs1.plot(self.theory_z2, self.meas_z2, linestyle='', marker='x')
        self.axs1.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs1.set_title('layer 2')
        self.axs1.set_xlabel('theory')
        self.axs1.set_ylabel('measured')

        plt.tight_layout()

        plt.draw()
        plt.pause(2)
        plt.show()

        self.loop_clock.tick()

    def init_weights(self):

        slm_arr = cp.empty((self.n, self.m))*0.
        slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
        slm_arr[0, :] = cp.array(self.dnn.b1.copy())

        slm2_arr = cp.zeros((self.k, self.m))*0.
        slm2_arr[2:4, :] = cp.array(self.dnn.w2.copy())

        self.ctrl.update_slm1(slm_arr, lut=True)
        self.ctrl.update_slm2(slm2_arr, lut=True)
        time.sleep(1)

    def run_batch(self, epoch_num, batch_num):

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
                    time.sleep(0.1)
                    continue

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

                error1 = (self.meas_a1 - self.theory_a1).std(axis=0)
                error2 = (self.meas_z2 - self.theory_z2).std(axis=0)

                if np.any(error1 > 0.2) or np.any(error2 > 0.2):
                    print('error too large')
                    success = False
                    fails += 1
                    if fails == 5:
                        print(colored('FAILED BATCH', 'red'))
                        return False
                    time.sleep(0.1)
                    continue

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
            num_val_batches = (self.res**2)//self.batch_size
            folder = 'boundary'
        else:
            xs = self.testx.copy()
            num_val_batches = xs.shape[0]//self.batch_size
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

            t = trange(num_val_batches, position=0,
                       desc=f"{epoch:2d}"+" "*6+f"-.---"+" "*2+f"{0.000:.3f}"+ " "*2+"--.-"+" "*5,
                       leave=True)

            for batch in t:

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

                dt = self.loop_clock.tick()
                t.set_description(f"{epoch:2d}"+" "*6+f"-.---"+" "*2+f"{dt:.3f}"+ " "*2+"--.-"+" "*5)

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

            if epoch == 999:
                pass

            else:

                label = self.testy.copy().argmax(axis=1)

                np.save(self.save_folder+f'{folder}/predictions/pred_epoch{epoch}.npy', pred)
                np.save(self.save_folder+f'{folder}/labels/labels_epoch{epoch}.npy', label)

                acc = (pred == label).sum()*100/self.testx.shape[0]
                self.epoch_accs.append(acc)

                np.save(self.save_folder+f'epoch_accuracies.npy', np.array(self.epoch_accs))

                self.accs_plot_epoch.pop(0).remove()
                self.accs_plot_epoch = self.axsb.plot(np.arange(epoch+1)*self.num_batches, self.epoch_accs,
                                                      linestyle='-', marker='x', c='r')

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
        np.save(self.save_folder+f'accuracies.npy', np.array(self.accs))

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
                self.z2_scatter = self.axs1.plot(self.theory_z2, self.meas_z2, linestyle='', marker='x')
            except AttributeError:
                pass

            self.img1.set_array(self.ctrl.captures['a1'].frames[frame])
            self.img2.set_array(self.ctrl.captures['z2'].frames[frame])

            # try:
                # [self.err1_plot.pop(0).remove() for _ in range(self.m)]

            self.axsc.cla()
            self.axsf.cla()
            self.err1_plot = self.axsc.plot(np.array(self.errors1), linestyle='-', marker='')
                # [self.err2_plot.pop(0).remove() for _ in range(2)]
            self.err2_plot = self.axsf.plot(np.array(self.errors2), linestyle='-', marker='')
            # except:
            #     pass

        self.accs_plot.pop(0).remove()
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='', c='b', linewidth=0.5)

        # plt.pause(0.1)

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
