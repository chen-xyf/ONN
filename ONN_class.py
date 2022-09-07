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
from datetime import datetime


class MyONN:

    def __init__(self, batch_size, num_batches, num_epochs, w1_0, w2_0, b1_0, lr, dimensions, save_folder,
                 trainx=None, valx=None, testx=None,
                 trainy=None, valy=None, testy=None,
                 forward='digital', backward='digital'):

        if forward == 'optical':
            self.ctrl = Controller(*dimensions)

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
        self.valx = valx
        self.testx = testx
        self.trainy = trainy
        self.valy = valy
        self.testy = testy

        self.batch_indxs_list = None
        self.save_folder = save_folder

        self.forward = forward
        self.backward = backward

        self.sm_scaling = 5.


        ##### initialise dnn parameters #####

        m_dw1 = np.zeros((self.n - 1, self.m))
        v_dw1 = np.zeros((self.n - 1, self.m))

        m_db1 = np.zeros(self.m)
        v_db1 = np.zeros(self.m)

        m_dw2 = np.zeros((self.m, self.k))
        v_dw2 = np.zeros((self.m, self.k))

        beta1 = 0.9
        beta2 = 0.999

        adam_params = (m_dw1, v_dw1, m_db1, v_db1, m_dw2, v_dw2, beta1, beta2)

        self.dnn = DNN_backprop(w1_0, w2_0, b1_0, lr, *adam_params)

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2, self.axsa, self.axsd],
                    [self.axs3, self.axs4, self.axs5, self.axsb, self.axse],
                    [self.axs6, self.axs7, self.axs8, self.axsc, self.axsf]] = plt.subplots(3, 5, figsize=(24, 12))

        self.axsa.set_title('Loss')
        self.axsa.set_xlim(0, self.num_batches*self.num_epochs)
        self.axsa.set_ylim(0, 0.6)

        self.axsb.set_title('Accuracy')
        self.axsb.set_xlim(0, self.num_epochs)
        self.axsb.set_ylim(0, 100)

        self.axsc.set_title('Errors')
        self.axsc.set_xlim(0, self.num_batches*self.num_epochs)
        self.axsc.set_ylim(0, 0.5)

        self.loss_plot = self.axsa.plot(self.loss, linestyle='-', marker='', c='b')
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='x', c='b')
        self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
        self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')
        self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')

        self.axsd.set_title('Test actual classes')
        self.label_scatter = self.axsd.scatter(valx[:, 0], valx[:, 1], c=valy.argmax(axis=1))
        self.axse.set_title('Test prediction classes')
        self.pred_scatter = self.axse.scatter(valx[:, 0], valx[:, 1])
        self.axsf.set_title('Correct predictions')
        self.correct_scatter = self.axsf.scatter(valx[:, 0], valx[:, 1])

        self.axs3.set_ylim(-3, 3)
        self.axs4.set_ylim(-3, 3)
        self.axs5.set_ylim(-3, 3)

        self.th_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line1 = self.axs3.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        self.th_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='o', c='b')[0]
        self.meas_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='x', c='r')[0]
        self.label_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='o', c='g')[0]
        self.softmax_line2 = self.axs4.plot(np.zeros(self.k), linestyle='', marker='x', c='orange')[0]

        self.th_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='o', c='b')[0]
        self.meas_line3 = self.axs5.plot(np.zeros(self.m), linestyle='', marker='x', c='r')[0]

        self.img1 = self.axs6.imshow(np.zeros((80, 672)), aspect='auto', vmin=0, vmax=255)
        self.img2 = self.axs7.imshow(np.zeros((90, 808)), aspect='auto', vmin=0, vmax=255)
        self.img3 = self.axs8.imshow(np.zeros((80, 672)), aspect='auto', vmin=0, vmax=255)

        plt.draw()
        plt.pause(2)

        self.loop_clock.tick()

    def run_calibration(self, initial):

        meass_a1 = []
        theorys_a1 = []
        meass_z2 = []
        theorys_z2 = []
        meass_back = []
        theorys_back = []

        if not initial:
            slm_arr = cp.empty((self.n, self.m))
            slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
            slm_arr[0, :] = cp.array(self.dnn.b1.copy())
            slm_arr = cp.clip(slm_arr, -1, 1)
            slm_arr = (slm_arr*64).astype(cp.int)/64

            slm2_arr = cp.array(self.dnn.w2.copy())
            slm2_arr = cp.clip(slm2_arr, -1, 1)
            slm2_arr = (slm2_arr*64).astype(cp.int)/64

            self.ctrl.update_slm1(slm_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)
            time.sleep(0.5)

        succeeded = 0
        fail = 0
        while succeeded < 5:

            # print(succeeded)
            cp.random.seed(succeeded)

            dmd_vecs = cp.random.normal(0.5, 0.4, (self.ctrl.num_frames, self.n))
            dmd_vecs[:, 0] = 1.  # cp.linspace(0., 1., 5)[succeeded]
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            dmd_errs = cp.random.normal(0., 0.5, (self.ctrl.num_frames, self.k))
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
            success, err = self.ctrl.check_ampls(calib=True)
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

                # if not initial:
                #     meas_a1 = self.ctrl.normalise_ampls(ampls=meas_a1, norm_params=self.ctrl.norm_params1)
                #     meas_z2 = self.ctrl.normalise_ampls(ampls=meas_z2, norm_params=self.ctrl.norm_params2)
                #     meas_back = self.ctrl.normalise_ampls(ampls=meas_back, norm_params=self.ctrl.norm_params3)
                    # meas_a1 *= np.sign(theory_a1)
                    # meas_z2 *= np.sign(theory_z2)
                    # meas_back *= np.sign(theory_back)

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
                theory_a1_plot -= theory_a1_plot.mean()
                theory_a1_plot /= theory_a1_plot.std()
                meas_a1_plot = meas_a1[0, :].copy()
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

            else:
                fail += 1
                if fail == 3:
                    fail = 0
                    continue

        meas_a1 = np.array(meass_a1).reshape(succeeded * self.ctrl.num_frames, self.m)
        theory_a1 = np.array(theorys_a1).reshape(succeeded * self.ctrl.num_frames, self.m)
        meas_z2 = np.array(meass_z2).reshape(succeeded * self.ctrl.num_frames, self.k)
        theory_z2 = np.array(theorys_z2).reshape(succeeded * self.ctrl.num_frames, self.k)
        meas_back = np.array(meass_back).reshape(succeeded * self.ctrl.num_frames, self.m)
        theory_back = np.array(theorys_back).reshape(succeeded * self.ctrl.num_frames, self.m)

        # if initial:
        self.ctrl.norm_params1 = self.ctrl.find_norm_params(theory_a1, meas_a1)
        self.ctrl.norm_params2 = self.ctrl.find_norm_params(theory_z2, meas_z2)
        self.ctrl.norm_params3 = self.ctrl.find_norm_params(theory_back, meas_back)

        meas_a1 = (meas_a1 - self.ctrl.norm_params1[:, 1].copy()) / self.ctrl.norm_params1[:, 0].copy()
        meas_z2 = (meas_z2 - self.ctrl.norm_params2[:, 1].copy()) / self.ctrl.norm_params2[:, 0].copy()
        meas_back = (meas_back - self.ctrl.norm_params3[:, 1].copy()) / self.ctrl.norm_params3[:, 0].copy()

        meas_a1 *= np.sign(theory_a1)
        meas_z2 *= np.sign(theory_z2)
        meas_back *= np.sign(theory_back)

        # if not initial:
        #     self.ctrl.norm_params1 = self.ctrl.update_norm_params(theory_a1, meas_a1, self.ctrl.norm_params1)
        #     self.ctrl.norm_params2 = self.ctrl.update_norm_params(theory_z2, meas_z2, self.ctrl.norm_params2)
        #     self.ctrl.norm_params3 = self.ctrl.update_norm_params(theory_back, meas_back, self.ctrl.norm_params3)


        error1 = (meas_a1 - theory_a1).std()
        error2 = (meas_z2 - theory_z2).std()
        error3 = (meas_back - theory_back).std()

        if initial:
            print(colored(f'error1 : {error1:.3f}, signal1 : {theory_a1.std():.3f}, '
                          f'ratio1 : {theory_a1.std()/error1:.3f}', 'blue'))
            print(colored(f'error2 : {error2:.3f}, signal2 : {theory_z2.std():.3f}, '
                          f'ratio2 : {theory_z2.std()/error2:.3f}', 'blue'))
            print(colored(f'error3 : {error3:.3f}, signal3 : {theory_back.std():.3f}, '
                          f'ratio3 : {theory_back.std()/error3:.3f}', 'blue'))

        self.axs0.cla()
        low = -2
        high = 2
        self.axs0.set_ylim(low, high)
        self.axs0.set_xlim(low, high)
        self.a1_scatter = self.axs0.plot(theory_a1, meas_a1, linestyle='', marker='x')
        self.axs0.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs0.set_title('layer 1')
        self.axs0.set_xlabel('theory')
        self.axs0.set_ylabel('measured')

        self.axs1.cla()
        low = -2
        high = 2
        self.axs1.set_ylim(low, high)
        self.axs1.set_xlim(low, high)
        self.z2_scatter = self.axs1.plot(theory_z2, meas_z2, linestyle='', marker='x')
        self.axs1.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs1.set_title('layer 2')
        self.axs1.set_xlabel('theory')
        self.axs1.set_ylabel('measured')

        self.axs2.cla()
        low = -2
        high = 2
        self.axs2.set_ylim(low, high)
        self.axs2.set_xlim(low, high)
        self.back_scatter = self.axs2.plot(theory_back, meas_back, linestyle='', marker='x')
        self.axs2.plot([low, high], [low, high], c='black', linewidth=1)
        self.axs2.set_title('backprop')
        self.axs2.set_xlabel('theory')
        self.axs2.set_ylabel('measured')

        plt.draw()
        # plt.pause(1)
        plt.show()

    def init_weights(self):

        slm_arr = cp.empty((self.n, self.m))
        slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
        slm_arr[0, :] = cp.array(self.dnn.b1.copy())

        slm2_arr = cp.array(self.dnn.w2.copy())

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

        self.theory_a1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
        self.theory_z2 = np.dot(self.theory_a1, self.dnn.w2.T)

        if self.forward == 'digital':
            self.dnn.a1 = self.theory_a1.copy()
            self.dnn.z2 = self.theory_z2.copy()

        elif self.forward == 'optical':

            self.meas_a1 = np.empty((self.batch_size, self.m))
            self.meas_z2 = np.empty((self.batch_size, self.k))

            dmd_vecs = cp.ones((self.batch_size, self.n))
            dmd_vecs[:, 1:] = cp.array(xs)
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            num_repeats = self.batch_size//self.ctrl.num_frames
            # only want to run 10 frames at a time on the DMD. If batch size is greater than 10,
            # then run multiple sets of frames

            # We allow 5 attempts at running the batch. After that, skip the batch.
            ii = 0
            fails = 0
            while ii < num_repeats:

                self.ctrl.run_batch(dmd_vecs[ii*self.ctrl.num_frames:(ii+1)*self.ctrl.num_frames, :])
                success, err = self.ctrl.check_ampls(a1=True, z2=True, back=False)
                if not success:
                    fails += 1
                    if fails == 5:
                        print(colored('FAILED BATCH', 'red'))
                        return False, err
                    else:
                        continue
                else:
                    self.meas_a1[ii*self.ctrl.num_frames:(ii+1)*self.ctrl.num_frames, :] = self.ctrl.ampls1.copy()
                    self.meas_z2[ii*self.ctrl.num_frames:(ii+1)*self.ctrl.num_frames, :] = self.ctrl.ampls2.copy()
                    ii += 1

            self.meas_a1 = (self.meas_a1 - self.ctrl.norm_params1[:, 1].copy()) / self.ctrl.norm_params1[:, 0].copy()
            self.meas_a1 *= np.sign(self.theory_a1)
            self.dnn.a1 = self.meas_a1.copy()

            self.meas_z2 = (self.meas_z2 - self.ctrl.norm_params2[:, 1].copy()) / self.ctrl.norm_params2[:, 0].copy()
            self.meas_z2 *= np.sign(self.theory_z2)
            self.dnn.z2 = self.meas_z2.copy()

            error1 = (self.meas_a1 - self.theory_a1).std() #/np.abs(self.meas_a1 - self.theory_a1).mean()
            error2 = (self.meas_z2 - self.theory_z2).std() #/np.abs(self.meas_z2 - self.theory_z2).mean()
            self.errors1.append(error1)
            self.errors2.append(error2)
            # print(colored(f'error1 : {error1:.3f}', 'blue'))
            # print(colored(f'error2 : {error2:.3f}', 'blue'))

        ########################################################
        # Now calculate error vector and perform backward MVM  #
        ########################################################

        self.dnn.a2 = softmax(self.dnn.z2*self.sm_scaling)

        self.dnn.loss = error(self.dnn.a2, self.dnn.ys)

        self.dnn.a2_delta = (self.dnn.a2 - self.dnn.ys) #/self.batch_size
        self.dnn.a2_delta = self.dnn.a2_delta/np.abs(self.dnn.a2_delta).max()

        self.theory_back = np.dot(self.dnn.a2_delta, self.dnn.w2)

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

        if self.forward == 'optical':

            slm_arr = cp.empty((self.n, self.m))
            slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
            slm_arr[0, :] = cp.array(self.dnn.b1.copy())

            slm2_arr = cp.array(self.dnn.w2.copy())

            self.ctrl.update_slm1(slm_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)

        if self.dnn.loss < self.min_loss:
            self.min_loss = self.dnn.loss
            self.best_w1 = self.dnn.w1.copy()
            self.best_w2 = self.dnn.w2.copy()
            self.best_b1 = self.dnn.b1.copy()

        self.loss.append(self.dnn.loss)
        # print(colored('loss : {:.2f}'.format(self.dnn.loss), 'green'))
        self.loss_plot.pop(0).remove()
        self.loss_plot = self.axsa.plot(self.loss, linestyle='-', marker='', c='b')

        return True, None

    def run_validation(self, epoch, test_or_val='val'):

        if test_or_val == 'val':
            xs = self.valx.copy()
            num_val_batches = 10
            ys = self.valy.copy()
            label = ys.argmax(axis=1)

        elif test_or_val == 'test':
            gridx1 = np.linspace(0., 1., 20).repeat(20)
            gridx2 = np.tile(np.linspace(0., 1., 20), 20)
            xs = np.empty((20**2, 2))
            xs[:, 0] = gridx1
            xs[:, 1] = gridx2
            num_val_batches = 40

        else:
            raise ValueError

        if self.forward == 'digital':
            a1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
            z2 = np.dot(a1, self.dnn.w2.T)
            a2 = softmax(z2*self.sm_scaling)
            pred = a2.argmax(axis=1)

        elif self.forward == 'optical':

            xs_arr = xs.reshape((num_val_batches, self.ctrl.num_frames, xs.shape[1]))
            meas_a1_raw = np.empty((num_val_batches, self.ctrl.num_frames, self.m))
            meas_z2_raw = np.empty((num_val_batches, self.ctrl.num_frames, self.k))

            batch = 0
            while batch < num_val_batches:

                # print(batch)

                dmd_vecs = cp.ones((self.ctrl.num_frames, self.n))
                dmd_vecs[:, 1:] = cp.array(xs_arr[batch])
                dmd_vecs = cp.clip(dmd_vecs, 0, 1)
                dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

                self.ctrl.run_batch(dmd_vecs)
                success, err = self.ctrl.check_ampls(a1=True, z2=True, back=False)
                # if not success:
                #     print('oh no, trying again')

                if success:
                    meas_a1_raw[batch, ...] = self.ctrl.ampls1.copy()
                    meas_z2_raw[batch, ...] = self.ctrl.ampls2.copy()

                    # self.th_line1.set_ydata(theory_a1[0, :])
                    # self.meas_line1.set_ydata(meas_a1[0, :])
                    # self.th_line2.set_ydata(theory_z2[0, :])
                    # self.meas_line2.set_ydata(meas_z2[0, :])

                    batch += 1

            meas_a1_raw = np.reshape(meas_a1_raw, (num_val_batches*self.ctrl.num_frames, self.m))
            meas_a1 = (meas_a1_raw - self.ctrl.norm_params1[:, 1].copy()) / self.ctrl.norm_params1[:, 0].copy()
            theory_a1 = np.dot(xs, self.dnn.w1) + self.dnn.b1
            meas_a1 *= np.sign(theory_a1)

            meas_z2_raw = np.reshape(meas_z2_raw, (num_val_batches*self.ctrl.num_frames, self.k))
            meas_z2 = (meas_z2_raw - self.ctrl.norm_params2[:, 1].copy()) / self.ctrl.norm_params2[:, 0].copy()
            theory_z2 = np.dot(theory_a1, self.dnn.w2.T)
            meas_z2 *= np.sign(theory_z2)

            pred = softmax(meas_z2*self.sm_scaling).argmax(axis=1)

        if test_or_val == 'test':
            np.save(self.save_folder+f'test_grid_predictions.npy', pred)
            xs0, ys0 = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
            self.axse.contourf(xs0, ys0, pred.reshape((20, 20)).T)

        if test_or_val == 'val':
            acc = accuracy(pred, label)
            self.accs.append(acc)
            np.save(self.save_folder+f'accuracies.npy', np.array(self.accs))
            np.save(self.save_folder+f'validation/predictions/predictions_epoch{epoch}.npy', pred)
            np.save(self.save_folder+f'validation/labels/labels_epoch{epoch}.npy', label)

            if self.forward == 'optical':
                np.save(self.save_folder+f'validation/raw_ampls1/raw_ampls_1_epoch{epoch}.npy', meas_a1_raw)
                np.save(self.save_folder+f'validation/raw_ampls2/raw_ampls_2_epoch{epoch}.npy', meas_z2_raw)
                np.save(self.save_folder+f'validation/meas_a1/meas_a1_epoch{epoch}.npy', meas_a1)
                np.save(self.save_folder+f'validation/meas_z2/meas_z2_epoch{epoch}.npy', meas_z2)
                np.save(self.save_folder+f'validation/theory_a1/theory_a1_epoch{epoch}.npy', theory_a1)
                np.save(self.save_folder+f'validation/theory_z2/theory_z2_epoch{epoch}.npy', theory_z2)



            self.label_scatter.remove()
            self.label_scatter = self.axsd.scatter(xs[:, 0], xs[:, 1], c=label)

            self.pred_scatter.remove()
            self.pred_scatter = self.axse.scatter(xs[:, 0], xs[:, 1], c=pred)

            correct = pred == label
            self.correct_scatter.remove()
            self.correct_scatter = self.axsf.scatter(xs[:, 0], xs[:, 1], c=correct, cmap='RdYlGn')

            self.accs_plot.pop(0).remove()
            self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='x', c='b')
            plt.draw()
            plt.pause(0.1)

    def save_batch(self, epoch, batch):

        if self.forward == 'optical':
            np.save(self.save_folder+f'training/frames1/frames1_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.frames1)
            np.save(self.save_folder+f'training/frames2/frames2_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.frames2)

            np.save(self.save_folder+f'training/raw_ampls1/raw_ampls1_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.ampls1)
            np.save(self.save_folder+f'training/raw_ampls2/raw_ampls2_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.ampls2)

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
                    self.ctrl.ampls3)
            np.save(self.save_folder+f'training/frames3/frames3_epoch{epoch}_batch{batch}.npy',
                    self.ctrl.frames3)

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

        self.th_line1.set_ydata(self.theory_a1[frame, :])
        self.meas_line1.set_ydata(self.meas_a1[frame, :])

        self.th_line2.set_ydata(self.theory_z2[frame, :])
        self.meas_line2.set_ydata(self.meas_z2[frame, :])
        self.softmax_line2.set_ydata(self.dnn.a2[frame, :])
        self.label_line2.set_ydata(self.dnn.ys[frame, :])

        [self.a1_scatter.pop(0).remove() for _ in range(self.m)]
        self.a1_scatter = self.axs0.plot(self.theory_a1, self.meas_a1, linestyle='', marker='x')
        [self.z2_scatter.pop(0).remove() for _ in range(self.k)]
        self.z2_scatter = self.axs1.plot(self.theory_z2, self.meas_z2, linestyle='', marker='x')

        self.img1.set_array(self.ctrl.frames1[frame])
        self.img2.set_array(self.ctrl.frames2[frame])

        self.err1_plot.pop(0).remove()
        self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
        self.err2_plot.pop(0).remove()
        self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')

        if self.backward == 'optical':

            self.th_line3.set_ydata(self.theory_back[frame, :])
            self.meas_line3.set_ydata(self.meas_back[frame, :])

            [self.back_scatter.pop(0).remove() for _ in range(self.m)]
            self.back_scatter = self.axs2.plot(self.theory_back, self.meas_back, linestyle='', marker='x')

            self.img3.set_array(self.ctrl.frames3[frame])

            self.err3_plot.pop(0).remove()
            self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')


if __name__ == "__main__":

    n, m, k = 3, 10, 3

    batch_size = 10
    num_batches = 10
    num_epochs = 20
    lr = 0.05

    slm_arr = np.random.normal(0, 0.5, (n, m))
    slm_arr = np.clip(slm_arr, -1, 1)
    slm_arr = (slm_arr*64).astype(np.int)/64

    slm2_arr = np.random.normal(0, 0.5, (k, m))
    slm2_arr = np.clip(slm2_arr, -1, 1)
    slm2_arr = (slm2_arr*64).astype(np.int)/64

    onn = MyONN(batch_size=batch_size, num_batches=num_batches, num_epochs=num_epochs,
                w1_0=slm_arr[1:, :], w2_0=slm2_arr, b1_0=slm_arr[0, :],
                lr=lr, dimensions=(n, m, k),
                save_folder=None,
                trainx=None, testx=None, trainy=None, testy=None,
                forward='optical', backward='optical')

    onn.run_calibration(initial=True)

    plt.show(block=True)

    print('done')
