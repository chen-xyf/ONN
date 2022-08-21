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

    def __init__(self, num_batches, num_epochs, w1_0, w2_0, b1_0, lr, ctrl_vars, save_folder,
                 trainx=None, testx=None, trainy=None, testy=None,
                 forward='digital', backward='digital'):

        if forward == 'optical':
            self.ctrl = Controller(*ctrl_vars)

        self.n = ctrl_vars[0]
        self.m = ctrl_vars[1]
        self.k = ctrl_vars[2]
        self.batch_size = ctrl_vars[3]
        self.num_frames = ctrl_vars[3]
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.loop_clock = clock.Clock()

        self.accs = [0.]
        self.loss = []
        self.min_loss = 10

        self.errors1 = []
        self.errors2 = []
        self.errors3 = []

        self.best_w1 = None
        self.best_w2 = None
        self.best_b1 = None

        self.trainx = trainx
        self.testx = testx
        self.trainy = trainy
        self.testy = testy

        self.batch_indxs_list = None
        self.save_folder = save_folder

        self.forward = forward
        self.backward = backward


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
                    [self.axs6, self.axs7, self.axs8, self.axsc, self.axsf]] = plt.subplots(3, 5, figsize=(12, 6))

        self.axsa.set_ylim(0, 0.5)
        self.axsa.set_title('Loss')
        self.axsb.set_ylim(0, 100)
        self.axsb.set_title('Accuracy')
        self.axsc.set_ylim(0, 2)
        self.axsc.set_title('Errors')

        self.loss_plot = self.axsa.plot(self.loss, linestyle='-', marker='x', c='b')
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='', c='b')
        self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
        self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')
        self.err3_plot = self.axsc.plot(self.errors3, linestyle='-', marker='', c='b')

        self.axsd.set_title('Test actual classes')
        self.axsd.scatter(testx[:, 0], testx[:, 1], c=testy.argmax(axis=1))
        self.axse.set_title('Test prediction classes')
        self.test_scatter = self.axse.scatter(testx[:, 0], testx[:, 1])
        self.axsf.set_title('Correct predictions')
        self.correct_scatter = self.axsf.scatter(testx[:, 0], testx[:, 1])

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

        self.loop_clock.tick()

    def run_calibration(self, initial=True):

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
            slm2_arr = cp.array(self.dnn.w2.copy())
            self.ctrl.update_slm1(slm_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)
            time.sleep(0.5)

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

        meas_a1 = np.array(meass_a1).reshape(5 * self.num_frames, self.m)
        theory_a1 = np.array(theorys_a1).reshape(5 * self.num_frames, self.m)
        meas_z2 = np.array(meass_z2).reshape(5 * self.num_frames, self.k)
        theory_z2 = np.array(theorys_z2).reshape(5 * self.num_frames, self.k)
        meas_back = np.array(meass_back).reshape(5 * self.num_frames, self.m)
        theory_back = np.array(theorys_back).reshape(5 * self.num_frames, self.m)

        # if initial:
        self.ctrl.norm_params1 = self.ctrl.find_norm_params(theory_a1, meas_a1)
        self.ctrl.norm_params2 = self.ctrl.find_norm_params(theory_z2, meas_z2)
        self.ctrl.norm_params3 = self.ctrl.find_norm_params(theory_back, meas_back)
        # else:
        #     self.ctrl.norm_params1 = self.ctrl.update_norm_params(theory_a1, meas_a1, self.ctrl.norm_params1)
        #     self.ctrl.norm_params2 = self.ctrl.update_norm_params(theory_z2, meas_z2, self.ctrl.norm_params2)
        #     self.ctrl.norm_params3 = self.ctrl.update_norm_params(theory_back, meas_back, self.ctrl.norm_params3)

        meas_a1 = (meas_a1 - self.ctrl.norm_params1[:, 1].copy()) / self.ctrl.norm_params1[:, 0].copy()
        meas_z2 = (meas_z2 - self.ctrl.norm_params2[:, 1].copy()) / self.ctrl.norm_params2[:, 0].copy()
        meas_back = (meas_back - self.ctrl.norm_params3[:, 1].copy()) / self.ctrl.norm_params3[:, 0].copy()

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

    def run_batch(self, batch_num):

        # print('Batch ', batch_num)

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

            dmd_vecs = cp.ones((self.num_frames, self.n))
            dmd_vecs[:, 1:] = cp.array(xs)
            dmd_vecs = cp.clip(dmd_vecs, 0, 1)
            dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

            self.ctrl.run_batch(dmd_vecs)
            success = self.ctrl.check_ampls(a1=True, z2=True, back=False)
            if not success:
                print('oh no')
                return False

            self.meas_a1 = self.ctrl.ampls1.copy()
            self.meas_a1 = (self.meas_a1 - self.ctrl.norm_params1[:, 1].copy()) / self.ctrl.norm_params1[:, 0].copy()
            self.meas_a1 *= np.sign(self.theory_a1)
            self.dnn.a1 = self.meas_a1.copy()

            self.meas_z2 = self.ctrl.ampls2.copy()
            self.meas_z2 = (self.meas_z2 - self.ctrl.norm_params2[:, 1].copy()) / self.ctrl.norm_params2[:, 0].copy()
            self.meas_z2 *= np.sign(self.theory_z2)
            self.dnn.z2 = self.meas_z2.copy()

            error1 = (self.meas_a1 - self.theory_a1).std()/np.abs(self.meas_a1 - self.theory_a1).mean()
            error2 = (self.meas_z2 - self.theory_z2).std()/np.abs(self.meas_z2 - self.theory_z2).mean()
            self.errors1.append(error1)
            self.errors2.append(error2)
            # print(colored(f'error1 : {error1:.3f}', 'blue'))
            # print(colored(f'error2 : {error2:.3f}', 'blue'))

        ########################################################
        # Now calculate error vector and perform backward MVM  #
        ########################################################

        self.dnn.a2 = softmax(self.dnn.z2)

        self.dnn.loss = error(self.dnn.a2, self.dnn.ys)

        self.dnn.a2_delta = (self.dnn.a2 - self.dnn.ys) #/self.batch_size
        self.dnn.a2_delta = self.dnn.a2_delta/np.abs(self.dnn.a2_delta).max()

        self.theory_back = np.dot(self.dnn.a2_delta, self.dnn.w2)

        if self.backward == 'digital':
            self.dnn.a1_delta = self.theory_back.copy()

        elif self.backward == 'optical':

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

            self.meas_back = self.ctrl.ampls3.copy()
            self.meas_back = (self.meas_back - self.ctrl.norm_params3[:, 1].copy()) / self.ctrl.norm_params3[:, 0].copy()
            self.meas_back *= np.sign(self.theory_back)
            self.dnn.a1_delta = self.meas_back.copy()

            error3 = (self.meas_back - self.theory_back).std()/np.abs(self.meas_back - self.theory_back).mean()
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

        return True

    def run_validation(self, epoch):

        self.dnn.w1 = self.best_w1.copy()
        self.dnn.w2 = self.best_w2.copy()
        self.dnn.b1 = self.best_b1.copy()

        if self.forward == 'digital':
            a1 = np.dot(self.testx.copy(), self.dnn.w1) + self.dnn.b1
            z2 = np.dot(a1, self.dnn.w2.T)
            a2 = softmax(z2)
            pred = a2.argmax(axis=1)
            label = self.testy.copy().argmax(axis=1)

        elif self.forward == 'optical':

            slm_arr = cp.empty((self.n, self.m))
            slm_arr[1:, :] = cp.array(self.dnn.w1.copy())
            slm_arr[0, :] = cp.array(self.dnn.b1.copy())

            slm2_arr = cp.array(self.dnn.w2.copy())

            self.ctrl.update_slm1(slm_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)
            time.sleep(1)

            xs = self.testx.copy().reshape((10, 10, self.testx.shape[1]))

            all_meas = np.empty((10, 10))

            batch = 0
            while batch < 10:

                print(batch)

                dmd_vecs = cp.ones((self.num_frames, self.n))
                dmd_vecs[:, 1:] = cp.array(xs[batch])
                dmd_vecs = cp.clip(dmd_vecs, 0, 1)
                dmd_vecs = (dmd_vecs * self.ctrl.dmd_block_w).astype(cp.int)/self.ctrl.dmd_block_w

                self.ctrl.run_batch(dmd_vecs)
                success = self.ctrl.check_ampls(a1=True, z2=True, back=False)
                if not success:
                    print('oh no, trying again')

                else:
                    meas_a1 = self.ctrl.ampls1.copy()
                    meas_a1 = (meas_a1 - self.ctrl.norm_params1[:, 1].copy()) / self.ctrl.norm_params1[:, 0].copy()
                    theory_a1 = np.dot(xs[batch], self.dnn.w1) + self.dnn.b1
                    meas_a1 *= np.sign(theory_a1)

                    meas_z2 = self.ctrl.ampls2.copy()
                    meas_z2 = (meas_z2 - self.ctrl.norm_params2[:, 1].copy()) / self.ctrl.norm_params2[:, 0].copy()
                    theory_z2 = np.dot(theory_a1, self.dnn.w2.T)
                    meas_z2 *= np.sign(theory_z2)

                    all_meas[batch, ...] = softmax(meas_z2).argmax(axis=1)

                    self.th_line1.set_ydata(theory_a1[0, :])
                    self.meas_line1.set_ydata(meas_a1[0, :])
                    self.th_line2.set_ydata(theory_z2[0, :])
                    self.meas_line2.set_ydata(meas_z2[0, :])

                    batch += 1

            pred = all_meas.reshape(100)
            label = self.testy.copy().argmax(axis=1)

        acc = accuracy(pred, label)
        self.accs.append(acc)

        self.test_scatter.remove()
        self.test_scatter = self.axse.scatter(self.testx[:, 0], self.testx[:, 1], c=pred)

        correct = pred == label
        self.correct_scatter.remove()
        self.correct_scatter = self.axsf.scatter(self.testx[:, 0], self.testx[:, 1], c=correct, cmap='RdYlGn')

        self.accs_plot.pop(0).remove()
        self.accs_plot = self.axsb.plot(self.accs, linestyle='-', marker='', c='b')
        plt.draw()
        plt.pause(0.1)

        np.save(self.save_folder+f'predictions/predictions_epoch{epoch}.npy', pred)
        np.save(self.save_folder+f'labels/labels_epoch{epoch}.npy', label)
        np.save(self.save_folder+f'accuracies.npy', np.array(self.accs))

    def save_batch(self, epoch, batch):

        np.save(self.save_folder+f'frames1/frames1_epoch{epoch}_batch{batch}.npy',
                self.ctrl.frames1)
        np.save(self.save_folder+f'frames2/frames2_epoch{epoch}_batch{batch}.npy',
                self.ctrl.frames2)
        np.save(self.save_folder+f'frames3/frames3_epoch{epoch}_batch{batch}.npy',
                self.ctrl.frames3)

        np.save(self.save_folder+f'raw_ampls1/raw_ampls1_epoch{epoch}_batch{batch}.npy',
                self.ctrl.ampls1)
        np.save(self.save_folder+f'raw_ampls2/raw_ampls2_epoch{epoch}_batch{batch}.npy',
                self.ctrl.ampls2)
        np.save(self.save_folder+f'raw_ampls3/raw_ampls3_epoch{epoch}_batch{batch}.npy',
                self.ctrl.ampls3)

        np.save(self.save_folder+f'meas_a1/meas_a1_epoch{epoch}_batch{batch}.npy',
                self.meas_a1)
        np.save(self.save_folder+f'meas_z2/meas_z2_epoch{epoch}_batch{batch}.npy',
                self.meas_z2)
        np.save(self.save_folder+f'meas_back/meas_back_epoch{epoch}_batch{batch}.npy',
                self.meas_back)

        np.save(self.save_folder+f'theory_a1/theory_a1_epoch{epoch}_batch{batch}.npy',
                self.theory_a1)
        np.save(self.save_folder+f'theory_z2/theory_z2_epoch{epoch}_batch{batch}.npy',
                self.theory_z2)
        np.save(self.save_folder+f'theory_back/theory_back_epoch{epoch}_batch{batch}.npy',
                self.theory_back)

        np.save(self.save_folder+f'xs/xs_epoch{epoch}_batch{batch}.npy',
                self.dnn.xs)
        np.save(self.save_folder+f'ys/ys_epoch{epoch}_batch{batch}.npy',
                self.dnn.ys)
        np.save(self.save_folder+f'w1s/w1_epoch{epoch}_batch{batch}.npy',
                self.dnn.w1)
        np.save(self.save_folder+f'b1s/b1_epoch{epoch}_batch{batch}.npy',
                self.dnn.b1)
        np.save(self.save_folder+f'w2s/w2_epoch{epoch}_batch{batch}.npy',
                self.dnn.w2)

        np.save(self.save_folder+f'loss.npy', np.array(self.loss))

    def graph_batch(self):

        self.th_line1.set_ydata(self.theory_a1[0, :])
        self.meas_line1.set_ydata(self.meas_a1[0, :])
        self.th_line2.set_ydata(self.theory_z2[0, :])
        self.meas_line2.set_ydata(self.meas_z2[0, :])
        self.th_line3.set_ydata(self.theory_back[0, :])
        self.meas_line3.set_ydata(self.meas_back[0, :])

        self.img1.set_array(self.ctrl.frames1[0])
        self.img2.set_array(self.ctrl.frames2[0])
        self.img3.set_array(self.ctrl.frames3[0])

        self.err1_plot.pop(0).remove()
        self.err1_plot = self.axsc.plot(self.errors1, linestyle='-', marker='', c='r')
        self.err2_plot.pop(0).remove()
        self.err2_plot = self.axsc.plot(self.errors2, linestyle='-', marker='', c='g')
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

    onn = MyONN(num_batches=num_batches, num_epochs=num_epochs,
                w1_0=slm_arr[1:, :], w2_0=slm2_arr, b1_0=slm_arr[0, :],
                lr=lr, ctrl_vars=(n, m, k, batch_size))

    onn.graphs()

    onn.run_calibration(initial=True)

    plt.show(block=True)

    print('done')
