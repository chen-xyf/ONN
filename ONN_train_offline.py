import time
import numpy as np
from termcolor import colored
from ONN_class import MyONN
from sklearn.datasets import make_classification
from datetime import datetime
import os
from tqdm import trange
from sklearn import datasets
from skimage.transform import rescale, resize


################################
# Classification problem data  #
################################

# x, y = make_classification(n_samples=500, n_features=24, n_informative=20, n_redundant=4, n_repeated=0, n_classes=3,
#                            n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=2.0, hypercube=True, shift=0.0,
#                            scale=1.0, shuffle=True, random_state=10)
#
# y_onehot = np.zeros((500, 3))
# for i in range(500):
#     y_onehot[i, int(y[i])] = 1
# trainy = y_onehot[:400, :].copy()
# valy = y_onehot[400:500, :].copy()
# # testy = y_onehot[500:, :].copy()
# testy = None
#
# xnorm = x.copy()
# for i in range(24):
#     xnorm[:, i] -= xnorm[:, i].min()
#     xnorm[:, i] *= 1./xnorm[:, i].max()
# trainx = xnorm[:400, :].copy()
# valx = xnorm[400:500, :].copy()
# # testx = xnorm[500:, :].copy()
# testx = None

x, y = datasets.load_digits(n_class=3, return_X_y=True)

rng = np.random.default_rng(10)
rng.shuffle(x)
rng = np.random.default_rng(10)
rng.shuffle(y)

x = x[:500]
y = y[:500]

x = np.array([resize(x[i].reshape((8, 8)), (5, 5)).reshape(25) for i in range(500)])

y_onehot = np.zeros((500, 3))
for i in range(500):
    y_onehot[i, int(y[i])] = 1
trainy = y_onehot[:400, :].copy()
valy = y_onehot[400:, :].copy()

trainx = x[:400, :].copy()/16
valx = x[400:, :].copy()/16

#######################
# Network parameters  #
#######################

n, m, k = 26, 10, 3

batch_size = 40
num_batches = 10
num_epochs = 15
lr = 0.005

np.random.seed(0)

slm_arr = np.random.normal(0, 0.5, (n, m))
slm_arr = np.clip(slm_arr, -1, 1)
slm_arr = (slm_arr*64).astype(np.int)/64

slm2_arr = np.random.normal(0, 0.5, (k, m))
slm2_arr = np.clip(slm2_arr, -1, 1)
slm2_arr = (slm2_arr*64).astype(np.int)/64

today_date = datetime.today().strftime('%Y-%m-%d')
save_folder = 'D:/ONN_backprop/'+today_date+'/offline_digits2/'

if os.path.exists(save_folder+'NOTES.txt'):
    print('Folder already exists!')
    raise RuntimeError
else:
    with open(save_folder+'NOTES.txt', mode="w") as f:
        f.write("OPTICAL TRAINING")

np.save(save_folder + 'trainx.npy', trainx)
np.save(save_folder + 'valx.npy', valx)
# np.save(save_folder + 'testx.npy', testx)
np.save(save_folder + 'trainy.npy', trainy)
np.save(save_folder + 'valy.npy', valy)
# np.save(save_folder + 'testy.npy', testy)

all_params = (n, m, k, batch_size, num_batches, num_epochs, lr)
np.save(save_folder + 'all_params.npy', all_params)

#####################
# Optical training  #
#####################

onn = MyONN(batch_size=batch_size, num_batches=num_batches, num_epochs=num_epochs,
            w1_0=slm_arr[1:, :], w2_0=slm2_arr, b1_0=slm_arr[0, :],
            lr=lr, dimensions=(n, m, k),
            save_folder=save_folder,
            trainx=trainx, valx=valx, testx=None,
            trainy=trainy, valy=valy, testy=None,
            forward='optical', backward='digital')


onn.run_calibration_linear(initial=True)
onn.init_weights()

# onn.run_validation(99)

print('Epoch  Loss   Time   Accuracy')
time.sleep(0.2)

for epoch in range(num_epochs):

    rng = np.random.default_rng(epoch)
    epoch_rand_indxs = np.arange(onn.trainx.shape[0])
    rng.shuffle(epoch_rand_indxs)
    onn.batch_indxs_list = [epoch_rand_indxs[i * onn.batch_size: (i + 1) * onn.batch_size]
                            for i in range(onn.num_batches)]

    t = trange(num_batches, position=0,
               desc=f"{0:2d}"+" "*6+f"{0.000:.3f}"+" "*2+f"{0.000:.3f}"+ " "*2+"--.-"+" "*5,
               leave=True)

    for batch in t:

        onn.forward = 'digital'
        onn.run_batch(batch)
        onn.save_batch(epoch, batch)
        dt = onn.loop_clock.tick()
        t.set_description(f"{epoch:2d}"+" "*6+f"{onn.loss[-1]:.3f}"+" "*2+f"{dt:.3f}"
                          + " "*2+"--.-"+" "*5)

        if batch == num_batches - 1:
            onn.forward = 'optical'
            onn.dnn.w1 = onn.best_w1.copy()
            onn.dnn.w2 = onn.best_w2.copy()
            onn.dnn.b1 = onn.best_b1.copy()
            np.save(onn.save_folder+f'validation/best_w1s/w1_epoch{epoch}.npy',
                    onn.dnn.w1)
            np.save(onn.save_folder+f'validation/best_b1s/b1_epoch{epoch}.npy',
                    onn.dnn.b1)
            np.save(onn.save_folder+f'validation/best_w2s/w2_epoch{epoch}.npy',
                    onn.dnn.w2)

            # onn.run_calibration(initial=False)

            # if epoch == num_epochs - 1:
            #     onn.run_validation(epoch, test_or_val='test')

            onn.run_validation(epoch)
            t.set_description(f"{epoch:2d}"+" "*6+f"{onn.loss[-1]:.3f}"+" "*2+f"{dt:.3f}"
                              + " "*2+f"{onn.accs[-1]:04.1f}"+" "*5)

    # t1 = time.perf_counter()
    # epoch_time = t1 - t0
    #
    # print('\n######################################################################')
    # print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
    #               .format(epoch, epoch_time, onn.accs[-1], onn.loss[-1]), 'green'))
    # print('######################################################################\n')

print('done')
