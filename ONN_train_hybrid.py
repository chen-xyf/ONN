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

seed = 1

np.random.seed(seed)

trainx = np.load('./datasets/rainbow/trainx.npy')
trainy = np.load('./datasets/rainbow/trainy.npy')

np.random.seed(seed)
np.random.shuffle(trainx)
np.random.seed(seed)
np.random.shuffle(trainy)

# testx = trainx[:80, :].copy()
# trainx = trainx[80:, :].copy()
# testy = trainy[:80, :].copy()
# trainy = trainy[80:, :].copy()

w1_0 = np.random.uniform(0., 1., (2, 10))
w1_0 = np.clip(w1_0, -1, 1)

b1_0 = np.random.uniform(0., 1., 10)
b1_0 = np.clip(b1_0, -1, 1)

w2_0 = np.random.uniform(0., 1., (10, 2))
w2_0 = np.clip(w2_0, -1, 1)
w2_0 = w2_0.T

#######################
# Network parameters  #
#######################

n, m, k = 3, 10, 5

batch_size = 40
num_batches = 20
num_epochs = 15
lr = 0.008
scaling = 100

today_date = datetime.today().strftime('%Y-%m-%d')
save_folder = 'D:/ONN_nonlinear/'+today_date+'/digital3/'

# if os.path.exists(save_folder+'NOTES.txt'):
#     print('Folder already exists!')
#     raise RuntimeError
# else:
#     with open(save_folder+'NOTES.txt', mode="w") as f:
#         f.write("HYBRID TRAINING")

np.save(save_folder + 'trainx.npy', trainx)
np.save(save_folder + 'trainy.npy', trainy)
# np.save(save_folder + 'testx.npy', testx)
# np.save(save_folder + 'testy.npy', testy)

testx = None
testy = None

all_params = (n, m, k, batch_size, num_batches, num_epochs, lr, scaling)
np.save(save_folder + 'all_params.npy', all_params)

#####################
# Optical training  #
#####################

onn = MyONN(batch_size=batch_size, num_batches=num_batches, num_epochs=num_epochs,
            w1_0=w1_0, w2_0=w2_0, b1_0=b1_0,
            lr=lr, scaling=scaling, dimensions=(n, m, k),
            save_folder=save_folder,
            trainx=trainx, testx=testx, trainy=trainy, testy=testy,
            forward='digital', backward='digital')

# onn.init_weights()

# rng = np.random.default_rng(0)
# epoch_rand_indxs = np.arange(onn.trainx.shape[0])
# rng.shuffle(epoch_rand_indxs)
# onn.batch_indxs_list = [epoch_rand_indxs[i * onn.batch_size: (i + 1) * onn.batch_size]
#                         for i in range(onn.num_batches)]
# success = onn.run_batch(0)
# onn.norm_params2 = onn.ctrl.update_norm_params(onn.theory_z2, onn.meas_z2, onn.norm_params2)

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

        success = onn.run_batch(batch)

        if success:

            onn.save_batch(epoch, batch)
            onn.graph_batch()
            dt = onn.loop_clock.tick()

            acc = f"{onn.accs[-1]:4.1f}" if epoch > 0 else "--.-"
            t.set_description(f"{epoch:2d}"+" "*6+f"{onn.loss[-1]:.3f}"+" "*2+f"{dt:.3f}"
                              + " "*2+acc+" "*5)

    # if epoch > 3:
    #     onn.norm_params2 = onn.ctrl.update_norm_params(onn.theory_z2, onn.meas_z2, onn.norm_params2)

    # onn.run_validation(epoch)

    onn.run_validation(epoch, grid=True)

print('done')
input()
