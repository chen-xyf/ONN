import time
import numpy as np
from termcolor import colored
from ONN_class import MyONN
from sklearn.datasets import make_classification
from datetime import datetime
import os
import matplotlib.pyplot as plt
from tqdm import trange

################################
# Classification problem data  #
################################

x, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=6)

y_onehot = np.zeros((500, 3))
for i in range(500):
    y_onehot[i, int(y[i])] = 1
trainy = y_onehot[:400, :].copy()
valy = y_onehot[400:500, :].copy()
# testy = y_onehot[500:, :].copy()
testy = None

xnorm = x.copy()
xnorm[:, 0] -= xnorm[:, 0].min()
xnorm[:, 0] *= 1./xnorm[:, 0].max()
xnorm[:, 1] -= xnorm[:, 1].min()
xnorm[:, 1] *= 1./xnorm[:, 1].max()
trainx = xnorm[:400, :].copy()
valx = xnorm[400:500, :].copy()
# testx = xnorm[500:, :].copy()
testx = None

#######################
# Network parameters  #
#######################

n, m, k = 3, 10, 3

batch_size = 40
num_batches = 10
num_epochs = 20
lr = 0.01

np.random.seed(0)

slm_arr = np.random.normal(0, 0.5, (n, m))
slm_arr = np.clip(slm_arr, -1, 1)
slm_arr = (slm_arr*64).astype(np.int)/64

slm2_arr = np.random.normal(0, 0.5, (k, m))
slm2_arr = np.clip(slm2_arr, -1, 1)
slm2_arr = (slm2_arr*64).astype(np.int)/64

today_date = datetime.today().strftime('%Y-%m-%d')
save_folder = 'D:/ONN_backprop/'+today_date+'/digital_run_1/'

# if os.path.exists(save_folder+'NOTES.txt'):
#     print('Folder already exists!')
#     raise RuntimeError
# else:
#     with open(save_folder+'NOTES.txt', mode="w") as f:
#         f.write("OPTICAL TRAINING")

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
            trainx=trainx, valx=valx, testx=testx,
            trainy=trainy, valy=valy, testy=testy,
            forward='digital', backward='digital')

onn.run_validation(0)

print('Epoch  Loss   Time   Accuracy')
time.sleep(0.2)

for epoch in range(1, num_epochs+1):

    rng = np.random.default_rng(epoch)
    epoch_rand_indxs = np.arange(onn.trainx.shape[0])
    rng.shuffle(epoch_rand_indxs)
    onn.batch_indxs_list = [epoch_rand_indxs[i * onn.batch_size: (i + 1) * onn.batch_size]
                            for i in range(onn.num_batches)]

    t = trange(num_batches, position=0, desc=f"epoch {epoch}", leave=True)

    for batch in t:

        onn.run_batch(batch)
        onn.save_batch(epoch, batch)
        dt = onn.loop_clock.tick()
        t.set_description(f"{epoch:2d}"+" "*6+f"{onn.loss[-1]:.3f}"+" "*2+f"{dt:.3f}"
                          + " "*2+"--.-"+" "*5)
        # time.sleep(0.1)

        if batch == num_batches - 1:
            onn.dnn.w1 = onn.best_w1.copy()
            onn.dnn.w2 = onn.best_w2.copy()
            onn.dnn.b1 = onn.best_b1.copy()
            np.save(onn.save_folder+f'validation/best_w1s/w1_epoch{epoch}.npy',
                    onn.dnn.w1)
            np.save(onn.save_folder+f'validation/best_b1s/b1_epoch{epoch}.npy',
                    onn.dnn.b1)
            np.save(onn.save_folder+f'validation/best_w2s/w2_epoch{epoch}.npy',
                    onn.dnn.w2)

            if epoch == num_epochs:
                onn.run_validation(epoch, test_or_val='test')

            onn.run_validation(epoch, test_or_val='val')
            t.set_description(f"{epoch:2d}"+" "*6+f"{onn.loss[-1]:.3f}"+" "*2+f"{dt:.3f}"
                              + " "*2+f"{onn.accs[-1]:04.1f}"+" "*5)



plt.show(block=True)
print('done')